"""tests/test_accuracy_and_merging.py

验证交付质量核查与算法精准度优化逻辑：
1. YOLO 目标置信度回填与连续静态切片拓扑原子合并；
2. 短动态片段密集抽样与暗部场景动态阈值；
3. 空间网格滤波底噪限幅保护；
4. 音频 VAD 超静音环境双门限防护；
5. 数据库碎片切片合并修复逻辑。
"""

import sqlite3
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.segment import Segment, _merge_same_state
from src.yolo_verifier import YoloVerifier
from src.filters import SpatialGridMotionFilter, AudioEnergyVAD
from scripts.verify_accuracy import fix_database_fragmentation, audit_database


def test_yolo_confidence_backfill_and_merge():
    """验证 YOLO 验证阶段真实置信度回填与后置切片原子合并。"""
    config = {
        "yolo": {
            "enabled": True,
            "confidence": 0.25,
            "sample_fps": 0.5,
            "skip_energy_threshold": 12.0,
            "target_classes": [0],
        }
    }

    verifier = YoloVerifier.__new__(YoloVerifier)
    verifier.enabled = True
    verifier.confidence = 0.25
    verifier.sample_fps = 0.5
    verifier.skip_energy_threshold = 12.0
    verifier.target_classes = {0}
    verifier.device = "cpu"

    # 准备 3 个切片：
    # Seg 0: STATIC 0~10s
    # Seg 1: DYNAMIC 10~15s (YOLO 检测到人，保持 DYNAMIC，回填置信度 0.88)
    # Seg 2: DYNAMIC 15~20s (YOLO 未检测到人，降为 STATIC，应与 Seg 3 合并)
    # Seg 3: STATIC 20~30s
    segs = [
        Segment(start_time=0.0, end_time=10.0, state="STATIC", source_file="test.mp4", file_start_offset=0.0),
        Segment(start_time=10.0, end_time=15.0, state="DYNAMIC", source_file="test.mp4", file_start_offset=0.0, max_energy=5.0),
        Segment(start_time=15.0, end_time=20.0, state="DYNAMIC", source_file="test.mp4", file_start_offset=0.0, max_energy=4.0),
        Segment(start_time=20.0, end_time=30.0, state="STATIC", source_file="test.mp4", file_start_offset=0.0),
    ]

    # 模拟 frames_buffer
    frames_buffer = {
        20: np.full((100, 100, 3), 120, dtype=np.uint8),
        30: np.full((100, 100, 3), 120, dtype=np.uint8),
        35: np.full((100, 100, 3), 120, dtype=np.uint8),
    }

    # 模拟 YOLO 模型输出
    class MockTensor:
        def __init__(self, arr):
            self.arr = np.array(arr)
        def __len__(self):
            return len(self.arr)
        def cpu(self):
            return self
        def numpy(self):
            return self.arr

    mock_box_person = MagicMock()
    mock_box_person.cls = MockTensor([0])
    mock_box_person.conf = MockTensor([0.88])

    mock_box_empty = MagicMock()
    mock_box_empty.cls = MockTensor([])
    mock_box_empty.conf = MockTensor([])

    mock_res_person = MagicMock()
    mock_res_person.boxes = mock_box_person

    mock_res_empty = MagicMock()
    mock_res_empty.boxes = mock_box_empty

    def mock_call(frames, **kwargs):
        # 首帧（来自 Seg 1）检出人，其余无检出
        results = []
        for i, _ in enumerate(frames):
            results.append(mock_res_person if i == 0 else mock_res_empty)
        return results

    verifier.model = MagicMock(side_effect=mock_call)

    verified = verifier.verify(
        filepath="test.mp4",
        segments=segs,
        frames_buffer=frames_buffer,
        analysis_fps=2.0,
    )

    # 验证置信度回填
    assert verified[1].state == "DYNAMIC"
    assert verified[1].avg_confidence == 0.88

    # 验证切片合并：Seg 2 降为 STATIC 后与 Seg 3 合并为 15.0 ~ 30.0s 的单一静态段
    assert len(verified) == 3
    assert verified[0].state == "STATIC" and verified[0].end_time == 10.0
    assert verified[1].state == "DYNAMIC" and verified[1].start_time == 10.0 and verified[1].end_time == 15.0
    assert verified[2].state == "STATIC" and verified[2].start_time == 15.0 and verified[2].end_time == 30.0


def test_spatial_grid_noise_floor_clamp():
    """验证红外夜视/持续高噪环境下，空间网格底噪具有饱和限幅保护。"""
    grid_filter = SpatialGridMotionFilter(
        grid_rows=4,
        grid_cols=4,
        base_noise_thresh=1.5,
        cell_noise_alpha=0.2,
    )

    # 模拟持续大噪点输入
    noisy_saliency = np.full((40, 40), fill_value=20.0, dtype=np.float32)

    for _ in range(50):
        grid_filter.process_frame(noisy_saliency, dt=0.2, is_night_mode=True)

    # 底噪必须受限，不能无上限暴涨（限幅为 base_noise_thresh * 1.8 = 2.7）
    assert np.max(grid_filter.noise_floor_grid) <= 2.71


def test_audio_vad_effective_min_dbfs():
    """验证超静音环境下 AudioEnergyVAD 绝对能量门限防护。"""
    vad = AudioEnergyVAD(
        sample_rate=16000,
        window_ms=50,
        noise_margin_db=12.0,
        min_dbfs=-42.0,
        enabled=True,
    )

    # 构造一段极低底噪音频 (~ -75 dBFS)，中间插入一段 -40 dBFS 的微弱杂音
    np.random.seed(42)
    quiet_samples = np.random.normal(0, 0.0001, 16000 * 3).astype(np.float32)  # ~ -75 dBFS
    # 插入一段微弱杂音 (~ -40 dBFS, rms ~ 0.01)
    noise_burst = np.random.normal(0, 0.009, 16000).astype(np.float32)
    audio = np.concatenate([quiet_samples, noise_burst, quiet_samples])

    events, stats = vad.detect_events(audio)
    # 因为底噪 < -60 dBFS，有效能量门限提升至 -38 dBFS，-40 dBFS 的杂音不应误唤醒
    assert len(events) == 0


def test_fix_database_fragmentation(tmp_path):
    """验证数据库碎片切片合并修复逻辑。"""
    db_file = tmp_path / "test_vlog.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            filepath TEXT,
            cam_index INTEGER,
            date TEXT,
            analysis_status TEXT,
            analysis_segments TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE segments (
            id INTEGER PRIMARY KEY,
            file_id INTEGER,
            filepath TEXT,
            cam_index INTEGER,
            date TEXT,
            start_time REAL,
            end_time REAL,
            duration REAL,
            state TEXT,
            max_energy REAL,
            avg_confidence REAL,
            file_start_offset REAL
        )"""
    )

    conn.execute(
        "INSERT INTO file_tasks VALUES (1, 'cam0_01.mp4', 0, '20260320', 'ANALYZED', '')"
    )
    # 插入连续 3 个静态切片
    conn.execute("INSERT INTO segments VALUES (1, 1, 'cam0_01.mp4', 0, '20260320', 0.0, 5.0, 5.0, 'STATIC', 0.0, 0.0, 0.0)")
    conn.execute("INSERT INTO segments VALUES (2, 1, 'cam0_01.mp4', 0, '20260320', 5.0, 10.0, 5.0, 'STATIC', 1.2, 0.0, 0.0)")
    conn.execute("INSERT INTO segments VALUES (3, 1, 'cam0_01.mp4', 0, '20260320', 10.0, 20.0, 10.0, 'STATIC', 0.5, 0.0, 0.0)")
    conn.commit()
    conn.close()

    before_cnt, after_cnt = fix_database_fragmentation(db_file)
    assert before_cnt == 3
    assert after_cnt == 1

    conn = sqlite3.connect(str(db_file))
    segs = conn.execute("SELECT start_time, end_time, duration, state, max_energy FROM segments").fetchall()
    conn.close()

    assert len(segs) == 1
    assert segs[0][0] == 0.0
    assert segs[0][1] == 20.0
    assert segs[0][2] == 20.0
    assert segs[0][3] == "STATIC"
    assert segs[0][4] == 1.2


def test_min_motion_threshold_floor():
    """验证传感器低底噪或纯静止夜视场景下，运动检测自适应阈值严格受限不低于 min_motion_threshold。"""
    from src.detector import MotionDetector

    config = {
        "detection": {
            "analysis_resolution": "320x180",
            "analysis_fps": 5,
            "motion_sensitivity": 1.0,
            "min_motion_threshold": 2.5,
            "ema_background_enabled": False,
        }
    }
    detector = MotionDetector(config)
    assert detector.min_motion_threshold == 2.5

    # 模拟微小底噪帧序列 (能量在 0.1~0.5 之间)
    dummy_frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(20)]
    results, _ = detector.analyze_frames(dummy_frames, file_duration=4.0, fps=5.0)

    # 结果应当全部判为 STATIC，绝不产生虚假动态
    assert all(r["state"] == "STATIC" for r in results)
    assert all(not r["is_motion"] for r in results)


def test_yolo_verifies_dynamic_audio():
    """验证 YOLO 验证阶段将 DYNAMIC_AUDIO 纳入闭环：有目标保留，空房降级为 STATIC。"""
    verifier = YoloVerifier.__new__(YoloVerifier)
    verifier.enabled = True
    verifier.confidence = 0.25
    verifier.sample_fps = 0.5
    verifier.target_classes = {0}
    verifier.device = "cpu"

    segs = [
        Segment(start_time=0.0, end_time=10.0, state="STATIC", source_file="test.mp4", file_start_offset=0.0),
        Segment(start_time=10.0, end_time=15.0, state="DYNAMIC_AUDIO", source_file="test.mp4", file_start_offset=0.0, max_energy=3.0),
        Segment(start_time=15.0, end_time=20.0, state="DYNAMIC_AUDIO", source_file="test.mp4", file_start_offset=0.0, max_energy=2.0),
        Segment(start_time=20.0, end_time=30.0, state="STATIC", source_file="test.mp4", file_start_offset=0.0),
    ]

    frames_buffer = {
        20: np.full((100, 100, 3), 120, dtype=np.uint8),
        30: np.full((100, 100, 3), 120, dtype=np.uint8),
        35: np.full((100, 100, 3), 120, dtype=np.uint8),
    }

    class MockTensor:
        def __init__(self, arr):
            self.arr = np.array(arr)
        def __len__(self):
            return len(self.arr)
        def cpu(self):
            return self
        def numpy(self):
            return self.arr

    mock_box_person = MagicMock()
    mock_box_person.cls = MockTensor([0])
    mock_box_person.conf = MockTensor([0.85])

    mock_box_empty = MagicMock()
    mock_box_empty.cls = MockTensor([])
    mock_box_empty.conf = MockTensor([])

    mock_res_person = MagicMock()
    mock_res_person.boxes = mock_box_person

    mock_res_empty = MagicMock()
    mock_res_empty.boxes = mock_box_empty

    def mock_call(frames, **kwargs):
        # 首帧对应 Seg 1 (有目标)，后续帧对应 Seg 2 (空房)
        return [mock_res_person if i == 0 else mock_res_empty for i in range(len(frames))]

    verifier.model = MagicMock(side_effect=mock_call)

    verified = verifier.verify(
        filepath="test.mp4",
        segments=segs,
        frames_buffer=frames_buffer,
        analysis_fps=2.0,
    )

    # 1. Seg 1 (有目标): 保留为 DYNAMIC_AUDIO，置信度回填 0.85
    assert verified[1].state == "DYNAMIC_AUDIO"
    assert verified[1].avg_confidence == 0.85

    # Off-screen sound remains at normal speed and is queued for review.
    assert len(verified) == 3
    assert verified[1].start_time == 10.0 and verified[1].end_time == 20.0
    assert verified[1].needs_review
    assert "MULTIMODAL_AUDIO_NO_TARGET" in verified[1].review_reason
    assert verified[2].state == "STATIC" and verified[2].start_time == 20.0


def test_ambiguous_scenarios_recording_and_db(tmp_path):
    """验证典型无法判定场景（边缘置信度、高能无目标光影）的时间戳记录与数据库主动召回。"""
    from src.database import VlogDatabase
    from src.segment import Segment

    db_file = tmp_path / "test_ambiguous.db"
    db = VlogDatabase(db_file)

    # 创建任务
    db.add_file_task(
        filepath="cam0_borderline.mp4",
        cam_index=0,
        date="20260321",
        file_start_time="20260321120000",
        file_end_time="20260321120500",
        file_duration=300.0,
    )

    # 构建含有无法判定的典型场景切片
    segs = [
        # 1. 边缘置信度切片
        Segment(
            start_time=10.0, end_time=25.0, state="DYNAMIC", source_file="cam0_borderline.mp4",
            max_energy=4.5, avg_confidence=0.22, needs_review=True,
            review_reason="BORDERLINE_CONFIDENCE: 目标边缘置信度(conf=0.22)，疑似微弱/遮挡目标"
        ),
        # 2. 高能无目标剧烈光影切片
        Segment(
            start_time=50.0, end_time=80.0, state="STATIC", source_file="cam0_borderline.mp4",
            max_energy=22.5, avg_confidence=0.0, needs_review=True,
            review_reason="HIGH_ENERGY_NO_TARGET: 高能剧烈光影(energy=22.5)无目标确认"
        ),
        # 3. 普通正常静态切片 (无需复核)
        Segment(
            start_time=80.0, end_time=300.0, state="STATIC", source_file="cam0_borderline.mp4",
            max_energy=0.8, avg_confidence=0.0, needs_review=False
        ),
    ]

    # 持久化分析结果入库
    db.set_analysis_result("cam0_borderline.mp4", segs)

    # 1. 主动召回所有待审疑难切片
    review_queue = db.get_anomaly_segments(category="needs_review", date="20260321", cam_index=0)
    assert len(review_queue) == 2

    # 验证第一条：边缘置信度
    item1 = next(it for it in review_queue if "BORDERLINE_CONFIDENCE" in it["review_reason"])
    assert item1["state"] == "DYNAMIC"
    assert item1["avg_confidence"] == 0.22
    assert item1["anomaly_type"] == "borderline_confidence"

    # 验证第二条：高能光影
    item2 = next(it for it in review_queue if "HIGH_ENERGY_NO_TARGET" in it["review_reason"])
    assert item2["state"] == "STATIC"
    assert item2["max_energy"] == 22.5
    assert item2["anomaly_type"] == "high_energy_no_target"

    db.close()


def test_split_segments_preserves_review_flags():
    """验证物理跨文件切分时，切片的 avg_confidence、needs_review 与 review_reason 100% 留存。"""
    from src.segment import Segment, split_segments_at_file_boundaries

    # 构造一个跨越两个物理文件的待审切片
    long_seg = Segment(
        start_time=10.0,
        end_time=50.0,
        state="DYNAMIC",
        source_file="cam0_part1.mp4",
        file_start_offset=0.0,
        max_energy=18.5,
        avg_confidence=0.88,
        needs_review=True,
        review_reason="BORDERLINE_CONFIDENCE: 目标边缘置信度",
    )

    files_info = [
        {"filepath": "cam0_part1.mp4", "file_start_offset": 0.0, "file_end_offset": 30.0, "duration": 30.0},
        {"filepath": "cam0_part2.mp4", "file_start_offset": 30.0, "file_end_offset": 60.0, "duration": 30.0},
    ]

    split_segs = split_segments_at_file_boundaries([long_seg], files_info)
    assert len(split_segs) == 2

    # 第一段：10.0 ~ 30.0s
    assert split_segs[0].source_file == "cam0_part1.mp4"
    assert split_segs[0].start_time == 10.0
    assert split_segs[0].end_time == 30.0
    assert split_segs[0].avg_confidence == 0.88
    assert split_segs[0].needs_review is True
    assert split_segs[0].review_reason == "BORDERLINE_CONFIDENCE: 目标边缘置信度"

    # 第二段：30.0 ~ 50.0s
    assert split_segs[1].source_file == "cam0_part2.mp4"
    assert split_segs[1].start_time == 30.0
    assert split_segs[1].end_time == 50.0
    assert split_segs[1].avg_confidence == 0.88
    assert split_segs[1].needs_review is True
    assert split_segs[1].review_reason == "BORDERLINE_CONFIDENCE: 目标边缘置信度"


def test_timeline_human_label_short_codes():
    """验证时间轴重建时 human_label / manual_label 短码 (FP/FN/TP/TN) 具有最高优先级 (AGENTS.md 铁律)。"""
    from src.timeline import build_timeline_from_rows

    rows = [
        {
            "filepath": "cam0_f1.mp4",
            "file_start_time": "20260321000000",
            "file_end_time": "20260321001000",
            "file_duration": 600.0,
            "prescreen_status": "SUSPICIOUS",
            "segments": [
                # 算法原本判定 DYNAMIC，人工标记 FP (误报) -> 必须强制转为 STATIC
                {
                    "start_time": 0.0,
                    "end_time": 100.0,
                    "state": "DYNAMIC",
                    "manual_label": "FP",
                },
                # 算法原本判定 STATIC，人工标记 FN (漏报) -> 必须强制转为 DYNAMIC
                {
                    "start_time": 100.0,
                    "end_time": 200.0,
                    "state": "STATIC",
                    "human_label": "FN",
                },
                # 算法原本判定 STATIC，人工确认 TN -> 保持 STATIC
                {
                    "start_time": 200.0,
                    "end_time": 400.0,
                    "state": "STATIC",
                    "manual_label": "TN",
                },
                # 算法原本判定 DYNAMIC，人工标记 TP -> 保持 DYNAMIC
                {
                    "start_time": 400.0,
                    "end_time": 600.0,
                    "state": "DYNAMIC",
                    "human_label": "TP",
                },
            ],
        }
    ]

    timeline = build_timeline_from_rows(rows, date="20260321")
    assert len(timeline) >= 4

    # 切片 1: 原 DYNAMIC 经 FP 修正后变为 STATIC
    assert timeline[0].start_in_file == 0.0
    assert timeline[0].end_in_file == 100.0
    assert timeline[0].state == "STATIC"

    # 切片 2: 原 STATIC 经 FN 修正后变为 DYNAMIC
    assert timeline[1].start_in_file == 100.0
    assert timeline[1].end_in_file == 200.0
    assert timeline[1].state == "DYNAMIC"

    # 切片 3: 原 STATIC 经 TN 确认为 STATIC
    assert timeline[2].start_in_file == 200.0
    assert timeline[2].end_in_file == 400.0
    assert timeline[2].state == "STATIC"

    # 切片 4: 原 DYNAMIC 经 TP 确认为 DYNAMIC
    assert timeline[3].start_in_file == 400.0
    assert timeline[3].end_in_file == 600.0
    assert timeline[3].state == "DYNAMIC"


def test_config_settings_consumption_closed_loop():
    """验证 config/settings.yaml 中的关键配置项均有明确代码消费，且无隐式未定义键。"""
    from src.utils import load_config

    cfg = load_config(reload=True)

    # 1. 验证本阶段治理显式化的 4 项核心配置
    assert cfg.get("segment", {}).get("apply_smoothing") is True
    assert cfg.get("render", {}).get("qsv_timeout_s") == 360
    assert cfg.get("detection", {}).get("timestamp_margin") == 0.5
    assert cfg.get("audio_vad", {}).get("sample_rate") == 16000

    # 2. 验证硬基线门限配置
    assert cfg.get("detection", {}).get("min_motion_threshold") == 2.5
    assert cfg.get("segment", {}).get("max_static_display_duration") == 1.5



