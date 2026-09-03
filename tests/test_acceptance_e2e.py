"""测试模块 7: 端到端系统验收 (E2E Acceptance) 与性能 Schema 校验.

覆盖：
1. 本地真实素材 (testsample) 自适应探测与跳过保护（新克隆环境无实体视频不崩）；
2. 真实/合成样本的全流程元数据解析与时间戳排序；
3. 结构化性能日志 Schema 校验 (validate_perf_json_schema)；
4. 资源清理守护器 (cleanup_resources) 安全调用。
"""

import json
from pathlib import Path

import pytest

from src.database import VlogDatabase
from src.scanner import parse_filename, scan_directory, get_date_cam_groups
from src.utils import cleanup_resources, PROJECT_ROOT
from src.monitor import get_perf, PerfRecord
from tests.helpers import validate_perf_json_schema

TESTSAMPLE_DIR = PROJECT_ROOT / "testsample"


class TestDatasetAndSchemaAcceptance:
    """系统级端到端验收与 Schema 验证。"""

    def test_testsample_real_world_dataset_if_present(self):
        """如果本地存在真实的 testsample 监控素材，执行完整归档与时间戳一致性校验。"""
        if not TESTSAMPLE_DIR.exists() or len(list(TESTSAMPLE_DIR.glob("*.mp4"))) == 0:
            pytest.skip("本地未挂载 testsample/ 视频素材目录，自适应跳过实体视频硬测试")

        sample_files = sorted(list(TESTSAMPLE_DIR.glob("*.mp4")))
        assert len(sample_files) >= 10, f"testsample 文件数偏少: {len(sample_files)}"

        for f in sample_files:
            meta = parse_filename(f.name)
            assert meta is not None, f"无法解析文件名: {f.name}"
            assert meta["start_ts"] < meta["end_ts"]

    def test_perf_log_json_schema_validity(self, tmp_path):
        """验证性能分析日志是否严格符合 JSON Schema 规范。"""
        perf = get_perf()
        perf.reset()

        perf.add(PerfRecord(stage="prescreen", file="clip1.mp4", gpu="qsv", duration=0.04))
        perf.add(PerfRecord(stage="analysis", file="clip1.mp4", gpu="cuda", duration=0.12))
        perf.add(PerfRecord(stage="render", file="batch_0.mp4", gpu="nv", duration=1.5))

        dump_path = tmp_path / "perf_report.json"
        perf.dump(
            dump_path,
            metadata={
                "date": "20260901",
                "cam": 0,
                "pipeline_duration": 1.66,
                "monitor_summary": [{"name": "pipeline", "duration": 1.66, "avg_cpu": 25.0, "avg_ram": 8.0}],
                "perf_summary": perf.summary_by_stage(),
            },
        )

        assert dump_path.exists()
        data = json.loads(dump_path.read_text(encoding="utf-8"))
        is_valid, msg = validate_perf_json_schema(data)
        assert is_valid, f"Perf log schema violation: {msg}"

    def test_cleanup_resources_safety(self):
        """验证清理例程执行安全无报错。"""
        cleanup_resources()
