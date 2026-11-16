import logging
import time
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("homevlog")

class YoloVerifier:
    _model_lock = threading.Lock()
    _shared_model = None
    _shared_key = None
    # 推理串行锁：共享模型被多个分析线程并发调用，
    # ultralytics predictor 非线程安全，CUDA 上下文必须串行进入
    _inference_lock = threading.Lock()

    def __init__(self, config: dict, device: str | None = None):
        self.config = config
        self.last_telemetry: dict[str, Any] = {}
        yolo_cfg = config.get("yolo", {})
        self.enabled = yolo_cfg.get("enabled", False)
        if not self.enabled:
            return
            
        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            logger.error("ultralytics or torch not installed. YOLO verifier disabled.")
            self.enabled = False
            return
            
        raw_path = yolo_cfg.get("model_path", "models/yolo11n.pt")
        from src.core.config import PROJECT_ROOT
        p = Path(raw_path)
        if not p.is_absolute():
            # 依次探测: 相对当前工作目录 -> 相对 PROJECT_ROOT -> 相对 PROJECT_ROOT/models
            if p.exists():
                model_path = str(p)
            elif (PROJECT_ROOT / p).exists():
                model_path = str(PROJECT_ROOT / p)
            elif (PROJECT_ROOT / "models" / p.name).exists():
                model_path = str(PROJECT_ROOT / "models" / p.name)
            else:
                model_path = str(PROJECT_ROOT / p)
        else:
            model_path = str(p)

        self.target_classes = set(yolo_cfg.get("target_classes", [0, 1, 2, 3, 15, 16]))

        self.confidence = yolo_cfg.get("confidence", 0.25)
        self.batch_size = max(1, int(yolo_cfg.get("batch_size", 4)))
        
        if device is None:
            device = yolo_cfg.get("device", config.get("hardware", {}).get("device", "cpu"))
            
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU for YOLO.")
            device = "cpu"
            
        self.device = device

        # 抑制 ultralytics 模型加载/fuse 的 stdout 摘要噪音，保持终端仪表盘版面整洁
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

        weight = Path(model_path).resolve()
        stat = weight.stat()
        model_key = (str(weight), stat.st_size, stat.st_mtime_ns, device)
        with YoloVerifier._model_lock:
            if YoloVerifier._shared_model is None or YoloVerifier._shared_key != model_key:
                logger.info(f"Loading shared YOLO model {model_path} on {device}...")
                model = YOLO(model_path, task="detect")
                try:
                    model.to(device)
                    model.fuse()
                except Exception as e:
                    logger.debug(f"YOLO model fuse failed: {e}")
                YoloVerifier._shared_model = model
                YoloVerifier._shared_key = model_key
        self.model = YoloVerifier._shared_model
        
    def verify(self, filepath: str, segments: list, gpu: str = "qsv", device: str | None = None,
               frames_buffer: dict | None = None, analysis_fps: float = 5.0,
               labels: list | None = None) -> list:
        """Bounded inference over shared JPEGs. Missing evidence never means a negative result."""
        t_yolo_start = time.monotonic()
        if not self.enabled:
            self.last_telemetry = {
                "yolo_duration": 0.0,
                "yolo_lock_wait": 0.0,
                "yolo_infer_time": 0.0,
                "yolo_frames": 0,
                "yolo_candidate_segments": 0,
                "yolo_confirmed_segments": 0,
                "yolo_rejected_segments": 0,
                "yolo_status": "DISABLED",
            }
            return segments
        import cv2
        import torch
        from bisect import bisect_left, bisect_right
        from src.algorithms.segment import _merge_same_state

        label_times = [float(l.get("time", 0.0)) for l in labels] if labels else []
        frames_buffer = frames_buffer or {}
        keys = sorted(frames_buffer)
        # Inspect every retained candidate. The decoder already bounds this pool.
        # Local windows keep a late positive from labelling an entire long file,
        # and avoid the blind intervals created by eight samples per parent.
        windows = []
        for seg in segments:
            start = max(0, seg.start_time - seg.file_start_offset) * analysis_fps
            end = max(0, seg.end_time - seg.file_start_offset) * analysis_fps
            matched = keys[bisect_left(keys, start):bisect_left(keys, end)]
            if not seg.is_dynamic or len(matched) <= 8:
                windows.append(seg)
                continue
            groups = np.array_split(matched, int(np.ceil(len(matched) / 8)))
            left = seg.start_time
            for n, group in enumerate(groups):
                right = (seg.file_start_offset + (group[-1] + groups[n + 1][0]) / (2 * analysis_fps)
                         if n + 1 < len(groups) else seg.end_time)
                if labels:
                    lo = bisect_left(label_times, left)
                    hi = bisect_right(label_times, right)
                    win_energies = [float(labels[idx].get("energy", 0.0)) for idx in range(lo, hi)]
                    local_max = max(win_energies) if win_energies else seg.max_energy
                    windows.append(replace(seg, start_time=left, end_time=right, max_energy=local_max))
                else:
                    windows.append(replace(seg, start_time=left, end_time=right))
                left = right
        segments = windows
        jobs = []
        counts, hits, confirmed, confidences, max_thresh = {}, {}, {}, {}, {}
        for i, seg in enumerate(segments):
            if not seg.is_dynamic:
                continue
            start = max(0, seg.start_time - seg.file_start_offset) * analysis_fps
            end = max(0, seg.end_time - seg.file_start_offset) * analysis_fps
            matched = keys[bisect_left(keys, start):bisect_left(keys, end)]
            jobs.extend((i, k) for k in matched)
            counts[i], hits[i], confirmed[i], confidences[i], max_thresh[i] = 0, 0, False, 0.0, 0.0
        person_boxes_by_seg: dict[int, list[np.ndarray]] = {i: [] for i in counts}

        batch_size = max(1, int(getattr(self, "batch_size", 4)))
        yolo_lock_wait_total = 0.0
        yolo_infer_total = 0.0
        yolo_frames_total = 0
        try:
            for offset in range(0, len(jobs), batch_size):
                frames, owners, thresholds = [], [], []
                for i, k in jobs[offset:offset+batch_size]:
                    raw = frames_buffer[k]
                    try:
                        if isinstance(raw, (bytes, bytearray)) or getattr(raw, "ndim", 0) == 1:
                            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                            luma_src = img
                        elif getattr(raw, "ndim", 0) == 2:
                            img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                            luma_src = raw
                        else:
                            img = raw
                            luma_src = img
                        if img is None:
                            continue
                        luma = float(np.minimum(np.mean(luma_src), np.median(luma_src)))
                        threshold = min(self.confidence, 0.15 if luma < 50 else 0.22 if luma < 100 else self.confidence)
                        if segments[i].max_energy >= 12.0:
                            threshold = min(threshold, max(0.22, self.confidence - 0.05))
                        frames.append(img)
                        owners.append(i)
                        thresholds.append(threshold)
                    except (ValueError, cv2.error):
                        logger.warning("Invalid cached YOLO frame: %s index %s", filepath, k)
                if not frames:
                    continue
                yolo_frames_total += len(frames)
                t_lock_req = time.monotonic()
                with self._inference_lock:
                    t_lock_acq = time.monotonic()
                    yolo_lock_wait_total += (t_lock_acq - t_lock_req)
                    t_inf_start = time.monotonic()
                    with torch.inference_mode():
                        infer_kwargs = {"conf": min(thresholds), "verbose": False, "stream": True}
                        if self.device.startswith("cuda") and torch.cuda.is_available():
                            infer_kwargs["half"] = True
                        try:
                            results = list(self.model(frames, **infer_kwargs))
                        except TypeError:
                            infer_kwargs.pop("half", None)
                            results = list(self.model(frames, **infer_kwargs))
                    t_inf_end = time.monotonic()
                    yolo_infer_total += (t_inf_end - t_inf_start)
                if len(results) != len(frames):
                    raise RuntimeError("YOLO returned an incomplete batch")
                for i, threshold, result in zip(owners, thresholds, results):
                    counts[i] += 1
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    if hasattr(boxes, "data") and (torch.is_tensor(boxes.data) or isinstance(boxes.data, np.ndarray)):
                        data = boxes.data.cpu().numpy() if hasattr(boxes.data, "cpu") else np.asarray(boxes.data)
                        if len(data) == 0:
                            continue
                        xyxy_arr = data[:, :4]
                        conf_arr = data[:, 4]
                        cls_arr = data[:, 5]
                    else:
                        cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                        conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                        has_xyxy = hasattr(boxes, "xyxy") and (torch.is_tensor(boxes.xyxy) or isinstance(boxes.xyxy, np.ndarray))
                        xyxy_arr = (boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)) if has_xyxy else None
                    orig_shape = getattr(result, "orig_shape", None)
                    h_orig, w_orig = (float(orig_shape[0]), float(orig_shape[1])) if orig_shape and len(orig_shape) >= 2 else (1.0, 1.0)

                    best_person_conf = -1.0
                    best_person_box = None

                    frame_hit = False
                    max_thresh[i] = max(max_thresh[i], float(threshold))
                    for idx, (cls, conf) in enumerate(zip(cls_arr, conf_arr)):
                        c_int = int(cls)
                        if c_int in self.target_classes:
                            confidences[i] = max(confidences[i], float(conf))
                            if float(conf) >= threshold:
                                frame_hit = True
                        if c_int == 0 and float(conf) >= threshold and xyxy_arr is not None and idx < len(xyxy_arr):
                            if float(conf) > best_person_conf:
                                best_person_conf = float(conf)
                                b = xyxy_arr[idx]
                                best_person_box = np.array([
                                    b[0] / max(1.0, w_orig),
                                    b[1] / max(1.0, h_orig),
                                    b[2] / max(1.0, w_orig),
                                    b[3] / max(1.0, h_orig),
                                ], dtype=np.float32)
                    if frame_hit:
                        hits[i] += 1
                    if best_person_box is not None:
                        person_boxes_by_seg[i].append(best_person_box)
        except Exception as exc:
            logger.warning("YOLO verification failed for %s: %s; retaining motion", filepath, exc)
            for i in counts:
                segments[i].needs_review = True
                segments[i].review_reason = "YOLO_FAILED"
            self.last_telemetry = {
                "yolo_duration": round(time.monotonic() - t_yolo_start, 3),
                "yolo_lock_wait": round(yolo_lock_wait_total, 3),
                "yolo_infer_time": round(yolo_infer_total, 3),
                "yolo_frames": yolo_frames_total,
                "yolo_candidate_segments": len(counts),
                "yolo_confirmed_segments": 0,
                "yolo_rejected_segments": 0,
                "yolo_status": "FAILED",
            }
            return segments

        for i, count in counts.items():
            if hits[i] >= 2:
                confirmed[i] = True
            elif hits[i] == 1:
                confirmed[i] = count <= 2 or confidences[i] >= max(0.38, max_thresh[i] + 0.12)

        confirmed_count = sum(1 for c in confirmed.values() if c)
        rejected_count = 0
        for i, count in counts.items():
            seg = segments[i]
            seg.avg_confidence = round(confidences[i], 4) if (confirmed[i] or count < 2) else 0.0
            if count == 0 or (not confirmed[i] and count < 2):
                seg.needs_review = True
                seg.review_reason = "INSUFFICIENT_YOLO_EVIDENCE"
            elif confirmed[i]:
                # 人体空间位移门控 (Human-Centric Motion Gating):
                # 若检测到人体，但采样帧间包围盒位移极小且能量低于剧烈爆发门限 (< 25.0)，且非音频事件，
                # 判定为静坐/静卧陪伴 (PRESENCE 或 NIGHT_STATIONARY)，避免膨胀为 1x DYNAMIC
                p_boxes = person_boxes_by_seg.get(i, [])
                is_stationary_human = False
                if seg.duration >= 8.0 and len(p_boxes) >= 3 and seg.state != "DYNAMIC_AUDIO" and seg.max_energy < 25.0:
                    disps = []
                    ious = []
                    for b1, b2 in zip(p_boxes[:-1], p_boxes[1:]):
                        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                        c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                        disps.append(float(np.hypot(c2[0] - c1[0], c2[1] - c1[1])))
                        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                        ious.append(inter / max(1e-6, area1 + area2 - inter))
                    if max(disps) < 0.08 and min(ious) > 0.55:
                        is_stationary_human = True

                if is_stationary_human:
                    cfg = getattr(self, "config", None) or {}
                    pres_cfg = cfg.get("presence", {})
                    night_hours = tuple(pres_cfg.get("night_hours", [23, 7]))
                    hour = (seg.start_time % 86400) / 3600
                    is_seg_night = (night_hours[0] <= hour < night_hours[1] if night_hours[0] < night_hours[1]
                                    else hour >= night_hours[0] or hour < night_hours[1])
                    if is_seg_night and pres_cfg.get("night_stationary_enabled", True):
                        seg.state = "NIGHT_STATIONARY"
                    else:
                        seg.state = "PRESENCE"
                    seg.needs_review = False
                    seg.review_reason = "STATIONARY_PRESENCE: YOLO主体静止"
                elif confidences[i] < 0.28:
                    seg.needs_review = True
                    seg.review_reason = "BORDERLINE_CONFIDENCE"
            elif seg.state == "DYNAMIC_AUDIO":
                seg.needs_review = True
                seg.review_reason = "MULTIMODAL_AUDIO_NO_TARGET"
            else:
                cfg = getattr(self, "config", None) or {}
                micro_cfg = cfg.get("micro_motion", {})
                pres_cfg = cfg.get("presence", {})
                night_hours = tuple(pres_cfg.get("night_hours", [23, 7]))
                hour = (seg.start_time % 86400) / 3600
                is_seg_night = (night_hours[0] <= hour < night_hours[1] if night_hours[0] < night_hours[1]
                                else hour >= night_hours[0] or hour < night_hours[1])
                if "energy_threshold" in micro_cfg:
                    micro_thresh = float(micro_cfg["energy_threshold"])
                elif is_seg_night and "night_energy_threshold" in micro_cfg:
                    micro_thresh = float(micro_cfg["night_energy_threshold"])
                elif (not is_seg_night) and "day_energy_threshold" in micro_cfg:
                    micro_thresh = float(micro_cfg["day_energy_threshold"])
                elif "min_motion_threshold" in cfg.get("detection", {}):
                    micro_thresh = max(8.0, float(cfg["detection"]["min_motion_threshold"]) * 4.0)
                else:
                    micro_thresh = 16.0
                # 若差分能量显著 (>= micro_thresh)，表明存在真实的物理运动 (如夜间睡眠翻身、遮挡动作)，
                # 严禁将其作为纯静态抽帧丢弃！保全为 MICRO_MOTION 事件状态，防漏检
                if seg.max_energy >= micro_thresh:
                    seg.state = "MICRO_MOTION"
                    seg.needs_review = True
                    seg.review_reason = f"YOLO_NEGATIVE_ENERGY_HIGH: 显著运动未识别目标(energy={seg.max_energy:.1f})"
                else:
                    seg.state = "STATIC"
                    seg.needs_review = True
                    seg.review_reason = "YOLO_NEGATIVE_REQUIRES_AUDIT"
                    rejected_count += 1

        self.last_telemetry = {
            "yolo_duration": round(time.monotonic() - t_yolo_start, 3),
            "yolo_lock_wait": round(yolo_lock_wait_total, 3),
            "yolo_infer_time": round(yolo_infer_total, 3),
            "yolo_frames": yolo_frames_total,
            "yolo_candidate_segments": len(counts),
            "yolo_confirmed_segments": confirmed_count,
            "yolo_rejected_segments": rejected_count,
            "yolo_status": "OK",
        }
        from src.algorithms.segment import resolve_presence_segments
        cfg = getattr(self, "config", None) or {}
        pres_cfg = cfg.get("presence", {})
        if not pres_cfg.get("enabled", True):
            return _merge_same_state(segments, gap_tolerance=1.5)
        presence_gap = float(pres_cfg.get("max_presence_gap_s", 180.0))
        presence_conf = float(pres_cfg.get("person_conf_threshold", 0.20))
        night_enabled = bool(pres_cfg.get("night_stationary_enabled", True))
        night_hours = tuple(pres_cfg.get("night_hours", [23, 7]))
        stationary_energy_max = float(pres_cfg.get("stationary_energy_max", 2.5))
        min_stationary_duration_s = float(pres_cfg.get("min_stationary_duration_s", 180.0))
        merged = _merge_same_state(segments, gap_tolerance=1.5)
        return resolve_presence_segments(
            merged,
            max_presence_gap_s=presence_gap,
            person_conf_threshold=presence_conf,
            night_stationary_enabled=night_enabled,
            night_hours=night_hours,
            stationary_energy_max=stationary_energy_max,
            min_stationary_duration_s=min_stationary_duration_s,
        )

