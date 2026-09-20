"""Vlog companion assets generator: .srt subtitles, .meta.json manifest, and perf dumping."""
import json
import logging
from pathlib import Path
import time
from typing import Any

from src.core.config import LOGS_DIR
from src.core.database import VlogDatabase
from src.core.identity import camera_key
from src.core.utils import size_metrics
from src.hardware.ffmpeg import get_duration
from src.stages.timeline import (
    build_timeline_from_rows,
    compute_display_plans,
    save_timecode_subtitles,
)

logger = logging.getLogger("homevlog")


def dump_perf(
    perf: Any,
    monitor: Any,
    date: str,
    cam_index: int,
    pipeline_duration: float,
    headline: dict | None = None,
    worker_stats: dict | None = None,
) -> None:
    """Dump structured performance metrics JSON to logs/perf."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_dir = LOGS_DIR / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)
        perf_path = perf_dir / f"perf_{date}_cam{cam_index}_{timestamp}.json"
        yolo_sum = perf.yolo_summary() if hasattr(perf, "yolo_summary") else {}
        wait_sum = perf.wait_summary() if hasattr(perf, "wait_summary") else {}
        metadata = {
            "date": date,
            "cam": cam_index,
            "pipeline_duration": round(pipeline_duration, 2),
            "monitor_summary": monitor.stages_data(),
            "perf_summary": perf.summary_by_stage(),
        }
        if worker_stats:
            metadata["worker_stats"] = worker_stats
        if yolo_sum:
            metadata["yolo_summary"] = yolo_sum
        if wait_sum:
            metadata["wait_summary"] = wait_sum
        if headline:
            metadata["headline"] = headline
        perf.dump(perf_path, metadata=metadata)
        perf.reset()
    except Exception as e:
        logger.debug("dump_perf skipped or failed: %s", e)


def build_headline(output_path: Path, total_input_dur: float, elapsed_wall: float) -> dict:
    """汇总单日头条指标：处理倍速、浓缩率、产出体积（性能评估的顶层仪表盘）。"""
    headline: dict = {
        "input_dur_s": round(total_input_dur, 1),
        "wall_s": round(elapsed_wall, 1),
        "speedup_x": round(total_input_dur / max(elapsed_wall, 0.1), 2),
    }
    try:
        headline.update(size_metrics(output_path.stat().st_size))
    except OSError:
        pass
    try:
        out_dur = get_duration(str(output_path))
        if out_dur and out_dur > 0:
            headline["output_dur_s"] = round(out_dur, 1)
            headline["condensation_x"] = round(total_input_dur / out_dur, 1)
    except Exception:
        pass
    return headline


def save_vlog_companion_assets(
    output_path: Path,
    date: str,
    cam_index: int,
    cam_display: str,
    total_files: int,
    total_input_dur: float,
    elapsed_wall: float,
    db: VlogDatabase,
    config: dict,
) -> None:
    """生成同名标准交付资产包：.srt 现实世界时间码字幕 + .meta.json 自描述结构化清单。"""
    try:
        rows = db.get_all_file_tasks_for_date(date, cam_index)
        full_timeline = build_timeline_from_rows(rows, date, config=config, resolve_presence=False)

        # 1. 生成伴随 .srt 字幕（默认开启，可在配置中显式关闭）
        srt_path = output_path.with_suffix(".srt")
        if config.get("render", {}).get("generate_subtitles", True):
            try:
                save_timecode_subtitles(
                    full_timeline,
                    srt_path,
                    rows=rows,
                    base_date=date,
                )
            except Exception as e:
                logger.warning("save_timecode_subtitles failed: %s", e)

        # 2. 生成伴随 .meta.json 结构化清单
        meta_path = output_path.with_suffix(".meta.json")
        vlog_size = output_path.stat().st_size if output_path.exists() else 0
        vlog_dur = get_duration(str(output_path)) or 0.0

        dyn_dur = sum(s.duration for s in full_timeline if getattr(s, "state", "") in ("DYNAMIC", "DYNAMIC_AUDIO"))
        sta_dur = sum(s.duration for s in full_timeline if getattr(s, "state", "") == "STATIC")

        # 提取动态高光片段（用于下游相册/Web 秒级定位）
        highlights = []
        cur_vlog_pos = 0.0
        seg_cfg = config.get("segment", {})
        render_cfg = config.get("render", {})
        presence_cfg = config.get("presence", {})
        micro_cfg = config.get("micro_motion", {})
        plans = compute_display_plans(
            full_timeline,
            static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
            keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
            min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
            max_static_display_duration=seg_cfg.get("max_static_display_duration", 2.0),
            speed_ramping=render_cfg.get("speed_ramping_enabled", True),
            ramp_duration_s=float(render_cfg.get("ramp_duration_s", 1.0)),
            presence_speed_factor=float(presence_cfg.get("speed_factor", 4.0)),
            night_stationary_speed_factor=float(presence_cfg.get("night_speed_factor", 16.0)),
            micro_motion_cruise_speed=float(micro_cfg.get("cruise_speed", 16.0)),
        )
        for seg, (disp_dur, _) in zip(full_timeline, plans):
            start_vlog = cur_vlog_pos
            end_vlog = cur_vlog_pos + disp_dur
            cur_vlog_pos = end_vlog
            if getattr(seg, "state", "") in ("DYNAMIC", "DYNAMIC_AUDIO"):
                highlights.append({
                    "vlog_start_s": round(start_vlog, 2),
                    "vlog_end_s": round(end_vlog, 2),
                    "vlog_duration_s": round(disp_dur, 2),
                    "state": seg.state,
                    "source_file": Path(getattr(seg, "filepath", getattr(seg, "source_file", ""))).name,
                    "max_energy": round(float(getattr(seg, "max_energy", 0.0) or 0.0), 1),
                    "avg_confidence": round(float(getattr(seg, "avg_confidence", 0.0) or 0.0), 2),
                })

        first_fp = rows[0]["filepath"] if rows else ""
        cam_id = rows[0].get("camera_id") if rows and rows[0].get("camera_id") else (camera_key(first_fp, cam_index) if first_fp else f"cam_{cam_index}")

        manifest_data = {
            "version": "1.0",
            "date": date,
            "camera": {
                "id": cam_id,
                "name": cam_display,
                "cam_index": cam_index,
            },
            "metrics": {
                "raw_duration_s": round(total_input_dur, 2),
                "vlog_duration_s": round(vlog_dur, 2),
                "condensation_ratio": round(total_input_dur / max(vlog_dur, 0.1), 2),
                "dynamic_duration_s": round(dyn_dur, 2),
                "static_duration_s": round(sta_dur, 2),
                "total_source_files": total_files,
                **size_metrics(vlog_size),
            },
            "timeline_highlights": highlights,
            "performance": {
                "wall_clock_s": round(elapsed_wall, 2),
                "speedup_x": round(total_input_dur / max(elapsed_wall, 0.1), 2),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        meta_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved vlog companion assets: %s, %s", srt_path.name, meta_path.name)
    except Exception as e:
        logger.warning("Failed to save vlog companion assets for %s: %s", output_path.name, e)
