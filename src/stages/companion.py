"""Vlog companion assets generator: .srt subtitles, .meta.json manifest, and perf dumping."""
import json
import logging
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
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
    display_plan_options,
    save_timecode_subtitles,
)

logger = logging.getLogger("homevlog")


@contextmanager
def promote_vlog_assets(candidate: Path, output: Path, config: dict):
    """Keep the previous delivery until the caller commits its manifest/status.

    Filesystem replacement is atomic per file. Ordinary failures and cancellation
    restore the previous bundle; abandoned backup directories are never auto-cleaned.
    """
    from src.hardware.render_cache import _manifest_path
    sources = [(candidate, output), (candidate.with_suffix(".meta.json"), output.with_suffix(".meta.json"))]
    if config.get("render", {}).get("generate_subtitles", True):
        sources.append((candidate.with_suffix(".srt"), output.with_suffix(".srt")))
    for source, _ in sources:
        if not source.is_file():
            raise FileNotFoundError(f"required delivery asset missing: {source}")
    targets = [output, output.with_suffix(".meta.json"), output.with_suffix(".srt"), _manifest_path(output)]
    backup_dir = Path(tempfile.mkdtemp(prefix=".delivery_backup_", dir=output.parent))
    (backup_dir / "recovery.json").write_text(json.dumps(
        {str(index): str(target.resolve()) for index, target in enumerate(targets)}, indent=2
    ), encoding="utf-8")
    backups = []
    promoted = []
    committing = False
    try:
        for index, target in enumerate(targets):
            if target.exists():
                backup = backup_dir / str(index)
                target.replace(backup)
                backups.append((backup, target))
        for source, target in sources:
            source.replace(target)
            promoted.append(target)
        committing = True
        yield
    except BaseException:
        try:
            for target in promoted:
                target.unlink(missing_ok=True)
            if committing:
                _manifest_path(output).unlink(missing_ok=True)
            for backup, target in backups:
                backup.replace(target)
        except OSError:
            logger.exception("Delivery rollback incomplete; previous assets retained in %s", backup_dir)
            raise
        else:
            shutil.rmtree(backup_dir)
        raise
    else:
        try:
            shutil.rmtree(backup_dir)
        except OSError:
            logger.warning("Delivery committed; could not remove backup directory %s", backup_dir)


def cleanup_delivery_candidate(candidate: Path) -> None:
    for path in (candidate, candidate.with_suffix(".meta.json"), candidate.with_suffix(".srt")):
        try:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove pending delivery asset: %s", path)


def build_batch_summary_item(db, date, cam_index, output_dir, config, wall_s, status, *, skip_render=False):
    """Report this run's work without attributing historical assets to it."""
    from src.stages.scanner import resolve_camera_ui_name, resolve_output_filename
    rows = db.get_all_file_tasks_for_date(date, cam_index)
    sample = rows[0]["filepath"] if rows else None
    camera = resolve_camera_ui_name(str(Path(sample).parent) if sample else "", cam_index=cam_index, config=config)
    item = {
        "date": date, "cam_index": cam_index, "cam_name": camera, "total_files": len(rows),
        "input_duration_s": sum(float(r.get("file_duration") or 0) for r in rows),
        "vlog_duration_s": 0.0, "output_size_mb": 0.0, "wall_clock_s": round(wall_s, 2),
        "condensation_ratio": 0.0, "speedup_x": 0.0, "status": status,
    }
    if status == "SKIPPED":
        item["input_duration_s"] = item["wall_clock_s"] = 0.0
    elif status == "SUCCESS" and not skip_render:
        name = resolve_output_filename(config.get("output", {}).get("naming", "DailyVlog_{date}_{mac}.mp4"),
                                       date, cam_index, sample_filepath=sample, config=config)
        output = Path(output_dir) / name
        try:
            item["output_size_mb"] = round(output.stat().st_size / (1024 * 1024), 2)
            metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
            item["vlog_duration_s"] = metadata.get("metrics", {}).get("vlog_duration_s", 0.0)
            item["condensation_ratio"] = metadata.get("metrics", {}).get("condensation_ratio", 0.0)
            item["speedup_x"] = metadata.get("performance", {}).get("speedup_x", 0.0)
        except (OSError, ValueError):
            logger.warning("Could not read completed delivery summary for %s", output)
    return item


def dump_perf(
    perf: Any,
    monitor: Any,
    date: str,
    cam_index: int,
    pipeline_duration: float,
    headline: dict | None = None,
    worker_stats: dict | None = None,
    run_context: dict | None = None,
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
        if run_context:
            metadata["run_context"] = run_context
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


def build_headline(
    output_path: Path, total_input_dur: float, elapsed_wall: float,
    *, output_duration_s: float | None = None,
) -> dict:
    """汇总单日头条指标：处理倍速、浓缩率、产出体积（性能评估的顶层仪表盘）。"""
    headline: dict = {
        "input_dur_s": round(total_input_dur, 1),
        "wall_s": round(elapsed_wall, 1),
        "speedup_x": round(total_input_dur / max(elapsed_wall, 0.1), 2),
    }
    try:
        from src.hardware.render_cache import file_identity
        headline.update(size_metrics(output_path.stat().st_size))
        headline["output_identity"] = file_identity(output_path)
    except OSError:
        pass
    try:
        out_dur = output_duration_s if output_duration_s is not None else get_duration(str(output_path))
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
    rows: list[dict] | None = None,
    *,
    media_metrics: dict[str, float] | None = None,
) -> bool:
    """生成交付资产；成功后可返回未舍入的媒体测量，供本次报告复用。"""
    try:
        rows = rows if rows is not None else db.get_all_file_tasks_for_date(date, cam_index)
        full_timeline = build_timeline_from_rows(rows, date, config=config, resolve_presence=False)

        # 1. 生成伴随 .srt 字幕（默认开启，可在配置中显式关闭）
        srt_path = output_path.with_suffix(".srt")
        if config.get("render", {}).get("generate_subtitles", True):
            try:
                srt_tmp = srt_path.with_name(srt_path.name + ".tmp")
                save_timecode_subtitles(
                    full_timeline,
                    srt_tmp,
                    rows=rows,
                    base_date=date,
                )
                srt_tmp.replace(srt_path)
            except Exception as e:
                logger.warning("save_timecode_subtitles failed: %s", e)
                return False
        else:
            srt_path.unlink(missing_ok=True)

        # 2. 生成伴随 .meta.json 结构化清单
        meta_path = output_path.with_suffix(".meta.json")
        vlog_size = output_path.stat().st_size if output_path.exists() else 0
        vlog_dur = get_duration(str(output_path)) or 0.0
        if vlog_size <= 0 or vlog_dur <= 0:
            logger.warning("companion assets require a valid media artifact: %s", output_path)
            return False

        dyn_dur = sum(s.duration for s in full_timeline if getattr(s, "state", "") in ("DYNAMIC", "DYNAMIC_AUDIO"))
        sta_dur = sum(s.duration for s in full_timeline if getattr(s, "state", "") == "STATIC")

        # 提取动态高光片段（用于下游相册/Web 秒级定位）
        highlights = []
        cur_vlog_pos = 0.0
        plans = compute_display_plans(full_timeline, **display_plan_options(config))
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
            "skipped_sources": [
                {"source_file": Path(row["filepath"]).name,
                 "reason": "empty_file" if row.get("prescreen_status") == "SKIPPED_EMPTY" else (row.get("error_msg") or "excluded_source"),
                 "size_bytes": 0,
                 "source_start": row["file_start_time"], "source_end": row["file_end_time"],
                 "nominal_duration_s": row.get("file_duration")}
                for row in rows
                if row.get("prescreen_status") in ("SKIPPED_EMPTY", "SKIPPED_EXCLUDED")
                or row.get("analysis_status") in ("SKIPPED_EXCLUDED", "SKIPPED_CORRUPT")
            ],
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
        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        meta_tmp.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
        meta_tmp.replace(meta_path)
        if media_metrics is not None:
            media_metrics["duration_s"] = vlog_dur
        logger.info("Saved vlog companion assets: %s", ", ".join(
            p.name for p in (srt_path, meta_path) if p.exists()
        ))
        return True
    except Exception as e:
        logger.warning("Failed to save vlog companion assets for %s: %s", output_path.name, e)
        return False
