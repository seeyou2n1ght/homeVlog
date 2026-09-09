"""HomeVlog 轻量重渲染与即时秒级浓缩引擎 (Re-render Engine)

用于人工审核打标后，直接基于修正后的 segments 表重新压制成片。
具备前置素材物理存在性与 NAS 网络存储可访问性校验、断点保护与进度追踪。
"""

import time
import logging
import threading
from pathlib import Path
from typing import Any

from src.utils import OUTPUT_DIR, load_config
from src.database import VlogDatabase
from src.timeline import build_timeline, partition_timeline_by_batches
from src.renderer import build_batch_render, concat_output_files, FFmpegProcessRegistry

logger = logging.getLogger("homevlog.rerender")


def check_source_files_accessibility(file_tasks: list[dict]) -> tuple[bool, list[str], list[str]]:
    """前置校验素材文件的物理存在性与可访问性。
    
    返回:
        (can_proceed, valid_files, missing_files)
    """
    valid_files = []
    missing_files = []

    for t in file_tasks:
        fp_str = t.get("filepath")
        if not fp_str:
            continue
        p = Path(fp_str)
        try:
            if p.is_file() and p.stat().st_size > 0:
                valid_files.append(fp_str)
            else:
                missing_files.append(fp_str)
        except Exception:
            missing_files.append(fp_str)

    # 若 100% 缺失 (如 NAS 网络共享断开或路径失效)，硬阻断避免空跑
    if len(valid_files) == 0 and len(missing_files) > 0:
        return False, valid_files, missing_files

    return True, valid_files, missing_files


class ReRenderManager:
    """管理异步重渲染任务的生命周期与进度状态。"""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self.tasks: dict[str, dict[str, Any]] = {}
        self.active_threads: dict[str, threading.Thread] = {}
        self.cancel_flags: dict[str, threading.Event] = {}

    def get_task_key(self, date: str, cam_index: int) -> str:
        return f"{date}_cam{cam_index}"

    def get_status(self, date: str, cam_index: int) -> dict[str, Any]:
        key = self.get_task_key(date, cam_index)
        task = self.tasks.get(key)
        if not task:
            return {"status": "IDLE", "date": date, "cam_index": cam_index}
        return dict(task)

    def cancel_task(self, date: str, cam_index: int) -> bool:
        key = self.get_task_key(date, cam_index)
        event = self.cancel_flags.get(key)
        if event and not event.is_set():
            event.set()
            if key in self.tasks:
                self.tasks[key]["status"] = "CANCELLED"
            FFmpegProcessRegistry.kill_all()
            return True
        return False

    def start_rerender(self, db: VlogDatabase, date: str, cam_index: int, output_version: str = "reviewed") -> dict[str, Any]:
        key = self.get_task_key(date, cam_index)
        
        # 检查是否已有同任务在执行
        with self._lock:
            existing = self.tasks.get(key)
            if existing and existing.get("status") in ("CHECKING", "RENDERING", "CONCATING"):
                return {"error": "Task already running", "status": existing["status"]}

            cancel_event = threading.Event()
            self.cancel_flags[key] = cancel_event

            task_state = {
                "status": "CHECKING",
                "date": date,
                "cam_index": cam_index,
                "progress": 0,
                "current_batch": 0,
                "total_batches": 0,
                "missing_files": [],
                "output_file": "",
                "elapsed_s": 0.0,
                "error": "",
                "start_time": time.time(),
            }
            self.tasks[key] = task_state

        # 启动后台独立渲染线程
        t = threading.Thread(
            target=self._run_render_worker,
            args=(db, date, cam_index, output_version, cancel_event, key),
            daemon=True
        )
        self.active_threads[key] = t
        t.start()

        return {"status": "CHECKING", "message": "Render task initiated"}

    def _run_render_worker(
        self,
        db: VlogDatabase,
        date: str,
        cam_index: int,
        output_version: str,
        cancel_event: threading.Event,
        task_key: str
    ):
        t0 = time.time()
        logger.info("Re-render started for %s cam%d (version=%s)", date, cam_index, output_version)
        config = load_config()

        def update_state(**kwargs):
            if task_key in self.tasks:
                self.tasks[task_key].update(kwargs)
                self.tasks[task_key]["elapsed_s"] = round(time.time() - t0, 1)

        try:
            # 1. 查询当天任务行
            file_tasks = db.get_all_file_tasks_for_date(date, cam_index)
            if not file_tasks:
                update_state(status="FAILED", error="No file tasks found in database for date")
                return

            # 2. 前置物理存在性与可访问性探测
            update_state(status="CHECKING", progress=5)
            can_proceed, valid_files, missing_files = check_source_files_accessibility(file_tasks)
            update_state(missing_files=missing_files)

            if not can_proceed:
                err_msg = f"全部素材文件 ({len(missing_files)} 个) 不可达！请检查 NAS 网络共享或物理存储连接"
                logger.error("Re-render blocked: %s", err_msg)
                update_state(status="FAILED", error=err_msg)
                return

            if missing_files:
                logger.warning(
                    "Re-render %s cam%d: %d files missing, auto-dropping from timeline",
                    date, cam_index, len(missing_files)
                )

            if cancel_event.is_set():
                update_state(status="CANCELLED")
                return

            # 3. 基于修正后的 segments 表构建时间轴
            update_state(status="RENDERING", progress=10)
            valid_set = set(valid_files)
            healthy_rows = [r for r in file_tasks if r["filepath"] in valid_set]

            timeline = build_timeline(db, date, cam_index)
            # 过滤掉缺失文件对应的时间轴片段
            timeline = [t for t in timeline if t.filepath in valid_set]

            if not timeline:
                update_state(status="FAILED", error="Timeline is empty after filtering")
                return

            # 4. 批次切分
            seg_cfg = config.get("segment", {})
            out_cfg = config.get("output", {})
            audio_cfg = out_cfg.get("audio", {})
            render_cfg = config.get("render", {})

            batch_max_files = render_cfg.get("batch_max_files", 8)
            batches = partition_timeline_by_batches(timeline, batch_max_files=batch_max_files)
            total_batches = len(batches)
            update_state(total_batches=total_batches, current_batch=0, progress=15)

            # 5. 渲染输出路径
            out_dir = OUTPUT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            naming = out_cfg.get("naming", "DailyVlog_{date}_cam{index}.mp4")
            base_name = naming.replace("{date}", date).replace("{index}", str(cam_index))
            if output_version:
                stem = Path(base_name).stem
                suffix = Path(base_name).suffix
                final_out_name = f"{stem}_{output_version}{suffix}"
            else:
                final_out_name = base_name

            final_output_path = out_dir / final_out_name

            # 6. 执行批次渲染
            batch_outputs = []
            fps = out_cfg.get("fps", 20)
            from src.utils import parse_res
            width, height = parse_res(out_cfg.get("resolution", "1920x1080"))
            encoder = render_cfg.get("encoder", "nv")

            for bi, b_segs in enumerate(batches):
                if cancel_event.is_set():
                    update_state(status="CANCELLED")
                    return

                update_state(
                    current_batch=bi + 1,
                    progress=15 + int((bi / total_batches) * 70)
                )

                # 调用底层批次渲染
                res_batch = build_batch_render(
                    batch_segs=b_segs,
                    bi=bi,
                    enc_for_batch=encoder,
                    fps=fps,
                    width=width,
                    height=height,
                    seg_cfg=seg_cfg,
                    out_cfg=out_cfg,
                    audio_cfg=audio_cfg,
                    date=date,
                    cam_index=cam_index,
                    rows=healthy_rows
                )

                if not res_batch:
                    err = f"Batch {bi+1}/{total_batches} render failed"
                    logger.error(err)
                    update_state(status="FAILED", error=err)
                    return

                batch_outputs.append(Path(res_batch))

            if cancel_event.is_set():
                update_state(status="CANCELLED")
                return

            # 7. 合并最终成片
            update_state(status="CONCATING", progress=90)
            if len(batch_outputs) == 1:
                batch_outputs[0].replace(final_output_path)
                ok = True
            else:
                ok = concat_output_files(batch_outputs, final_output_path)
                if ok:
                    for p in batch_outputs:
                        p.unlink(missing_ok=True)

            if not ok or not final_output_path.exists():
                update_state(status="FAILED", error="Final concat failed")
                return

            # 8. 成功终态
            file_size_mb = round(final_output_path.stat().st_size / (1024 * 1024), 2)
            logger.info("Re-render finished: %s (%.1f MB) in %.1fs", final_output_path.name, file_size_mb, time.time() - t0)
            update_state(
                status="COMPLETED",
                progress=100,
                output_file=str(final_output_path),
                file_size_mb=file_size_mb
            )

        except Exception as e:
            logger.exception("Re-render worker encountered unhandled exception: %s", e)
            update_state(status="FAILED", error=str(e))
