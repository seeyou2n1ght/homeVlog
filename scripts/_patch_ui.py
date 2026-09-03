"""One-shot patch: replace tqdm progress bars with Rich PipelineDashboard."""

PIPELINE_PATH = "src/pipeline.py"

with open(PIPELINE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Replace tqdm import with PipelineDashboard import
content = content.replace(
    "from tqdm import tqdm",
    "from src.ui import PipelineDashboard"
)

# 2. In __init__: replace self.pbars = {} with dashboard setup
content = content.replace(
    "        # 进度条\n        self.pbars = {}",
    "        # Rich 仪表盘\n        self.dashboard_enabled = dashboard_enabled\n        self.dashboard: PipelineDashboard | None = None\n        self._prescreen_t0 = 0.0\n        self._analysis_t0 = 0.0\n        self._render_t0 = 0.0"
)

# 3. In run(): replace tqdm bar creation with PipelineDashboard
old_run_bars = '''        self.pbars["prescreen"] = tqdm(total=len(pending_prescreen), desc=f" {self.date} Prescreen", unit="file", position=0)
        self.pbars["analysis"] = tqdm(total=len(pending_analysis), desc=f" {self.date} Analysis ", unit="file", position=1)
        if self.render_enabled:
            self.pbars["render"] = tqdm(total=0, desc=f" {self.date} Rendering", unit="batch", position=2)'''

new_run_bars = '''        self.dashboard = PipelineDashboard(
            date=self.date, cam_index=self.cam_index,
            total_prescreen=len(pending_prescreen),
            render_enabled=self.render_enabled,
            enabled=self.dashboard_enabled,
        )
        self.dashboard.start()
        self._prescreen_t0 = time.monotonic()
        self._analysis_t0 = time.monotonic()'''

content = content.replace(old_run_bars, new_run_bars)

# 4. Replace pbar.close() with dashboard.stop()
content = content.replace(
    "        for p in self.pbars.values():\n            p.close()",
    "        if self.dashboard:\n            self.dashboard.stop()"
)

# 5. In _prescreen_worker: replace tqdm update with dashboard.update_prescreen()
old_prescreen_update = '''                if "prescreen" in self.pbars:
                    self.pbars["prescreen"].update(1)
                    el_s = max(0.1, time.monotonic() - getattr(self, "_prescreen_t0", t0))
                    n_done = self.pbars["prescreen"].n
                    fps_val = n_done / el_s
                    el_m, el_sec = divmod(int(el_s), 60)
                    self.pbars["prescreen"].set_postfix_str(
                        f"实耗 {el_m:02d}:{el_sec:02d} | 速率 {fps_val:.1f}文件/s (Latest: {Path(filepath).name[:22]}..)"
                    )'''

new_prescreen_update = '''                if self.dashboard:
                    el_s = max(0.1, time.monotonic() - self._prescreen_t0)
                    n_done = self.dashboard.prescreen_done + 1
                    fps_val = n_done / el_s
                    self.dashboard.update_prescreen(
                        completed=n_done,
                        latest_file=Path(filepath).name,
                        speed_str=f"{fps_val:.1f}文件/s",
                    )
                    self.dashboard.set_queue_status(
                        prescreen_q=self.prescreen_queue.qsize(),
                        analysis_q=self.analysis_queue.qsize(),
                        render_q=self.render_batch_queue.qsize(),
                    )'''

content = content.replace(old_prescreen_update, new_prescreen_update)

# 6. In _prescreen_worker: replace analysis total update
old_analysis_total = '''                if res["status"] == "SUSPICIOUS":
                    self.analysis_queue.put(task)
                    if "analysis" in self.pbars:
                        self.pbars["analysis"].total += 1
                        self.pbars["analysis"].refresh()'''

new_analysis_total = '''                if res["status"] == "SUSPICIOUS":
                    self.analysis_queue.put(task)
                    if self.dashboard:
                        self.dashboard.update_analysis(
                            completed=self.dashboard.analysis_done,
                            total=self.dashboard.analysis_total + 1,
                        )'''

content = content.replace(old_analysis_total, new_analysis_total)

# 7. In _analysis_worker: replace tqdm update with dashboard.update_analysis()
old_analysis_update = '''        if "analysis" in self.pbars:
            self.pbars["analysis"].update(1)
            el_s = max(0.1, time.monotonic() - getattr(self, "_analysis_t0", t0))
            n_done = self.pbars["analysis"].n
            n_tot = max(n_done, self.pbars["analysis"].total)
            avg_s = el_s / max(1, n_done)
            rem_s = max(0, (n_tot - n_done) * avg_s)
            el_m, el_sec = divmod(int(el_s), 60)
            rm_m, rm_sec = divmod(int(rem_s), 60)
            self.pbars["analysis"].set_postfix_str(
                f"实耗 {el_m:02d}:{el_sec:02d} | 预估余 {rm_m:02d}:{rm_sec:02d} | 均速 {avg_s:.1f}s/文件 ({gpu.upper()})"
            )'''

new_analysis_update = '''        if self.dashboard:
            el_s = max(0.1, time.monotonic() - self._analysis_t0)
            n_done = self.dashboard.analysis_done + 1
            n_tot = max(n_done, self.dashboard.analysis_total)
            avg_s = el_s / max(1, n_done)
            rem_s = max(0, (n_tot - n_done) * avg_s)
            el_m, el_sec = divmod(int(el_s), 60)
            rm_m, rm_sec = divmod(int(rem_s), 60)
            self.dashboard.update_analysis(
                completed=n_done,
                latest_file=Path(filepath).name,
                speed_str=f"实耗 {el_m:02d}:{el_sec:02d} | 余 {rm_m:02d}:{rm_sec:02d} | {avg_s:.1f}s/文件 ({gpu.upper()})",
            )
            self.dashboard.set_queue_status(
                prescreen_q=self.prescreen_queue.qsize(),
                analysis_q=self.analysis_queue.qsize(),
                render_q=self.render_batch_queue.qsize(),
            )'''

content = content.replace(old_analysis_update, new_analysis_update)

# 8. In _render_manager: replace tqdm update for render
old_render_total = '''            if "render" in self.pbars:
                self.pbars["render"].total += 1
                self.pbars["render"].refresh()'''

new_render_total = '''            if self.dashboard:
                self.dashboard.update_render(
                    completed=self.dashboard.render_done,
                    total=self.dashboard.render_total + 1,
                )'''

# This appears in _enqueue_batch - need to handle carefully
content = content.replace(old_render_total, new_render_total)

old_render_update = '''                        if "render" in self.pbars:
                            self.pbars["render"].update(1)
                            n_done = self.pbars["render"].n
                            n_total = max(n_done, self.pbars["render"].total)
                            avg_s = sum(render_start_t) / max(1, len(render_start_t))
                            rem_s = max(0, (n_total - n_done) * avg_s)
                            el_m, el_sec = divmod(int(sum(render_start_t)), 60)
                            rm_m, rm_sec = divmod(int(rem_s), 60)
                            self.pbars["render"].set_postfix_str(
                                f"渲染实耗: {el_m:02d}:{el_sec:02d} | 预估余时: {rm_m:02d}:{rm_sec:02d} | 均速 {avg_s:.1f}s/批 (Batch {b_idx} on {gpu.upper()})"
                            )'''

new_render_update = '''                        if self.dashboard:
                            n_done = self.dashboard.render_done + 1
                            n_total = max(n_done, self.dashboard.render_total)
                            avg_s = sum(render_start_t) / max(1, len(render_start_t))
                            rem_s = max(0, (n_total - n_done) * avg_s)
                            el_m, el_sec = divmod(int(sum(render_start_t)), 60)
                            rm_m, rm_sec = divmod(int(rem_s), 60)
                            self.dashboard.update_render(
                                completed=n_done,
                                latest_batch=f"Batch {b_idx} ({gpu.upper()})",
                                speed_str=f"渲染 {el_m:02d}:{el_sec:02d} | 余 {rm_m:02d}:{rm_sec:02d} | {avg_s:.1f}s/批",
                            )'''

content = content.replace(old_render_update, new_render_update)

# 9. In _render_manager: update render batch queue size after enqueue
# Add dashboard queue status update after _enqueue_batch calls in render_manager

# Write the result
with open(PIPELINE_PATH, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(content)

print("Pipeline patched: tqdm -> Rich PipelineDashboard")

# Verify
remaining_tqdm = content.count("tqdm")
remaining_pbars = content.count("self.pbars")
dashboard_refs = content.count("self.dashboard")
print(f"  Remaining tqdm refs: {remaining_tqdm}")
print(f"  Remaining self.pbars refs: {remaining_pbars}")
print(f"  Dashboard refs: {dashboard_refs}")
