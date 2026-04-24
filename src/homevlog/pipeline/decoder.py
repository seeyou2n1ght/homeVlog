"""
FFmpeg 解码器 (V2.4 高性能版)
改进点：
  - 移除 showinfo 滤镜，消除 stderr 管道争用
  - PTS 通过帧序号 / fps 直接计算（fps 滤镜输出为 CFR，数学精确）
  - 支持 QSV / CUDA / 软解三级 fallback
"""
import subprocess
import threading
import queue
import numpy as np
import bisect
from typing import Tuple, Optional, List


class VideoDecoder:

    def __init__(self, hwaccel: str = "qsv", io_timeout: int = 30):
        self.hwaccel = hwaccel
        self.io_timeout = io_timeout
        self.process: Optional[subprocess.Popen] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=64)
        self.stop_event = threading.Event()
        self._width = 640
        self._height = 640
        self._fps = 5
        self.pts_list: List[float] = []

    @property
    def method(self) -> str:
        return self.hwaccel

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> int:
        return self._fps

    def _scan_pts(self, file_path: str) -> List[float]:
        """[Phase 2] 预扫描原视频的所有 PTS，解决 VFR 时间轴问题"""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0", file_path
        ]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            # 解析为 float 列表并排序
            pts_list = []
            for line in output.splitlines():
                line = line.strip()
                if line:
                    try:
                        pts_list.append(float(line))
                    except ValueError:
                        pass
            return sorted(pts_list)
        except subprocess.CalledProcessError as e:
            print(f"[Decoder] ffprobe error: {e.output}")
            return []
        self._fps = 5
        self._frame_interval: float = 0.2  # 1/fps

    def _build_cmd(self, file_path: str) -> List[str]:
        """[Phase 1] 纯硬件解码与硬件缩放滤镜"""
        vf = ""
        base_cmd = ["ffmpeg", "-hide_banner", "-y"]
        
        if self.hwaccel == "qsv":
            base_cmd.extend(["-hwaccel", "qsv", "-hwaccel_output_format", "qsv", "-i", file_path])
            vf = f"vpp_qsv=w={self._width}:h={self._height},hwdownload,format=nv12,fps={self._fps}"
        elif self.hwaccel == "cuda":
            base_cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", file_path])
            vf = f"scale_cuda={self._width}:{self._height},hwdownload,format=nv12,fps={self._fps}"
        else:
            base_cmd.extend(["-i", file_path])
            vf = f"scale={self._width}:{self._height},fps={self._fps}"
            
        output_args = [
            "-map", "0:v:0",
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1"
        ]
        
        return base_cmd + output_args

    def _drain_stderr(self) -> None:
        """守护线程：持续消费 stderr 防止管道缓冲区死锁（不解析内容）"""
        while not self.stop_event.is_set():
            line = self.process.stderr.readline()
            if not line:
                break

    def _read_frames(self) -> None:
        """守护线程：从 stdout 读取定长像素帧，PTS 通过帧序号计算"""
        frame_size = self._width * self._height * 3
        frame_idx = 0

        while not self.stop_event.is_set():
            raw = bytearray()
            remaining = frame_size
            while remaining > 0 and not self.stop_event.is_set():
                chunk = self.process.stdout.read(remaining)
                if not chunk:
                    self.stop_event.set()
                    return
                raw.extend(chunk)
                remaining -= len(chunk)

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self._height, self._width, 3)
            )
            # 获取目标 PTS，并在原 PTS 列表中寻找最接近的真实 PTS
            target_pts = frame_idx * self._frame_interval
            pts = target_pts
            if self.pts_list:
                idx = bisect.bisect_left(self.pts_list, target_pts)
                if idx == 0:
                    pts = self.pts_list[0]
                elif idx == len(self.pts_list):
                    pts = self.pts_list[-1]
                else:
                    before = self.pts_list[idx - 1]
                    after = self.pts_list[idx]
                    pts = after if (after - target_pts) < (target_pts - before) else before

            while not self.stop_event.is_set():
                try:
                    self.frame_queue.put((pts, frame), timeout=1.0)
                    break
                except queue.Full:
                    pass

            frame_idx += 1

        self.stop_event.set()

    def start(self, file_path: str, pts_list: List[float] = None, fps: int = 5, infer_resolution: int = 640) -> None:
        """启动解码进程"""
        self._width = infer_resolution
        self._height = infer_resolution
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self.stop_event.clear()
        
        # 优先使用传入的 PTS 列表，否则自行扫描
        self.pts_list = pts_list if pts_list is not None else self._scan_pts(file_path)

        cmd = self._build_cmd(file_path)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._width * self._height * 3 * 16,
        )

        # 两个守护线程：读帧 + 排空 stderr
        threading.Thread(target=self._read_frames, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def get_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        """获取下一帧 (pts, bgr_frame)，返回 None 表示流结束"""
        while not self.stop_event.is_set():
            try:
                return self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        """关闭解码器并强制回收进程"""
        self.stop_event.set()
        if self.process:
            try:
                # 尝试优雅关闭
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, Exception):
                # 强制杀掉
                if self.process and self.process.poll() is None:
                    try:
                        self.process.kill()
                    except:
                        pass
            finally:
                self.process = None
        # 清空队列防止引用计数无法归零
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break

    def __del__(self):
        """析构保护：确保对象销毁时进程也退出"""
        try:
            self.close()
        except:
            pass
