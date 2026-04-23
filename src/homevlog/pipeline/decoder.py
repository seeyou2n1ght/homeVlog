import subprocess
import threading
import queue
import re
import numpy as np
from typing import Tuple, Optional

class VideoDecoder:
    """
    FFmpeg 解码器 (V2.2 降维提速版)
    职责：底层硬件解码缩放、固定帧率抽取、安全像素流解析。
    """
    def __init__(self, hwaccel: str = "qsv", io_timeout: int = 30):
        self.hwaccel = hwaccel
        self.io_timeout = io_timeout
        self.process = None
        self.pts_queue = queue.Queue(maxsize=100)
        self.frame_queue = queue.Queue(maxsize=64) # 缓冲增大以应对 Batch 索取
        self.stop_event = threading.Event()
        self._width = 640
        self._height = 640

    def _parse_stderr(self):
        """线程：从 stderr 解析 showinfo 输出的 PTS"""
        pts_re = re.compile(r"n:\s*\d+\s+pts:\s*\d+\s+pts_time:(\d+\.?\d*)")
        while not self.stop_event.is_set():
            line = self.process.stderr.readline().decode('utf-8', errors='ignore')
            if not line:
                break
            match = pts_re.search(line)
            if match:
                pts_time = float(match.group(1))
                while not self.stop_event.is_set():
                    try:
                        self.pts_queue.put(pts_time, timeout=1.0)
                        break
                    except queue.Full:
                        pass

    def start(self, file_path: str, fps: int = 5, infer_resolution: int = 640):
        """启动 FFmpeg 降维解码进程"""
        self._width = infer_resolution
        self._height = infer_resolution
        
        print(f"[Decoder] HW Pipeline: {self.hwaccel} -> {self._width}x{self._height} @ {fps}fps")
        
        # 构建命令：核心在于 scale+fps 极速降维，彻底摒弃 4K 数据通过管道
        filter_str = f"scale={self._width}:{self._height},fps={fps},showinfo"
        
        cmd = [
            'ffmpeg', '-hide_banner', '-y',
            '-hwaccel', self.hwaccel,
            '-i', file_path,
            '-map', '0:v:0',
            '-vf', filter_str,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1'
        ]
        
        if self.hwaccel == "none":
            cmd.pop(3); cmd.pop(3)

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self._width * self._height * 3 * 16 # 适度增加系统缓冲
        )
        
        self.pts_thread = threading.Thread(target=self._parse_stderr, daemon=True)
        self.pts_thread.start()
        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reader_thread.start()

    def _read_frames(self):
        """线程：从 stdout 安全读取定长降维像素"""
        frame_size = self._width * self._height * 3
        while not self.stop_event.is_set():
            raw_frame = bytearray()
            bytes_needed = frame_size
            while bytes_needed > 0 and not self.stop_event.is_set():
                chunk = self.process.stdout.read(bytes_needed)
                if not chunk:
                    break
                raw_frame.extend(chunk)
                bytes_needed -= len(chunk)
            
            if bytes_needed > 0:
                break 
            
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self._height, self._width, 3))
            
            pts = None
            while not self.stop_event.is_set():
                try:
                    pts = self.pts_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    if self.process.poll() is not None:
                        self.stop_event.set()
                    continue
                    
            if pts is None:
                break

            while not self.stop_event.is_set():
                try:
                    self.frame_queue.put((pts, frame), timeout=1.0)
                    break
                except queue.Full:
                    pass
        
        self.stop_event.set()

    def get_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        while not self.stop_event.is_set():
            try:
                return self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                if self.stop_event.is_set():
                    return None
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        print("[Decoder] Process closed.")
