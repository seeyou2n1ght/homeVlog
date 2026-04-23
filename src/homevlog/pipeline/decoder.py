import subprocess
import threading
import queue
import re
import numpy as np
from typing import Tuple, Optional

class VideoDecoder:
    """
    FFmpeg 解码器
    职责：硬件加速解码，并精准提取每帧的 PTS。
    """
    def __init__(self, hwaccel: str = "qsv", io_timeout: int = 30):
        self.hwaccel = hwaccel
        self.io_timeout = io_timeout
        self.process = None
        self.pts_queue = queue.Queue(maxsize=100)
        self.frame_queue = queue.Queue(maxsize=30) # 缓冲约 1.5 秒的 4K 帧
        self.stop_event = threading.Event()
        self._width = 0
        self._height = 0

    def _parse_stderr(self):
        """线程：从 stderr 解析 showinfo 输出的 PTS"""
        # 匹配示例: n:   0 pts:      0 pts_time:0 ...
        pts_re = re.compile(r"n:\s*\d+\s+pts:\s*\d+\s+pts_time:(\d+\.?\d*)")
        
        while not self.stop_event.is_set():
            line = self.process.stderr.readline().decode('utf-8', errors='ignore')
            if not line:
                break
            
            match = pts_re.search(line)
            if match:
                pts_time = float(match.group(1))
                # 循环等待放入，响应 stop_event 避免死锁
                while not self.stop_event.is_set():
                    try:
                        self.pts_queue.put(pts_time, timeout=1.0)
                        break
                    except queue.Full:
                        pass

    def start(self, file_path: str):
        """启动 FFmpeg 解码进程并动态探测分辨率"""
        # 1. 动态探测分辨率
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', file_path
            ]
            output = subprocess.check_output(probe_cmd, text=True).strip()
            w, h = output.split('x')
            self._width = int(w)
            self._height = int(h)
            print(f"[Decoder] Probed resolution: {self._width}x{self._height}")
        except Exception as e:
            print(f"[Decoder] Warning: ffprobe failed, using 4K fallback: {e}")
            self._width = 3840
            self._height = 2160
        
        # 2. 构建命令
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-y',
            '-hwaccel', self.hwaccel,
            '-i', file_path,
            '-map', '0:v:0',   # 明确仅提取视频流，避免音频干扰 rawvideo 和 showinfo
            '-vf', 'showinfo', # 用于输出 PTS 到 stderr
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1'
        ]
        
        # 如果是本地开发环境且没有 QSV，去掉硬件加速参数
        if self.hwaccel == "none":
            cmd.pop(3); cmd.pop(3)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._width * self._height * 3
        )
        
        # 启动 PTS 解析线程
        self.pts_thread = threading.Thread(target=self._parse_stderr, daemon=True)
        self.pts_thread.start()
        
        # 启动帧读取线程
        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reader_thread.start()

    def _read_frames(self):
        """线程：从 stdout 读取原始像素（防止 Short Reads 和死锁）"""
        frame_size = self._width * self._height * 3
        while not self.stop_event.is_set():
            # 安全读取：拼凑出一个完整的帧，防止读取短缺
            raw_frame = bytearray()
            bytes_needed = frame_size
            while bytes_needed > 0 and not self.stop_event.is_set():
                chunk = self.process.stdout.read(bytes_needed)
                if not chunk:
                    break
                raw_frame.extend(chunk)
                bytes_needed -= len(chunk)
            
            if bytes_needed > 0:
                break # EOF 或者被终止
            
            # 将 bytes 转换为 numpy 数组
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self._height, self._width, 3))
            
            # 获取对应的 PTS 并放入下游队列，支持中断
            pts = None
            while not self.stop_event.is_set():
                try:
                    pts = self.pts_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    # 如果进程意外死了，提早退出
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
        """外部接口：获取一帧数据 (PTS, Frame)"""
        while not self.stop_event.is_set():
            try:
                return self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                if self.stop_event.is_set():
                    return None

        # 将队列排空收尾
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        """释放资源"""
        self.stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        print("[Decoder] Process closed.")
