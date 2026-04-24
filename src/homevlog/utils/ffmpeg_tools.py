import subprocess
import os
from typing import List, Tuple
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class FFmpegToolkit:
    """
    FFmpeg 工具箱 (V2.2 并发提速版)
    职责：并发无损切割、子片段合并。
    """
    @staticmethod
    def cut_segments(
        input_path: str, 
        segments: List[Tuple[float, float]], 
        temp_dir: str, 
        parallel_jobs: int = 4,
        stop_event: threading.Event = None
    ) -> List[str]:
        """
        利用线程池并发进行无损片段切割。
        """
        output_files = []
        os.makedirs(temp_dir, exist_ok=True)
        
        def _cut_single(i, start, end):
            if stop_event and stop_event.is_set():
                return None
                
            duration = end - start
            if duration <= 0:
                return None
            
            out_name = f"{Path(input_path).stem}_part_{i:04d}.mp4"
            out_path = os.path.join(temp_dir, out_name)
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-ss', str(round(start, 3)),
                '-t', str(round(duration, 3)),
                '-i', input_path,
                '-c', 'copy',
                '-map', '0',
                out_path
            ]
            subprocess.run(cmd, check=True)
            return out_path

        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {executor.submit(_cut_single, i, s, e): i for i, (s, e) in enumerate(segments)}
            
            # 保证原本的时间顺序不变
            results = [None] * len(segments)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res_path = future.result()
                    results[idx] = res_path
                except Exception as exc:
                    print(f"[Cutter] Segment {idx} failed: {exc}")
                    
        # 剔除失败或为 None 的结果
        output_files = [p for p in results if p is not None]
        return output_files

    @staticmethod
    def merge_videos(clip_list: List[str], final_output: str):
        if not clip_list:
            return
            
        list_file = os.path.join(os.path.dirname(final_output), "concat_list.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for clip in clip_list:
                abs_path = os.path.abspath(clip).replace('\\', '/')
                # 转义单引号，防止 ffmpeg concat 文件解析错误或注入
                safe_path = abs_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', list_file, '-c', 'copy',
            final_output
        ]
        
        try:
            subprocess.run(cmd, check=True)
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)
                
    @staticmethod
    def cleanup(clip_list: List[str]):
        for clip in clip_list:
            if os.path.exists(clip):
                os.remove(clip)
