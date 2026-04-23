import subprocess
import os
from typing import List, Tuple
from pathlib import Path

class FFmpegToolkit:
    """
    FFmpeg 工具箱
    职责：无损切割、子片段合并。
    """
    @staticmethod
    def cut_segments(input_path: str, segments: List[Tuple[float, float]], temp_dir: str) -> List[str]:
        """
        根据时间轴无损切割出多个子片段。
        """
        output_files = []
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, (start, end) in enumerate(segments):
            duration = end - start
            if duration <= 0:
                continue
                
            out_name = f"part_{i:04d}.mp4"
            out_path = os.path.join(temp_dir, out_name)
            
            # -ss 放在 -i 前面利用关键帧索引加速
            # -c copy 强制全流拷贝
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-ss', str(round(start, 3)),
                '-t', str(round(duration, 3)),
                '-i', input_path,
                '-c', 'copy',
                '-map', '0', # 拷贝所有流（视频、音频、数据）
                out_path
            ]
            
            subprocess.run(cmd, check=True)
            output_files.append(out_path)
            
        return output_files

    @staticmethod
    def merge_videos(clip_list: List[str], final_output: str):
        """
        使用 concat 协议合并所有片段。
        """
        if not clip_list:
            return
            
        # 创建临时 concat 列表文件
        list_file = os.path.join(os.path.dirname(final_output), "concat_list.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for clip in clip_list:
                # 转换路径格式以适应 ffmpeg
                abs_path = os.path.abspath(clip).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            final_output
        ]
        
        try:
            subprocess.run(cmd, check=True)
        finally:
            # 清理 concat 列表
            if os.path.exists(list_file):
                os.remove(list_file)
                
    @staticmethod
    def cleanup(clip_list: List[str]):
        """清理临时生成的子片段"""
        for clip in clip_list:
            if os.path.exists(clip):
                os.remove(clip)
