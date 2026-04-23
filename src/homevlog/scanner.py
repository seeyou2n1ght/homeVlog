import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
from .config import ScannerConfig

class FileScanner:
    """
    NAS 文件扫描器
    职责：发现新素材，并通过多重采样确保文件已写完。
    """
    def __init__(self, config: ScannerConfig, nas_path: str):
        self.config = config
        self.nas_path = Path(nas_path)
        # 记录采样状态: {path: (last_size, last_mtime, stable_count)}
        self.samples: Dict[str, Tuple[int, float, int]] = {}

    def scan_once(self, exclude_paths: List[str] = None) -> List[str]:
        """
        执行一次扫描，返回“已锁定并可处理”的文件列表。
        """
        exclude_paths = exclude_paths or []
        ready_files = []
        
        # 1. 获取当前目录下符合条件的所有文件
        current_files = list(self.nas_path.rglob(self.config.file_pattern))
        
        for file_path in current_files:
            p_str = str(file_path)
            if p_str in exclude_paths:
                continue
                
            try:
                stat = file_path.stat()
                curr_size = stat.st_size
                curr_mtime = stat.st_mtime
                
                # 2. 如果是第一次发现该文件，记录初始状态
                if p_str not in self.samples:
                    self.samples[p_str] = (curr_size, curr_mtime, 1)
                    continue
                    
                last_size, last_mtime, count = self.samples[p_str]
                
                # 3. 校验状态是否发生变化
                if curr_size == last_size and curr_mtime == last_mtime:
                    new_count = count + 1
                    if new_count >= self.config.write_detect_stable_count:
                        # 达到稳定采样次数，判定为可处理
                        ready_files.append(p_str)
                        # 从采样中移除，防止重复产出（外部会记录到 DB）
                        del self.samples[p_str]
                    else:
                        self.samples[p_str] = (curr_size, curr_mtime, new_count)
                else:
                    # 文件还在变，重置稳定计数器
                    self.samples[p_str] = (curr_size, curr_mtime, 1)
                    
            except Exception as e:
                print(f"[Scanner] Warning: Error checking file {p_str}: {e}")
                
        return ready_files

    def wait_for_files(self, db_manager):
        """
        阻塞式生成器：持续扫描直到发现可用的新文件。
        """
        print(f"[Scanner] Monitoring NAS path: {self.nas_path}")
        while True:
            # 扫描时过滤掉数据库中已处理的文件
            # (这里是一个简化的逻辑示例，实际应由调度器组合)
            files = self.scan_once()
            if files:
                for f in files:
                    if not db_manager.is_file_processed(f):
                        yield f
            
            time.sleep(self.config.write_detect_interval_sec)
