"""
时间工具模块
职责：从监控文件名中解析出精确的录制时间范围，支持跨天检测与日期归属。

文件名规范：{cam_id}_{YYYYMMDDHHMMSS}_{YYYYMMDDHHMMSS}.mp4
示例：00_20260423203412_20260423203931.mp4
"""
import re
from datetime import datetime, date
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 匹配格式：任意前缀_开始时间14位_结束时间14位.mp4
_FILENAME_PATTERN = re.compile(
    r".*?_(\d{14})_(\d{14})\.mp4$", re.IGNORECASE
)
_DT_FMT = "%Y%m%d%H%M%S"


@dataclass
class VideoFileInfo:
    """解析后的视频文件元数据"""
    path: str
    start_dt: datetime
    end_dt: datetime

    @property
    def date_str(self) -> str:
        """主日期（以开始时间为准），格式 YYYYMMDD"""
        return self.start_dt.strftime("%Y%m%d")

    @property
    def spans_midnight(self) -> bool:
        """是否跨越了午夜 00:00（即跨天素材）"""
        return self.start_dt.date() != self.end_dt.date()

    @property
    def start_date(self) -> date:
        return self.start_dt.date()

    @property
    def end_date(self) -> date:
        return self.end_dt.date()


def parse_video_filename(path: str) -> Optional[VideoFileInfo]:
    """
    从文件名解析时间信息，失败返回 None（不抛异常，由调用方降级处理）。
    """
    name = Path(path).name
    m = _FILENAME_PATTERN.match(name)
    if not m:
        return None
    try:
        start_dt = datetime.strptime(m.group(1), _DT_FMT)
        end_dt   = datetime.strptime(m.group(2), _DT_FMT)
        return VideoFileInfo(path=path, start_dt=start_dt, end_dt=end_dt)
    except ValueError:
        return None
