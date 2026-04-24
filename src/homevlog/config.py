import yaml
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

class PathConfig(BaseModel):
    nas_input: str
    local_output: str
    sqlite_db: str

class ScannerConfig(BaseModel):
    write_detect_interval_sec: int = 5
    write_detect_stable_count: int = 2
    file_pattern: str = "*.mp4"

class DetectionConfig(BaseModel):
    model_path: str
    classes: List[str]
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.8
    debounce_frames: int = 5
    static_interval_ms: int = 60000
    fps: int = 5
    infer_resolution: int = 640
    enable_pixel_motion: bool = False
    pixel_motion_grid: int = 16
    pixel_motion_history_size: int = 20
    pixel_motion_threshold_factor: float = 2.5

class CuttingConfig(BaseModel):
    keyframe_padding_ms: int = 2500
    io_timeout_sec: int = 30
    parallel_jobs: int = 4

class HardwareConfig(BaseModel):
    use_mock: bool = False
    ffmpeg_hwaccel: str = "qsv"
    batch_size: int = 1
    gpu_id: int = 0

class AppConfig(BaseModel):
    paths: PathConfig
    scanner: ScannerConfig
    detection: DetectionConfig
    cutting: CuttingConfig
    hardware: HardwareConfig

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """加载并解析 YAML 配置文件"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return AppConfig.model_validate(data)

# 全局配置单例（按需在 main 中加载）
# config = load_config()
