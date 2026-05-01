import argparse
import sys
import os
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import load_config, setup_logging
from src.prescreen import prescreen_file

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Test Prescreen logic on a single file")
    parser.add_argument("filepath", type=str, help="Path to video file")
    parser.add_argument("--diff-threshold", type=float, help="Override diff_threshold")
    parser.add_argument("--samples", type=int, help="Override sample points count")
    parser.add_argument("--gpu", type=str, default="qsv", choices=["qsv", "cuda", "cpu"], help="Hardware decode acceleration")
    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return

    config = load_config()
    
    if args.diff_threshold is not None:
        if "detection" not in config: config["detection"] = {}
        config["detection"]["diff_threshold"] = args.diff_threshold
    if args.samples is not None:
        if "detection" not in config: config["detection"] = {}
        config["detection"]["prescreen_samples"] = args.samples

    print(f"Testing Prescreen on: {filepath}")
    print(f"Using GPU: {args.gpu}")
    print(f"Config: {config.get('detection', {})}")

    try:
        from src.ffmpeg import get_duration
        dur = get_duration(str(filepath))
        if dur == 0:
            dur = 300.0
            print("Warning: could not get duration, defaulting to 300s")
        
        status = prescreen_file(str(filepath), dur, config, args.gpu)
        print(f"\nFinal Result status: {status}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
