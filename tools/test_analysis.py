import argparse
import sys
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import load_config, setup_logging
from src.detector import MotionDetector

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Test Analysis (detector) logic on a single file")
    parser.add_argument("filepath", type=str, help="Path to video file")
    parser.add_argument("--sensitivity", type=float, help="Override sensitivity")
    parser.add_argument("--min-motion", type=float, help="Override min_motion_duration")
    parser.add_argument("--gpu", type=str, default="cuda", choices=["qsv", "cuda", "cpu"], help="Hardware decode acceleration")
    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return

    config = load_config()
    
    if args.sensitivity is not None:
        if "detection" not in config: config["detection"] = {}
        config["detection"]["sensitivity"] = args.sensitivity
    if args.min_motion is not None:
        if "detection" not in config: config["detection"] = {}
        config["detection"]["min_motion_duration"] = args.min_motion

    print(f"Testing Analysis on: {filepath}")
    print(f"Using GPU: {args.gpu}")
    print(f"Config: {config.get('detection', {})}")

    detector = MotionDetector(config, decode_gpu=args.gpu)
    try:
        results = detector.detect_motion(str(filepath), start_offset=0.0)
        
        motion_count = sum(1 for r in results if r["is_motion"])
        total = len(results)
        print(f"\nTotal frames analyzed: {total}")
        print(f"Motion frames detected: {motion_count} ({motion_count/total*100:.1f}%)")
        print(f"Performance: {detector.last_perf}")
        
        # Build segments to see the actual cuts
        from src.segment import build_segments
        segments = build_segments(str(filepath), results, config)
        print(f"\nResulting Segments ({len(segments)}):")
        for i, seg in enumerate(segments):
            print(f"  {i+1}: start={seg.start:.2f} end={seg.end:.2f} (dur={seg.duration:.2f}, static={seg.is_static})")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
