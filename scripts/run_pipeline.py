"""
Test end-to-end video processing pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.yolox_detector import YOLOXDetector
from src.video.pipeline import VideoProcessingPipeline


def main():
    """Main test function."""
    print("=" * 60)
    print("VIDEO PROCESSING PIPELINE TEST")
    print("=" * 60)

    # Check for model
    model_path = "models/yolox-s.onnx"
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please run: python scripts/download_yolox_model.py")
        return

    # Initialize detector
    print("\n[INFO] Initializing detector...")
    detector = YOLOXDetector(
        model_path=model_path,
        conf_threshold=0.5,
        nms_threshold=0.45,
        class_filter=[0]  # Person only
    )

    # Initialize pipeline
    print("\n[INFO] Initializing pipeline...")
    pipeline = VideoProcessingPipeline(
        detector=detector,
        show_display=True,
        save_output=True,
        output_dir="output/videos"
    )

    # Find test video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Look for videos in data/videos
        videos_dir = Path("data/videos")
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

        if not video_files:
            print("\n[ERROR] No videos found in data/videos/")
            print("\nUsage: python scripts/test_pipeline.py <video_path>")
            print("Or add videos to data/videos/")
            return

        video_path = str(video_files[0])
        print(f"\n[INFO] Using video: {video_path}")

    # Process video
    print("\n[INFO] Starting video processing...")
    print("[INFO] Press 'q' to quit, SPACE to pause/resume")

    stats = pipeline.process_video(
        video_path=video_path,
        target_fps=15.0,  # Process at 15 FPS for demo
        max_frames=None   # Process all frames (change to e.g., 100 for quick test)
    )

    # Done
    print("\n[OK] Pipeline test completed!")


if __name__ == "__main__":
    main()
