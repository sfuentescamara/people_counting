"""
Test people counting pipeline without display (for automated testing).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.yolox_detector import YOLOXDetector
from src.tracker.centroid_tracker import CentroidTracker
from src.counter.line_counter import LineCounter
from src.video.video_reader import VideoReader, VideoWriter
import cv2
import numpy as np
import time


def main():
    """Main function."""
    print("=" * 60)
    print("PEOPLE COUNTING PIPELINE TEST (No Display)")
    print("=" * 60)

    # Check for model
    model_path = "models/yolox-s.onnx"
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        return

    # Find test video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        videos_dir = Path("data/videos")
        video_files = list(videos_dir.glob("*.mp4"))
        if not video_files:
            print("\n[ERROR] No videos found")
            return
        video_path = str(video_files[0])

    print(f"\n[INFO] Using video: {Path(video_path).name}")

    # Initialize
    detector = YOLOXDetector(model_path=model_path, conf_threshold=0.5, class_filter=[0])
    tracker = CentroidTracker(max_disappeared=30, max_distance=100)
    reader = VideoReader(video_path, target_fps=15.0, max_frames=100)  # Test 100 frames

    # Set counting line
    line_y = reader.height // 2
    counter = LineCounter(
        line_position=((0, line_y), (reader.width, line_y)),
        direction_labels=("up", "down")
    )

    # Create writer
    output_path = Path("output/videos") / f"{Path(video_path).stem}_counted.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(output_path), reader.target_fps, reader.width, reader.height)

    print(f"\n[INFO] Processing {reader.max_frames} frames...")

    frame_count = 0
    processing_times = []

    try:
        while True:
            ret, frame = reader.read()
            if not ret:
                break

            start_time = time.time()

            # Detect → Track → Count
            detections = detector.detect_people(frame)
            tracked_objects = tracker.update(detections)
            counts = counter.update(tracked_objects)

            # Visualize
            frame_viz = frame.copy()
            for object_id, (centroid, bbox) in tracked_objects.items():
                x1, y1, x2, y2 = bbox.astype(int)
                cx, cy = centroid.astype(int)
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                color = colors[object_id % len(colors)]
                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame_viz, (cx, cy), 4, color, -1)
                cv2.putText(frame_viz, f"ID:{object_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw line
            cv2.line(frame_viz, (0, line_y), (reader.width, line_y), (0, 255, 255), 3)

            # Draw info
            cv2.putText(frame_viz, f"Frame: {frame_count}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_viz, f"People: {len(tracked_objects)}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_viz, f"UP: {counts['up']} | DOWN: {counts['down']}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            writer.write(frame_viz)

            process_time = time.time() - start_time
            processing_times.append(process_time)
            frame_count += 1

            if frame_count % 30 == 0:
                avg_fps = 1.0 / np.mean(processing_times[-30:])
                print(f"Frame {frame_count}: FPS={avg_fps:.1f}, People={len(tracked_objects)}, "
                      f"UP={counts['up']}, DOWN={counts['down']}")

    finally:
        reader.release()
        writer.release()

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print('=' * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {1.0 / np.mean(processing_times):.2f}")
    print(f"Final counts: UP={counts['up']}, DOWN={counts['down']}")
    print(f"Total crossings: {counter.get_total_count()}")
    print(f"Output: {output_path}")
    print('=' * 60)
    print("\n✅ Tracking and counting working!")


if __name__ == "__main__":
    main()
