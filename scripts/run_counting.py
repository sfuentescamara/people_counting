"""
Test complete people counting pipeline with tracking and counting.
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
    """Main function for complete pipeline."""
    print("=" * 60)
    print("PEOPLE COUNTING PIPELINE - Full Demo")
    print("=" * 60)

    # Check for model
    model_path = "models/yolox-s.onnx"
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please run: python scripts/download_yolox_model.py")
        return

    # Find test video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        videos_dir = Path("data/videos")
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

        if not video_files:
            print("\n[ERROR] No videos found in data/videos/")
            print("\nUsage: python scripts/test_counting.py <video_path>")
            return

        video_path = str(video_files[0])
        print(f"\n[INFO] Using video: {video_path}")

    # Initialize components
    print("\n[INFO] Initializing detector...")
    detector = YOLOXDetector(
        model_path=model_path,
        conf_threshold=0.5,
        nms_threshold=0.45,
        class_filter=[0]  # Person only
    )

    print("[INFO] Initializing tracker...")
    tracker = CentroidTracker(max_disappeared=30, max_distance=100)

    # Create video reader
    print("\n[INFO] Loading video...")
    reader = VideoReader(video_path, target_fps=15.0)

    # Set counting line (horizontal line at middle of frame)
    line_y = reader.height // 2
    line_start = (0, line_y)
    line_end = (reader.width, line_y)

    print(f"[INFO] Setting counting line at y={line_y}")
    counter = LineCounter(
        line_position=(line_start, line_end),
        direction_labels=("up", "down")
    )

    # Create video writer
    output_path = Path("output/videos") / f"{Path(video_path).stem}_counted.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(str(output_path), reader.target_fps, reader.width, reader.height)

    print("\n[INFO] Processing video...")
    print("[INFO] Press 'q' to quit, SPACE to pause/resume\n")

    frame_count = 0
    processing_times = []

    try:
        while True:
            # Read frame
            ret, frame = reader.read()
            if not ret:
                break

            start_time = time.time()

            # 1. Detect
            detections = detector.detect_people(frame)

            # 2. Track
            tracked_objects = tracker.update(detections)

            # 3. Count
            counts = counter.update(tracked_objects)

            # 4. Visualize
            # Draw tracked objects
            frame_viz = frame.copy()
            for object_id, (centroid, bbox) in tracked_objects.items():
                x1, y1, x2, y2 = bbox.astype(int)
                cx, cy = centroid.astype(int)

                # Color based on ID
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color = colors[object_id % len(colors)]

                # Draw bbox and ID
                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame_viz, (cx, cy), 4, color, -1)
                cv2.putText(frame_viz, f"ID:{object_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw counting line
            cv2.line(frame_viz, line_start, line_end, (0, 255, 255), 3)

            # Draw info
            process_time = time.time() - start_time
            processing_times.append(process_time)
            fps = 1.0 / process_time if process_time > 0 else 0

            # Info overlay
            info_lines = [
                f"Frame: {frame_count}",
                f"People: {len(tracked_objects)}",
                f"FPS: {fps:.1f}",
                f"UP: {counts['up']} | DOWN: {counts['down']}"
            ]

            y_offset = 40
            for line in info_lines:
                cv2.putText(frame_viz, line, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 35

            # Display
            cv2.imshow("People Counting Demo", frame_viz)

            # Write
            writer.write(frame_viz)

            frame_count += 1

            # Progress
            if frame_count % 30 == 0:
                avg_fps = 1.0 / np.mean(processing_times[-30:])
                print(f"Processed {frame_count} frames | Avg FPS: {avg_fps:.1f} | "
                      f"People: {len(tracked_objects)} | Counts: UP={counts['up']} DOWN={counts['down']}")

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] User requested quit")
                break
            elif key == ord(' '):
                print("[INFO] Paused - press any key to continue")
                cv2.waitKey(0)

    finally:
        reader.release()
        writer.release()
        cv2.destroyAllWindows()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {1.0 / np.mean(processing_times):.2f}")
    print(f"Final counts: UP={counts['up']}, DOWN={counts['down']}")
    print(f"Total crossings: {counter.get_total_count()}")
    print(f"Output saved: {output_path}")
    print('=' * 60)


if __name__ == "__main__":
    main()
