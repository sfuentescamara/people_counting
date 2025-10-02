"""
Test detector on a single frame and save the result.
"""

import sys
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_detector import draw_detections
from src.detector.yolox_detector import YOLOXDetector


def main():
    """Test on single frame."""
    print("Testing detector on single frame...")

    # Initialize detector
    model_path = "models/yolox-s.onnx"
    detector = YOLOXDetector(
        model_path=model_path,
        conf_threshold=0.5,
        nms_threshold=0.45,
        class_filter=[0]
    )

    # Load first frame from video
    video_path = "data/videos/mall_stairs.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read frame")
        return

    print(f"Frame shape: {frame.shape}")

    # Detect
    detections = detector.detect_people(frame)
    print(f"Found {len(detections)} people")

    # Print first few detections
    for i, det in enumerate(detections[:5]):
        x1, y1, x2, y2, conf, class_id = det
        print(f"Person {i+1}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={conf:.3f}")

    # Add blur to anonymize
    frame = cv2.GaussianBlur(frame, (25, 25), 5)

    # Draw
    result = draw_detections(frame, detections)

    # Save a single frame out-of-focus
    output_path = Path("output") / "single_frame_test_out_of_focus.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), result)

    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
