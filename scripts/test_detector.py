"""
Test YOLOX detector on sample images/video frames.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.yolox_detector import COCO_CLASSES, YOLOXDetector


def draw_detections(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image
        detections: Array of detections [x1, y1, x2, y2, confidence, class_id]

    Returns:
        Image with drawn bounding boxes
    """
    img_draw = image.copy()

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # Draw rectangle
        color = (0, 255, 0)  # Green for person
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_draw


def test_on_image(detector: YOLOXDetector, image_path: str):
    """Test detector on a single image."""
    print(f"\n{'=' * 60}")
    print(f"Testing on image: {image_path}")
    print('=' * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect people
    print("Running detection...")
    detections = detector.detect_people(image)

    print(f"Found {len(detections)} person(s)")

    # Draw detections
    if len(detections) > 0:
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, _ = det
            print(f"  Person {i+1}: bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf={conf:.3f}")

        result_img = draw_detections(image, detections)

        # Save result
        output_path = Path("output") / "test_detection.jpg"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), result_img)
        print(f"\nResult saved: {output_path}")

        # Display (optional - may not work in some environments)
        cv2.imshow("Detection Result", result_img)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No people detected")


def test_on_video(detector: YOLOXDetector, video_path: str, max_frames: int = 30):
    """Test detector on video frames."""
    print(f"\n{'=' * 60}")
    print(f"Testing on video: {video_path}")
    print('=' * 60)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    print(f"Processing first {max_frames} frames...\n")

    frame_count = 0
    detection_counts = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        detections = detector.detect_people(frame)
        num_people = len(detections)
        detection_counts.append(num_people)

        # Draw detections
        result_frame = draw_detections(frame, detections)

        # Display frame info
        cv2.putText(
            result_frame,
            f"Frame {frame_count+1}/{max_frames} | People: {num_people}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Show frame
        cv2.imshow("Detection Test", result_frame)

        print(f"Frame {frame_count+1}: {num_people} person(s)")

        frame_count += 1

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Statistics
    print(f"\n{'=' * 60}")
    print("STATISTICS")
    print('=' * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Average people per frame: {np.mean(detection_counts):.2f}")
    print(f"Max people in frame: {np.max(detection_counts)}")
    print(f"Min people in frame: {np.min(detection_counts)}")


def main():
    """Main test function."""
    print("=" * 60)
    print("YOLOX DETECTOR TEST")
    print("=" * 60)

    # Initialize detector
    model_path = "models/yolox-s.onnx"

    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please run: python scripts/download_yolox_model.py")
        return

    detector = YOLOXDetector(
        model_path=model_path,
        conf_threshold=0.5,
        nms_threshold=0.45,
        class_filter=[0]  # Person only
    )

    # Check for test data
    videos_dir = Path("data/videos")
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

    if len(sys.argv) > 1:
        # Test on provided file
        test_file = sys.argv[1]
        if test_file.endswith(('.mp4', '.avi', '.mov')):
            test_on_video(detector, test_file, max_frames=50)
        else:
            test_on_image(detector, test_file)
    elif video_files:
        # Test on first video found
        print(f"\n[INFO] Found video: {video_files[0]}")
        test_on_video(detector, str(video_files[0]), max_frames=50)
    else:
        print("\n" + "=" * 60)
        print("TO TEST WITH REAL DATA:")
        print("=" * 60)
        print("  python scripts/test_detector.py <path_to_image_or_video>")
        print("\nOr add videos to data/videos/ and run again")


if __name__ == "__main__":
    main()
