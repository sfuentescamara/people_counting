"""
Debug YOLOX detector to understand output format.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.yolox_detector import YOLOXDetector


def main():
    """Debug detector."""
    print("=" * 60)
    print("YOLOX DETECTOR DEBUG")
    print("=" * 60)

    # Load model
    model_path = "models/yolox-s.onnx"
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        return

    # Initialize detector
    detector = YOLOXDetector(
        model_path=model_path,
        conf_threshold=0.5,
        nms_threshold=0.45,
        class_filter=None  # All classes for debugging
    )

    # Load test image
    video_path = "data/videos/mall_stairs.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read frame")
        return

    print(f"\nOriginal frame shape: {frame.shape}")
    print(f"Original frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Preprocess
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    input_img, preprocess_info = detector.preprocess(frame)
    scale, pad_w, pad_h = preprocess_info

    print(f"Input shape: {input_img.shape}")
    print(f"Scale: {scale}")
    print(f"Padding: ({pad_w}, {pad_h})")

    # Run inference
    print("\n" + "=" * 60)
    print("INFERENCE")
    print("=" * 60)
    outputs = detector.session.run(detector.output_names, {detector.input_name: input_img})

    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
        print(f"Output {i} dtype: {output.dtype}")
        print(f"Output {i} min/max: {output.min():.3f} / {output.max():.3f}")

    # Look at raw predictions
    print("\n" + "=" * 60)
    print("RAW PREDICTIONS (first output)")
    print("=" * 60)
    predictions = outputs[0][0]  # Remove batch dimension
    print(f"Predictions shape: {predictions.shape}")

    # Check first few predictions
    print(f"\nFirst 3 predictions:")
    for i in range(min(3, len(predictions))):
        pred = predictions[i]
        print(f"\nPrediction {i}:")
        print(f"  BBox (x,y,w,h): {pred[0]:.2f}, {pred[1]:.2f}, {pred[2]:.2f}, {pred[3]:.2f}")
        print(f"  Objectness: {pred[4]:.4f}")
        print(f"  Max class score: {pred[5:].max():.4f}")
        print(f"  Class with max score: {pred[5:].argmax()}")

    # Check predictions with high confidence
    print(f"\n" + "=" * 60)
    print("HIGH CONFIDENCE PREDICTIONS")
    print("=" * 60)

    obj_conf = predictions[:, 4:5]
    class_scores = predictions[:, 5:]
    class_conf = obj_conf * class_scores
    conf = np.max(class_conf, axis=1)
    class_pred = np.argmax(class_conf, axis=1)

    high_conf_mask = conf >= 0.3  # Lower threshold for debugging
    high_conf_boxes = predictions[high_conf_mask, :4]
    high_conf_scores = conf[high_conf_mask]
    high_conf_classes = class_pred[high_conf_mask]

    print(f"Found {len(high_conf_boxes)} predictions with conf >= 0.3")

    if len(high_conf_boxes) > 0:
        print(f"\nFirst 5 high-confidence predictions:")
        for i in range(min(5, len(high_conf_boxes))):
            print(f"\nPrediction {i}:")
            print(f"  BBox (x,y,w,h): {high_conf_boxes[i]}")
            print(f"  Confidence: {high_conf_scores[i]:.4f}")
            print(f"  Class: {high_conf_classes[i]} (person=0)")

            # Try manual coordinate conversion
            x_center, y_center, width, height = high_conf_boxes[i]
            print(f"  Before scaling:")
            print(f"    Center: ({x_center:.2f}, {y_center:.2f})")
            print(f"    Size: ({width:.2f}, {height:.2f})")

            # Remove padding
            x_center_no_pad = x_center - pad_w
            y_center_no_pad = y_center - pad_h
            print(f"  After removing padding:")
            print(f"    Center: ({x_center_no_pad:.2f}, {y_center_no_pad:.2f})")

            # Scale back
            x_center_scaled = x_center_no_pad / scale
            y_center_scaled = y_center_no_pad / scale
            width_scaled = width / scale
            height_scaled = height / scale
            print(f"  After scaling back:")
            print(f"    Center: ({x_center_scaled:.2f}, {y_center_scaled:.2f})")
            print(f"    Size: ({width_scaled:.2f}, {height_scaled:.2f})")

            # Convert to corners
            x1 = x_center_scaled - width_scaled / 2
            y1 = y_center_scaled - height_scaled / 2
            x2 = x_center_scaled + width_scaled / 2
            y2 = y_center_scaled + height_scaled / 2
            print(f"  Final corners: ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f})")

    # Now run full detection
    print("\n" + "=" * 60)
    print("FULL DETECTION PIPELINE")
    print("=" * 60)
    detections = detector.detect(frame)
    print(f"Found {len(detections)} detections")

    if len(detections) > 0:
        print(f"\nFirst 3 detections:")
        for i in range(min(3, len(detections))):
            det = detections[i]
            print(f"\nDetection {i}:")
            print(f"  BBox: {det[:4]}")
            print(f"  Confidence: {det[4]:.4f}")
            print(f"  Class: {int(det[5])}")


if __name__ == "__main__":
    main()
