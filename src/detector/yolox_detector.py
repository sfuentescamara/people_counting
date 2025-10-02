"""
YOLOX Object Detector using ONNX Runtime.
Detects people in images/frames.
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple, List, Optional
from pathlib import Path


class YOLOXDetector:
    """
    YOLOX object detector wrapper for ONNX models.
    Specifically optimized for person detection (COCO class 0).
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        class_filter: Optional[List[int]] = None
    ):
        """
        Initialize YOLOX detector.

        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            class_filter: List of class IDs to detect (None = all classes)
                         For person detection only, use [0]
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_filter = class_filter

        # Verify model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Initialize ONNX Runtime session
        print(f"[INFO] Loading YOLOX model: {self.model_path}")
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']  # Can add 'CUDAExecutionProvider' for GPU
        )

        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Input size: {self.input_size}")
        print(f"[INFO] Confidence threshold: {self.conf_threshold}")
        print(f"[INFO] NMS threshold: {self.nms_threshold}")
        if self.class_filter:
            print(f"[INFO] Class filter: {self.class_filter}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, int, int]]:
        """
        Preprocess image for YOLOX model.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Tuple of (preprocessed_image, (scale_ratio, pad_w, pad_h))
        """
        # Get original image size
        img_h, img_w = image.shape[:2]

        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size[0] / img_w, self.input_size[1] / img_h)

        # Resize image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image (letterbox - top-left alignment for YOLOX)
        padded_img = np.ones((self.input_size[1], self.input_size[0], 3), dtype=np.uint8) * 114
        padded_img[:new_h, :new_w] = resized_img

        # No padding offset for top-left alignment
        pad_w = 0
        pad_h = 0

        # Convert to RGB and normalize
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # Transpose to CHW format and add batch dimension
        input_img = padded_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

        return input_img, (scale, pad_w, pad_h)

    def _generate_grids_and_strides(self):
        """
        Generate grid coordinates and strides for YOLOX output decoding.
        Based on official YOLOX implementation.
        """
        strides = [8, 16, 32]
        hsizes = [self.input_size[1] // stride for stride in strides]
        wsizes = [self.input_size[0] // stride for stride in strides]

        grids = []
        expanded_strides = []

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        return grids, expanded_strides

    def postprocess(
        self,
        outputs: np.ndarray,
        preprocess_info: Tuple[float, int, int],
        img_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Post-process YOLOX outputs to get bounding boxes.
        Uses official YOLOX postprocessing logic.

        Args:
            outputs: Raw model outputs
            preprocess_info: Tuple of (scale, pad_w, pad_h) from preprocessing
            img_shape: Original image shape (height, width)

        Returns:
            Array of detections [x1, y1, x2, y2, confidence, class_id]
        """
        scale, pad_w, pad_h = preprocess_info

        # YOLOX output format: [batch, num_predictions, 85]
        # 85 = 4 (bbox) + 1 (objectness) + 80 (class scores)
        predictions = outputs[0].copy()  # Remove batch dimension and copy

        # Decode bounding boxes using official YOLOX postprocessing
        # Generate grids and strides
        grids, expanded_strides = self._generate_grids_and_strides()

        # Decode box coordinates from grid-relative to pixel coordinates
        # predictions[..., :2] are offsets from grid cell centers
        # predictions[..., 2:4] are log-space width/height
        predictions[..., :2] = (predictions[..., :2] + grids) * expanded_strides
        predictions[..., 2:4] = np.exp(predictions[..., 2:4]) * expanded_strides

        # Now predictions[:, :4] are [center_x, center_y, width, height] in pixel coordinates
        # relative to the 640x640 input image

        boxes = predictions[:, :4]
        obj_conf = predictions[:, 4:5]
        class_scores = predictions[:, 5:]
        class_conf = obj_conf * class_scores
        class_pred = np.argmax(class_conf, axis=1)
        conf = np.max(class_conf, axis=1)

        # Filter by confidence threshold
        mask = conf >= self.conf_threshold
        boxes = boxes[mask]
        conf = conf[mask]
        class_pred = class_pred[mask]

        # Filter by class (e.g., person only)
        if self.class_filter is not None:
            class_mask = np.isin(class_pred, self.class_filter)
            boxes = boxes[class_mask]
            conf = conf[class_mask]
            class_pred = class_pred[class_mask]

        if len(boxes) == 0:
            return np.array([])

        # Scale boxes from 640x640 input to original image size
        boxes = boxes / scale

        # Convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Clip to image boundaries
        x1 = np.clip(x1, 0, img_shape[1])
        y1 = np.clip(y1, 0, img_shape[0])
        x2 = np.clip(x2, 0, img_shape[1])
        y2 = np.clip(y2, 0, img_shape[0])

        boxes_corner = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = self._nms(boxes_corner, conf, self.nms_threshold)

        # Combine results
        detections = np.concatenate([
            boxes_corner[indices],
            conf[indices, np.newaxis],
            class_pred[indices, np.newaxis]
        ], axis=1)

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """
        Non-Maximum Suppression.

        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            threshold: IoU threshold

        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            denominator = areas[i] + areas[order[1:]] - inter
            # Avoid division by zero
            iou = np.divide(inter, denominator, out=np.zeros_like(inter), where=denominator != 0)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects in an image.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Array of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Preprocess
        input_img, preprocess_info = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_img})

        # Post-process
        detections = self.postprocess(outputs[0], preprocess_info, image.shape[:2])

        return detections

    def detect_people(self, image: np.ndarray) -> np.ndarray:
        """
        Detect people (COCO class 0) in an image.
        Convenience method that sets class filter to [0].

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Array of person detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Temporarily set class filter to person only
        original_filter = self.class_filter
        self.class_filter = [0]

        detections = self.detect(image)

        # Restore original filter
        self.class_filter = original_filter

        return detections


# COCO class names for reference
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
