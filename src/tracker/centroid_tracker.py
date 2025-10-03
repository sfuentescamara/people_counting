"""
Simple centroid-based object tracker.
Tracks objects across frames using centroid distance.
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from typing import List, Tuple, Dict


class CentroidTracker:
    """
    Simple centroid-based tracker for multi-object tracking.
    Assigns unique IDs to detected objects and tracks them across frames.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        """
        Initialize centroid tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear
                           before being deregistered
            max_distance: Maximum distance for matching objects between frames
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> centroid
        self.disappeared = OrderedDict()  # object_id -> disappeared_count
        self.bboxes = OrderedDict()  # object_id -> bbox [x1, y1, x2, y2]

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: np.ndarray, bbox: np.ndarray):
        """
        Register a new object with a unique ID.

        Args:
            centroid: Centroid coordinates [x, y]
            bbox: Bounding box [x1, y1, x2, y2]
        """
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        """
        Deregister an object ID.

        Args:
            object_id: ID to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]

    def update(self, detections: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Update tracked objects with new detections.

        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, class_id]

        Returns:
            Dictionary mapping object_id -> (centroid, bbox)
        """
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self._get_objects()

        # Calculate centroids from bounding boxes
        input_centroids = np.zeros((len(detections), 2), dtype="float")
        input_bboxes = detections[:, :4]

        for i, bbox in enumerate(input_bboxes):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids[i] = (cx, cy)

        # If no existing objects, register all new detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            # Match existing objects with new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between each existing object and new detections
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find best matches
            # Sort by distance (row-wise minimum)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                # Skip if already used
                if row in used_rows or col in used_cols:
                    continue

                # Skip if distance is too large
                if D[row, col] > self.max_distance:
                    continue

                # Update existing object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Find unmatched objects (disappeared)
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new detections
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])

        return self._get_objects()

    def _get_objects(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get current tracked objects.

        Returns:
            Dictionary mapping object_id -> (centroid, bbox)
        """
        result = OrderedDict()
        for object_id in self.objects.keys():
            result[object_id] = (
                self.objects[object_id],
                self.bboxes[object_id]
            )
        return result

    def get_count(self) -> int:
        """Get current number of tracked objects."""
        return len(self.objects)

    def reset(self):
        """Reset tracker state."""
        self.next_object_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.bboxes.clear()
