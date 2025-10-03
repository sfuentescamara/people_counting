"""
Line-crossing counter for people counting.
Counts objects crossing a virtual line in the frame.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from enum import Enum


class Direction(Enum):
    """Direction of crossing."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    IN = "in"
    OUT = "out"


class LineCounter:
    """
    Counts objects crossing a virtual line in the frame.
    Supports horizontal and vertical lines with directional counting.
    """

    def __init__(
        self,
        line_position: Tuple[Tuple[int, int], Tuple[int, int]],
        direction_labels: Tuple[str, str] = ("in", "out")
    ):
        """
        Initialize line counter.

        Args:
            line_position: Line endpoints ((x1, y1), (x2, y2))
            direction_labels: Labels for two directions (e.g., ("in", "out"))
        """
        self.line_start = np.array(line_position[0])
        self.line_end = np.array(line_position[1])
        self.direction_labels = direction_labels

        # Track object positions
        self.object_positions = {}  # object_id -> last_centroid
        self.crossed_objects = {}  # object_id -> direction

        # Counters
        self.counts = {
            direction_labels[0]: 0,
            direction_labels[1]: 0
        }

        # Determine if line is horizontal or vertical
        self.is_horizontal = abs(self.line_start[1] - self.line_end[1]) < abs(
            self.line_start[0] - self.line_end[0]
        )

    def update(self, tracked_objects: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, int]:
        """
        Update counter with tracked objects.

        Args:
            tracked_objects: Dictionary mapping object_id -> (centroid, bbox)

        Returns:
            Current counts dictionary
        """
        for object_id, (centroid, bbox) in tracked_objects.items():
            # Check if this is a new object
            if object_id not in self.object_positions:
                self.object_positions[object_id] = centroid
                continue

            # Get previous and current positions
            prev_centroid = self.object_positions[object_id]
            curr_centroid = centroid

            # Check if crossed the line
            direction = self._check_line_crossing(prev_centroid, curr_centroid, object_id)

            if direction is not None:
                # Count the crossing
                self.counts[direction] += 1
                self.crossed_objects[object_id] = direction

            # Update position
            self.object_positions[object_id] = curr_centroid

        # Clean up positions for objects that no longer exist
        current_ids = set(tracked_objects.keys())
        stored_ids = set(self.object_positions.keys())
        removed_ids = stored_ids - current_ids

        for object_id in removed_ids:
            del self.object_positions[object_id]
            if object_id in self.crossed_objects:
                del self.crossed_objects[object_id]

        return self.counts.copy()

    def _check_line_crossing(
        self,
        prev_pos: np.ndarray,
        curr_pos: np.ndarray,
        object_id: int
    ) -> Optional[str]:
        """
        Check if object crossed the line.

        Args:
            prev_pos: Previous centroid position [x, y]
            curr_pos: Current centroid position [x, y]
            object_id: Object ID

        Returns:
            Direction string if crossed, None otherwise
        """
        # Skip if already counted this object
        if object_id in self.crossed_objects:
            return None

        # Check line crossing using line intersection
        if self.is_horizontal:
            # Horizontal line - check if y crossed the line
            line_y = (self.line_start[1] + self.line_end[1]) / 2

            if prev_pos[1] < line_y <= curr_pos[1]:
                # Crossed downward
                return self.direction_labels[1]
            elif prev_pos[1] > line_y >= curr_pos[1]:
                # Crossed upward
                return self.direction_labels[0]
        else:
            # Vertical line - check if x crossed the line
            line_x = (self.line_start[0] + self.line_end[0]) / 2

            if prev_pos[0] < line_x <= curr_pos[0]:
                # Crossed rightward
                return self.direction_labels[1]
            elif prev_pos[0] > line_x >= curr_pos[0]:
                # Crossed leftward
                return self.direction_labels[0]

        return None

    def get_total_count(self) -> int:
        """Get total count across all directions."""
        return sum(self.counts.values())

    def reset(self):
        """Reset all counters and tracking."""
        self.object_positions.clear()
        self.crossed_objects.clear()
        for key in self.counts.keys():
            self.counts[key] = 0

    def get_stats(self) -> Dict:
        """Get detailed statistics."""
        return {
            "counts": self.counts.copy(),
            "total": self.get_total_count(),
            "active_objects": len(self.object_positions),
            "crossed_objects": len(self.crossed_objects)
        }


class ZoneCounter:
    """
    Counts objects entering/exiting a rectangular zone.
    """

    def __init__(
        self,
        zone: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        direction_labels: Tuple[str, str] = ("enter", "exit")
    ):
        """
        Initialize zone counter.

        Args:
            zone: Zone rectangle (x1, y1, x2, y2)
            direction_labels: Labels for enter/exit
        """
        self.zone = zone
        self.direction_labels = direction_labels

        # Track object states
        self.object_in_zone = {}  # object_id -> bool
        self.crossed_objects = {}  # object_id -> direction

        # Counters
        self.counts = {
            direction_labels[0]: 0,  # enter
            direction_labels[1]: 0   # exit
        }

    def update(self, tracked_objects: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, int]:
        """
        Update counter with tracked objects.

        Args:
            tracked_objects: Dictionary mapping object_id -> (centroid, bbox)

        Returns:
            Current counts dictionary
        """
        x1, y1, x2, y2 = self.zone

        for object_id, (centroid, bbox) in tracked_objects.items():
            cx, cy = centroid

            # Check if centroid is in zone
            in_zone = (x1 <= cx <= x2) and (y1 <= cy <= y2)

            # Check for zone crossing
            if object_id in self.object_in_zone:
                was_in_zone = self.object_in_zone[object_id]

                if not was_in_zone and in_zone:
                    # Entered zone
                    if object_id not in self.crossed_objects:
                        self.counts[self.direction_labels[0]] += 1
                        self.crossed_objects[object_id] = self.direction_labels[0]

                elif was_in_zone and not in_zone:
                    # Exited zone
                    if object_id not in self.crossed_objects:
                        self.counts[self.direction_labels[1]] += 1
                        self.crossed_objects[object_id] = self.direction_labels[1]

            # Update state
            self.object_in_zone[object_id] = in_zone

        # Clean up removed objects
        current_ids = set(tracked_objects.keys())
        stored_ids = set(self.object_in_zone.keys())
        removed_ids = stored_ids - current_ids

        for object_id in removed_ids:
            del self.object_in_zone[object_id]
            if object_id in self.crossed_objects:
                del self.crossed_objects[object_id]

        return self.counts.copy()

    def get_total_count(self) -> int:
        """Get total count across all directions."""
        return sum(self.counts.values())

    def reset(self):
        """Reset all counters and tracking."""
        self.object_in_zone.clear()
        self.crossed_objects.clear()
        for key in self.counts.keys():
            self.counts[key] = 0

    def get_stats(self) -> Dict:
        """Get detailed statistics."""
        return {
            "counts": self.counts.copy(),
            "total": self.get_total_count(),
            "objects_in_zone": sum(1 for in_zone in self.object_in_zone.values() if in_zone),
            "crossed_objects": len(self.crossed_objects)
        }
