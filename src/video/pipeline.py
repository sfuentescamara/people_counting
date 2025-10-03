"""
Video processing pipeline: read → detect → display.
Simple synchronous pipeline for demo purposes.
"""

import cv2
import numpy as np
import time
from typing import Optional, Callable
from pathlib import Path

from src.video.video_reader import VideoReader, VideoWriter
from src.detector.yolox_detector import YOLOXDetector, COCO_CLASSES


class VideoProcessingPipeline:
    """
    Simple video processing pipeline.
    Reads frames, runs detection, and displays/saves results.
    """

    def __init__(
        self,
        detector: YOLOXDetector,
        show_display: bool = True,
        save_output: bool = False,
        output_dir: str = "output/videos"
    ):
        """
        Initialize pipeline.

        Args:
            detector: YOLOX detector instance
            show_display: Whether to show live display
            save_output: Whether to save output video
            output_dir: Directory for output videos
        """
        self.detector = detector
        self.show_display = show_display
        self.save_output = save_output
        self.output_dir = Path(output_dir)

        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        self.processing_times = []

        # Create output directory
        if self.save_output:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on frame.

        Args:
            frame: Input frame
            detections: Detections array [x1, y1, x2, y2, conf, class_id]
            show_labels: Whether to show labels

        Returns:
            Frame with drawn detections
        """
        frame_draw = frame.copy()

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)

            # Color based on class (green for person)
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)

            # Draw rectangle
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                # Draw label
                label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame_draw,
                    (x1, y1 - text_height - baseline - 2),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    frame_draw,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        return frame_draw

    def draw_info(
        self,
        frame: np.ndarray,
        frame_number: int,
        num_detections: int,
        fps: float
    ) -> np.ndarray:
        """
        Draw processing info on frame.

        Args:
            frame: Input frame
            frame_number: Current frame number
            num_detections: Number of detections
            fps: Processing FPS

        Returns:
            Frame with info overlay
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw text
        y_offset = 40
        cv2.putText(
            frame_with_overlay,
            f"Frame: {frame_number}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y_offset += 30
        cv2.putText(
            frame_with_overlay,
            f"People: {num_detections}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y_offset += 30
        cv2.putText(
            frame_with_overlay,
            f"FPS: {fps:.1f}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return frame_with_overlay

    def process_video(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        frame_callback: Optional[Callable] = None
    ) -> dict:
        """
        Process a video file.

        Args:
            video_path: Path to video file
            target_fps: Target processing FPS (None = use video FPS)
            max_frames: Maximum frames to process (None = all)
            frame_callback: Optional callback function(frame, detections)

        Returns:
            Processing statistics dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"Processing video: {video_path}")
        print('=' * 60)

        # Reset statistics
        self.total_frames = 0
        self.total_detections = 0
        self.processing_times = []

        # Create video reader
        reader = VideoReader(video_path, target_fps=target_fps, max_frames=max_frames)

        # Create video writer if saving
        writer = None
        if self.save_output:
            output_path = self.output_dir / f"{Path(video_path).stem}_processed.mp4"
            writer = VideoWriter(
                str(output_path),
                reader.target_fps,
                reader.width,
                reader.height
            )

        # Processing loop
        try:
            while True:
                # Read frame
                ret, frame = reader.read()
                if not ret:
                    break

                start_time = time.time()

                # Detect people
                detections = self.detector.detect_people(frame)
                num_people = len(detections)

                # Draw detections
                frame_with_detections = self.draw_detections(frame, detections)

                # Calculate FPS
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                current_fps = 1.0 / process_time if process_time > 0 else 0

                # Draw info overlay
                frame_display = self.draw_info(
                    frame_with_detections,
                    reader.current_frame,
                    num_people,
                    current_fps
                )

                # Update statistics
                self.total_frames += 1
                self.total_detections += num_people

                # Display
                if self.show_display:
                    cv2.imshow("People Counting Demo", frame_display)

                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[INFO] User requested quit")
                        break
                    elif key == ord(' '):
                        # Pause
                        print("[INFO] Paused - press any key to continue")
                        cv2.waitKey(0)

                # Save
                if writer is not None:
                    writer.write(frame_display)

                # Callback
                if frame_callback is not None:
                    frame_callback(frame, detections)

                # Progress
                if self.total_frames % 30 == 0:
                    avg_fps = 1.0 / np.mean(self.processing_times[-30:])
                    print(f"Processed {self.total_frames} frames | Avg FPS: {avg_fps:.1f} | People: {num_people}")

        finally:
            # Cleanup
            reader.release()
            if writer is not None:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

        # Calculate statistics
        stats = self._calculate_statistics(reader)

        # Print summary
        self._print_summary(stats)

        return stats

    def _calculate_statistics(self, reader: VideoReader) -> dict:
        """Calculate processing statistics."""
        avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        avg_people = self.total_detections / self.total_frames if self.total_frames > 0 else 0

        return {
            "video_path": str(reader.video_path),
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_people_per_frame": avg_people,
            "avg_processing_time_ms": avg_process_time * 1000,
            "avg_fps": avg_fps,
            "video_fps": reader.fps,
            "target_fps": reader.target_fps,
            "resolution": f"{reader.width}x{reader.height}"
        }

    def _print_summary(self, stats: dict):
        """Print processing summary."""
        print(f"\n{'=' * 60}")
        print("PROCESSING SUMMARY")
        print('=' * 60)
        print(f"Video: {Path(stats['video_path']).name}")
        print(f"Resolution: {stats['resolution']}")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total people detected: {stats['total_detections']}")
        print(f"Average people per frame: {stats['avg_people_per_frame']:.2f}")
        print(f"Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Original video FPS: {stats['video_fps']:.2f}")
        print('=' * 60)
