"""
Video reader class for processing video files.
Wrapper around OpenCV VideoCapture with additional features.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path
import time


class VideoReader:
    """
    Video reader wrapper with frame extraction and FPS control.
    """

    def __init__(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        start_frame: int = 0,
        max_frames: Optional[int] = None
    ):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file
            target_fps: Target FPS for processing (None = use video FPS)
            start_frame: Frame number to start from
            max_frames: Maximum number of frames to process (None = all frames)
        """
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.start_frame = start_frame
        self.max_frames = max_frames

        # Verify video exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))

        # Set target FPS
        if self.target_fps is None:
            self.target_fps = self.fps

        # Calculate frame skip for FPS control
        self.frame_skip = max(1, int(self.fps / self.target_fps)) if self.target_fps < self.fps else 1

        # State
        self.current_frame = 0
        self.frames_processed = 0

        # Seek to start frame
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.current_frame = self.start_frame

        print(f"[INFO] Loaded video: {self.video_path.name}")
        print(f"[INFO] Resolution: {self.width}x{self.height}")
        print(f"[INFO] Original FPS: {self.fps:.2f}")
        print(f"[INFO] Target FPS: {self.target_fps:.2f}")
        print(f"[INFO] Frame skip: {self.frame_skip}")
        print(f"[INFO] Total frames: {self.total_frames}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.

        Returns:
            Tuple of (success, frame)
        """
        # Check if reached max frames
        if self.max_frames is not None and self.frames_processed >= self.max_frames:
            return False, None

        # Read frame
        ret, frame = self.cap.read()

        if ret:
            self.current_frame += 1
            self.frames_processed += 1

            # Skip frames if necessary for FPS control
            for _ in range(self.frame_skip - 1):
                self.cap.read()
                self.current_frame += 1

        return ret, frame

    def read_all(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields all frames.

        Yields:
            Frame as numpy array
        """
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def seek(self, frame_number: int) -> bool:
        """
        Seek to specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            Success status
        """
        if 0 <= frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        return False

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame without changing current position.

        Args:
            frame_number: Frame number to get

        Returns:
            Frame or None if failed
        """
        current_pos = self.current_frame
        if self.seek(frame_number):
            ret, frame = self.cap.read()
            self.seek(current_pos)  # Restore position
            return frame if ret else None
        return None

    def reset(self):
        """Reset to beginning of video."""
        self.seek(self.start_frame)
        self.frames_processed = 0

    def get_info(self) -> dict:
        """
        Get video information.

        Returns:
            Dictionary with video properties
        """
        return {
            "path": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "target_fps": self.target_fps,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "frames_processed": self.frames_processed,
            "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0
        }

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.total_frames

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def release(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        """Destructor."""
        self.release()


class VideoWriter:
    """
    Video writer wrapper for saving processed videos.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        fourcc: str = 'mp4v'
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            width: Frame width
            height: Frame height
            fourcc: Four-character code for codec (e.g., 'mp4v', 'XVID')
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            self.fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {self.output_path}")

        self.frames_written = 0

        print(f"[INFO] Created video writer: {self.output_path}")
        print(f"[INFO] Resolution: {self.width}x{self.height} @ {self.fps} FPS")

    def write(self, frame: np.ndarray):
        """
        Write frame to video.

        Args:
            frame: Frame to write (BGR format)
        """
        # Resize if needed
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)
        self.frames_written += 1

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def release(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
            print(f"[INFO] Video saved: {self.output_path} ({self.frames_written} frames)")

    def __del__(self):
        """Destructor."""
        self.release()


class FrameBuffer:
    """
    Simple frame buffer for multi-threaded processing.
    """

    def __init__(self, maxsize: int = 32):
        """
        Initialize frame buffer.

        Args:
            maxsize: Maximum buffer size
        """
        from queue import Queue
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, frame: np.ndarray, block: bool = True, timeout: Optional[float] = None):
        """
        Put frame in buffer.

        Args:
            frame: Frame to add
            block: Block if buffer is full
            timeout: Timeout in seconds
        """
        self.queue.put(frame, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get frame from buffer.

        Args:
            block: Block if buffer is empty
            timeout: Timeout in seconds

        Returns:
            Frame or None
        """
        try:
            return self.queue.get(block=block, timeout=timeout)
        except:
            return None

    def empty(self) -> bool:
        """Check if buffer is empty."""
        return self.queue.empty()

    def full(self) -> bool:
        """Check if buffer is full."""
        return self.queue.full()

    def size(self) -> int:
        """Get current buffer size."""
        return self.queue.qsize()

    def clear(self):
        """Clear buffer."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
