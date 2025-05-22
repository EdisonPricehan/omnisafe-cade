import cv2
import numpy as np
from typing import Optional, Tuple, Iterator


class VideoReader:
    """
    A simple video reader that yields frames as NumPy arrays, with optional resizing.
    """

    def __init__(self, filepath: str, resize: Optional[Tuple[int, int]] = None):
        """
        Args:
            filepath: Path to the video file (e.g. .mp4).
            resize: If given, a (width, height) tuple to resize each frame to.
        """
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise IOError(f'Cannot open video file {filepath}.')
        self.resize = resize

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.cap.read()

        if not ret:
            self.cap.release()
            raise StopIteration

        # frame is a H×W×3 BGR image as NumPy array, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)  # resize to (width, height)

        return frame

    def release(self) -> None:
        """Manually release the video resource."""
        self.cap.release()


if __name__ == '__main__':
    video_path: str = '/home/edison/afid_videos/wildcat/output_2fps_720x480.mp4'
    shape = (128, 128)

    reader = VideoReader(
        filepath=video_path,
        resize=shape,
    )

    for idx, frame in enumerate(reader):
        print(f'{frame.shape=}')
        cv2.imshow('Frame', frame)
        cv2.waitKey(10)

    reader.release()
