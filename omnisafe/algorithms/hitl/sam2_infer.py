import os
import sys
import cv2
from typing import Union, Optional, Tuple
import numpy as np

from sam2.sam2_stream_predictor import SAM2StreamPredictor


class Sam2Infer:
    def __init__(self, img_height: int = 128, img_width: int = 128):
        """
        Initialize the SAM2 inference object with the specified image dimensions.
        Note the image resolution is the desired one in post-processing, not the input image resolution to SAM2.

        Args:
            img_width (int): The desired width of the output image and mask.
            img_height (int): The desired height of the output image and mask.
        """
        # A trick to ensure config files of sam2 can be found
        original_cwd = os.getcwd()
        sam2_root = "/home/orin-nano/sam2"
        os.chdir(sam2_root)
        if sam2_root not in sys.path:
            sys.path.insert(0, sam2_root)

        # Configuration
        self.model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
        self.checkpoint = '/home/orin-nano/sam2/checkpoints/sam2.1_hiera_small.pt'
        self.img_height = img_height
        self.img_width = img_width

        # Create inference object
        self.sam2_inference = SAM2StreamPredictor(self.model_cfg, self.checkpoint)
        self.frame_processor = None
        self.processed_frames: int = 0

        # Restore original directory
        os.chdir(original_cwd)

    def infer(
        self,
        img: Union[str, np.ndarray],
        mask_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on a single image or frame.
        Args:
            img: The input image, can be a file path or a numpy array.
            mask_path: Optional path to save the output mask image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The processed image and the binary mask.
            image is resized to (img_width, img_height, 3) and mask is binary in uint8 format (img_width, img_height).
        """
        # Load the image, (h, w, 3)
        if isinstance(img, str):
            img_arr = cv2.imread(img)
        else:
            img_arr = img

        # Do inference
        if self.frame_processor is None:
            self.frame_processor = self.sam2_inference.infer_frame_by_frame(img_arr)

            image, mask = next(self.frame_processor)
            print(f'Initial frame processed: {image.shape}, {mask.shape}')
        else:
            image, mask = self.frame_processor.send(img_arr)

        self.processed_frames += 1

        # Resize the image and mask to the desired dimensions
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        # Ensure the mask is binary within unit8 range (0 or 255)
        mask = (mask * 255).astype(np.uint8)

        # Optionally save the mask
        if mask_path is not None:
            cv2.imwrite(mask_path, mask)

        return image, mask


if __name__ == "__main__":
    # Example usage
    video_path = '/home/orin-nano/sam2/notebooks/videos/wabash_upstream_fastforward_60x_512x512.mp4'
    every_n_frames = 10  # Process every 10th frame, to simulate real-time processing for temporal discrete inference

    # Initialize the SAM2 inference object
    sam2_infer = Sam2Infer()

    # Perform inference on the video with frame skipping
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_n_frames == 0:
            # Perform inference
            image, mask = sam2_infer.infer(frame)
            print(f'Processed frame {frame_count}: {image.shape}, {mask.shape}')

            # Display the results
            cv2.imshow('Image', image)
            cv2.imshow('Mask', mask)
            cv2.waitKey(1)  # Wait for a short time to display the images

        frame_count += 1






