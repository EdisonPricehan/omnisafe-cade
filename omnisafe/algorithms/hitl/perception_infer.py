import torch
import os
import time
import numpy as np
from typing import Optional, Tuple, List, Union
from PIL import Image
from loguru import logger

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(f'Please make sure tensorrt is installed and can be found under "/usr/src/", '
                      f'and "/usr/lib/python3.10/dist-packages/ is in the PYTHONPATH.". '
                      f'If having "version GLIBCXX_3.4.30 not found" error, try upgrading the python version'
                      f'to the latest by "conda install -c conda-forge libstdcxx-ng", {e}.')

try:
    from cuda import cuda
except ImportError as e:
    raise ImportError(f'Please install cuda using "pip install cuda-python", {e}.')


class PerceptionInfer:
    height: int = 128
    width: int = 128

    def __init__(self, engine_path: str):
        """
        Initialize the PerceptionInfer class with the path to the TensorRT engine.

        Args:
            engine_path (str): The file path to the TensorRT engine.
        """
        self.engine_path = engine_path
        assert os.path.exists(self.engine_path), f'{self.engine_path} does not exist.'

        self.engine = None
        self.context = None

        self._load_engine()

    def _load_engine(self):
        """
        Load the TensorRT engine and create a context.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            logger.info(f'TensorRT engine loaded from {self.engine_path}.')

    @staticmethod
    def load_and_process_img(img_path: str, h: int = height, w: int = width) -> np.ndarray:
        """
        Load the rgb image from path, then process it as numpy array, as input to trt engine.
        """
        assert os.path.exists(img_path), f'{img_path} does not exist!'

        # Read image
        img = Image.open(img_path)

        # Resize image
        img = img.resize(size=(w, h), resample=Image.Resampling.NEAREST)

        # Convert to numpy array
        img_arr = np.array(img)

        # Transpose to (1, C, H, W) and normalize to range [0, 1]
        img_arr = img_arr.transpose((2, 0, 1))[None].astype(np.float32) / 255
        # print(f'{img_arr.shape=}')

        return img_arr

    @staticmethod
    def trt_inference(engine, context, data):
        """
        Low-level inference of trt engine.
        """
        nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
        nOutput = engine.num_bindings - nInput

        # For debugging
        # print('nInput:', nInput)
        # print('nOutput:', nOutput)
        # for i in range(nInput):
        #     print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_dtype(i))
        # for i in range(nInput, nInput + nOutput):
        #     print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_dtype(i))

        bufferH = []
        bufferH.append(np.ascontiguousarray(data.reshape(-1)))

        for i in range(nInput, nInput + nOutput):
            bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))

        bufferD = []
        for i in range(nInput + nOutput):
            bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)

        for b in bufferD:
            cuda.cuMemFree(b)

        return bufferH

    @staticmethod
    def post_process_mask(
        trt_output: List[float],
        h: int = height,
        w: int = width,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Post-process trt inferenced mask to the desired shape and data type.

        Args:
            trt_output (List[float]): The output from the TensorRT inference, expected to be a flat list of floats.
            h (int): The height of the output mask.
            w (int): The width of the output mask.
            return_tensor (bool): If True, return a torch tensor; otherwise, return a numpy array.

        Returns:
            Union[np.ndarray, torch.Tensor]: The processed mask, either as a numpy array or a torch tensor.
        """
        # Resize
        pred_mask = np.array(trt_output).reshape(h, w)
        # print(f'{type(pred_mask)=} {pred_mask.shape=} {pred_mask.dtype=}')

        # Convert to torch, convert to probability, threshold it, change dtype to uint8
        pred_mask_torch = torch.tensor(pred_mask, dtype=torch.float32)
        pred_mask_torch = pred_mask_torch.sigmoid()
        pred_mask_torch = ((pred_mask_torch > 0.5).float() * 255).to(torch.uint8)

        if return_tensor:
            return pred_mask_torch
        else:
            return pred_mask_torch.detach().cpu().numpy()

    def infer(
        self,
        img: Union[str, np.ndarray],
        mask_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference an image with trt engine, outputs the resized image and water mask.

        Args:
            img (Union[str, np.ndarray]): The input image, either a file path or a numpy array of shape (C, H, W).
            mask_path (Optional[str]): If provided, the predicted mask will be saved to this path.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the resized image array and the predicted water mask.

        Raises:
            AssertionError: If the TensorRT engine or context is not loaded.
            ValueError: If the input image is not in the expected format.
        """
        assert self.engine is not None and self.context is not None, \
            f'TensorRT engine or context is not loaded. Please call _load_engine() first.'

        # Load image, resize to desired shape, then convert to numpy array
        if isinstance(img, str):
            img_arr = PerceptionInfer.load_and_process_img(img)
        else:  # Assume the image_arr is of shape (C, H, W) and of uint8 type, need assertions TODO
            img_arr = img
            if len(img_arr.shape) < 4 and img_arr.dtype == np.float32 and img_arr.max() <= 1:
                img_arr = img_arr[None]  # [1, C, H, W]
            else:
                raise ValueError(f'Input image numpy array needs to be from FluvialDataset class (CHW, normalized), '
                                 f'given image shape: {img_arr.shape}, dtype: {img_arr.dtype}.')

        # context.set_binding_shape(0, img_arr.shape)  # This seems not necessary

        trt_start_time = time.time()

        # Do trt inference
        trt_outputs = PerceptionInfer.trt_inference(engine=self.engine, context=self.context, data=img_arr)
        # print(f'{type(trt_outputs)=} {len(trt_outputs)=} {type(trt_outputs[1])=}')

        # Do post-processing to get water mask
        pred_mask_np = PerceptionInfer.post_process_mask(trt_outputs[1], return_tensor=False)

        trt_end_time = time.time()
        trt_infer_duration = trt_end_time - trt_start_time
        logger.info(f'Inference finished. Time cost: {trt_infer_duration:.3f} seconds.')

        # Save mask to image
        if mask_path is not None:
            pred_mask_np_hwc = np.repeat(pred_mask_np[:, :, np.newaxis], 3, axis=2)  # Convert to HWC format
            mask = Image.fromarray(pred_mask_np_hwc)
            mask.save(mask_path)
            logger.info(f'Predicted mask saved to {mask_path}.')

        return img_arr, pred_mask_np


if __name__ == '__main__':
    # Example usage
    engine_path = '/home/orin-nano/Aerial-Fluvial-Semantic-Segmentation/src/models/unet-resnet34-128x128-fp16.trt'
    img_path = './wildcat_downward_0000.jpg'
    mask_save_path = './predicted_mask.png'

    perception_infer = PerceptionInfer(engine_path=engine_path)

    # Infer from saved image
    img_arr, pred_mask = perception_infer.infer(img=img_path, mask_path=mask_save_path)
    logger.info(f'Image shape: {img_arr.shape}, Predicted mask shape: {pred_mask.shape}')

    # Infer from random array
    dummy_img = np.random.rand(3, 128, 128).astype(np.float32)
    img_arr, pred_mask = perception_infer.infer(img=dummy_img, mask_path=None)
    print(f'Image shape: {img_arr.shape}, Predicted mask shape: {pred_mask.shape}')

