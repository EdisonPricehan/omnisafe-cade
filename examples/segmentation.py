import os
import torch
import numpy as np
from PIL import Image

from seg_model import LitSegModel as LSM


class Segmenter:
    def __init__(
        self,
        filepath: str,
        architecture: str = 'Unet',
        encoder: str = 'resnet34',
    ):
        assert os.path.exists(filepath), f'{filepath} does not exist.'
        assert filepath.endswith(('pt', 'pth')), f'{filepath} should be a pytorch model.'

        # Define pth model
        self.model = LSM(arch=architecture, encoder_name=encoder, in_channels=3, out_classes=1)

        # Load state_dict from pth model
        state_dict = torch.load(filepath)

        # Populate model with loaded state_dict
        self.model.model.load_state_dict(state_dict)
        self.model.eval()
        print(f'Semantic segmentation model has been loaded.')

    def segment(self, img: np.ndarray, binarize: bool = True) -> torch.Tensor:
        img_tensor = torch.as_tensor(img, dtype=torch.float32) / 255

        assert img_tensor.dim() == 3, f'RGN image should be of dimension 3, given dim: {img_tensor.dim()}.'

        if img_tensor.shape[-1] < 4:
            img_tensor = img_tensor.permute(2, 0, 1)

        img_tensor = img_tensor.unsqueeze(0)  # [B, C, H, W]
        # print(f'{img_tensor.shape=}')

        mask = self.model(img_tensor)  # logits

        if binarize:
            mask = self.model.logits_to_mask(mask, uint8=True)
        # print(f'{mask.shape=}')

        return mask


if __name__ == '__main__':
    # Define paths
    model_path: str = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/src/models/unet-resnet34-128x128.pth'
    img_path: str = '../images/wildcat_forward_0018.jpg'

    segmenter = Segmenter(filepath=model_path, architecture='Unet', encoder='resnet34')

    img_pil = Image.open(img_path)

    img_pil = img_pil.resize(size=(128, 128), resample=Image.Resampling.NEAREST)

    img_arr = np.array(img_pil)

    mask = segmenter.segment(img_arr, binarize=True)

    img_pil.show('Image')
    mask_arr = mask.detach().squeeze(0).squeeze(0).numpy()
    # print(f'{mask_arr=}')
    mask_pil = Image.fromarray(mask_arr, mode='L')
    mask_pil.show('Mask')
