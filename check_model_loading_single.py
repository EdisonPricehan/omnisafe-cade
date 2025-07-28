"""
Test script to check if a model can be loaded correctly.
"""

import os
import torch


if __name__ == "__main__":
    pth_path = 'examples/models/torch_save/epoch-350.pt'

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file {pth_path} does not exist.")

    try:
        model_params = torch.load(pth_path)
        print(f"Model loaded successfully from {pth_path}.")
    except Exception as e:
        print(f"Failed to load model from {pth_path}. Error: {e}")
