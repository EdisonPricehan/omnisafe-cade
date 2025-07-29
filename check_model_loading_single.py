"""
Test script to check if a model can be loaded correctly.
"""

import os
import torch


if __name__ == "__main__":
    # pth_path = 'examples/models/torch_save/epoch-350.pt'  # Successful
    # pth_path = 'examples/models/torch_save/real-episode-000-hitl-True-loss-Indirect-20250629_165120.pt'  # Successful
    # pth_path = 'examples/models/torch_save/wabash_0630/upstream/real-episode-001-hitl-True-loss-Indirect-20250630_101703.pt'  # Failed
    # pth_path = 'examples/models/torch_save/real-episode-000-hitl-True-loss-Indirect-20250729_121811.pt'  # Successful
    # pth_path = 'examples/models/torch_save/real-episode-001-hitl-True-loss-Indirect-20250729_121912.pt'  # Successful
    # pth_path = 'examples/models/torch_save/real-episode-000-hitl-True-loss-Indirect-20250729_134513.pt'  # Successful
    pth_path = 'examples/models/torch_save/real-episode-000-hitl-True-loss-Indirect-20250729_132336.pt'  # Successful


    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file {pth_path} does not exist.")

    try:
        model_params = torch.load(pth_path)
        print(f"Model loaded successfully from {pth_path}.")
    except Exception as e:
        print(f"Failed to load model from {pth_path}. Error: {e}")
