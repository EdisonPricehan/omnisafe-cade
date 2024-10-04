import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import csv
from typing import Union
import matplotlib.pyplot as plt

from encoder.vae_v1 import VAE
from omnisafe.models.dynamics.ldm import LatentDynamicsModel
from omnisafe.models.dynamics.ldm_mlp import LatentDynamicsModelMLP
from omnisafe.utils.riverine_dataset import RiverDataset, get_train_test_datasets
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.math import l1_loss, iou_loss
from omnisafe.utils.patchification import get_patchified_mask, inflate_patch_mask


def encode_input(
    vae: VAE,
    rgb_cur: torch.Tensor,
    mask_cur: torch.Tensor,
) -> torch.Tensor:
    """
    Get VAE encoding (mean vector) for vision input
    Args:
        vae:
        rgb_cur:
        mask_cur:

    Returns:

    """
    # Encode the 4-channel (RGB + Mask) input using the fixed VAE model
    combined_input = torch.cat([rgb_cur, mask_cur], dim=1)  # Assuming (batch_size, 4, H, W)
    mu, _ = vae.encode(combined_input)
    return mu


def train_ldm(
    train_dataset: RiverDataset,
    ldm: Union[LatentDynamicsModel, LatentDynamicsModelMLP],
    vae: VAE,
    batch_size: int = 8,  # Used as sequence_length for LDM and batch_size for LDM-MLP
    epochs: int = 100,
    device: Union['str', torch.device] = 'cpu',
):
    """
    Train Latent Dynamics Model
    Args:
        train_dataset:
        ldm:
        vae:
        batch_size:
        epochs:
        device:

    Returns:

    """
    ldm.to(device)
    vae.to(device)
    vae.eval()  # Ensure the VAE model is in evaluation mode

    if isinstance(ldm, LatentDynamicsModel):
        # Training LDM with a sliding window
        total_length = len(train_dataset)

        for epoch in range(epochs):
            running_loss: float = 0.0

            # Sliding window across the entire trajectory
            # Note: by choosing step size to be batch_size, windows are not overlapping.
            #       But you can reduce it to make windows overlap, which just incurs more training time.
            for start_idx in range(0, total_length - batch_size + 1, batch_size):
                # Extract the windowed batch data
                window_data = [train_dataset[start_idx + i] for i in range(batch_size)]

                rgb_cur_batch = torch.stack([item[0] for item in window_data]).to(device)
                mask_cur_batch = torch.stack([item[1] for item in window_data]).to(device)
                act_batch = torch.stack([item[2] for item in window_data]).to(device)
                rgb_next_batch = torch.stack([item[3] for item in window_data]).to(device)
                mask_next_batch = torch.stack([item[4] for item in window_data]).to(device)

                # Encode current and next states using the VAE
                latent_cur = encode_input(vae, rgb_cur_batch, mask_cur_batch)  # Shape: (batch_size, latent_dim)
                latent_next = encode_input(vae, rgb_next_batch, mask_next_batch)  # Shape: (batch_size, latent_dim)

                h_prev = None  # Reset hidden state at the start of the sequence

                # Forward pass through the LDM for the entire sequence
                pred_latent, h_prev = ldm.forward(latent_cur, act_batch, h_prev)  # Outputs all steps
                pred_latent = pred_latent.squeeze(1)  # Remove the singleton batch dimension

                # Compute the loss over all steps
                loss = ldm.loss_mse(pred_latent, latent_next)

                # Backpropagation
                ldm.backprop(loss)

                running_loss += loss.item()
                if start_idx % 10 == 9:
                    print(
                        f'Epoch [{epoch + 1}/{epochs}], Step [{start_idx + 1}/{total_length}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

    elif isinstance(ldm, LatentDynamicsModelMLP):
        # Training LDM-MLP with shuffled batches
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss: float = 0.0

            for batch_idx, (rgb_cur, mask_cur, act, rgb_next, mask_next) in enumerate(train_loader):
                # Move data to device
                rgb_cur = rgb_cur.to(device)
                mask_cur = mask_cur.to(device)
                act = act.to(device)
                rgb_next = rgb_next.to(device)
                mask_next = mask_next.to(device)

                # Encode current and next states using the VAE
                latent_cur = encode_input(vae, rgb_cur, mask_cur)  # Shape: (batch_size, latent_dim)
                latent_next = encode_input(vae, rgb_next, mask_next)  # Shape: (batch_size, latent_dim)

                # Combine latent and action vectors and pass through the MLP
                latent_act = torch.cat((latent_cur, act), dim=-1)  # Shape: (batch_size, latent_dim + act_dim)
                pred_latent = ldm.forward(latent_act)

                # Compute the MSE loss
                loss = ldm.loss_mse(pred_latent, latent_next)

                # Backpropagation
                ldm.backprop(loss)

                running_loss += loss.item()
                if batch_idx % 10 == 9:
                    print(
                        f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

    # Save the final model
    if isinstance(ldm, LatentDynamicsModel):
        torch.save(ldm.state_dict(), 'ldm.pth')
    elif isinstance(ldm, LatentDynamicsModelMLP):
        torch.save(ldm.state_dict(), 'ldm_mlp.pth')
    else:
        raise NotImplementedError


def test_ldm(
    test_dataset: RiverDataset,
    ldm: Union[LatentDynamicsModel, LatentDynamicsModelMLP],
    vae: VAE,
    ldm_path: str,
    device: Union[str, torch.device] = 'cpu',
    horizon: int = 5,
    step: int = 1,
    metric: str = 'iou',
    output_csv: str = 'ldm_test_metrics.csv',
    debug: bool = False,
):
    """
    Test Latent Dynamics Model
    Args:
        test_dataset:
        ldm:
        vae:
        ldm_path:
        device:
        horizon:
        step:
        metric:
        output_csv:
        debug:

    Returns:

    """
    # Check path validity
    assert os.path.exists(ldm_path), f'LDM path {ldm_path} does not exist!'

    # Load the pre-trained model weights
    ldm.load_state_dict(torch.load(ldm_path))
    ldm.to(device)
    ldm.eval()
    vae.to(device)
    vae.eval()

    # Initialize variables for metric accumulation
    total_loss = 0.0
    num_samples = 0

    # Prepare the CSV file
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['start_index', 'step', 'metric_value'])

        # Iterate over the test data
        with torch.no_grad():
            for start_idx in range(0, len(test_dataset) - horizon, step):
                h_prev = None  # Reset the hidden state on prediction for LDM

                cur_rgb, cur_mask, act, next_rgb, next_mask = test_dataset[start_idx]

                # Move data to device
                cur_rgb = cur_rgb.to(device)
                cur_mask = cur_mask.to(device)
                act = act.to(device)
                next_rgb = next_rgb.to(device)
                next_mask = next_mask.to(device)

                # Encode the current state using the VAE
                cur_latent = encode_input(vae, cur_rgb.unsqueeze(0), cur_mask.unsqueeze(0)).squeeze(0)

                # Prepare for prediction over the specified horizon
                for t in range(horizon):
                    if isinstance(ldm, LatentDynamicsModelMLP):
                        latent_act = torch.cat((cur_latent, act), dim=-1)
                        pred_latent = ldm(latent_act)
                    elif isinstance(ldm, LatentDynamicsModel):
                        pred_latent, h_prev = ldm(cur_latent.unsqueeze(0), act.unsqueeze(0), h_prev)
                        pred_latent = pred_latent.squeeze(0)
                    else:
                        raise NotImplementedError

                    # Decode the predicted latent state back to the 4-channel RGB + mask
                    reconstructed = vae.decode(pred_latent.unsqueeze(0)).squeeze(0)  # Output shape: (4, H, W)

                    # Separate the reconstructed rgb and mask
                    pred_rgb = reconstructed[:-1]  # (3, H, W)
                    pred_mask = reconstructed[-1].unsqueeze(0)  # Mask is in the 4th channel. (1, H, W)

                    # Patchify the reconstructed predicted next mask and ground truth next mask
                    pred_mask_patch, true_mask_patch = get_patchified_mask(pred_mask), get_patchified_mask(next_mask)

                    if debug:
                        visualize_masks(true_mask_patch, pred_mask_patch)

                    # Evaluate based on the chosen metric
                    if metric == 'iou':
                        loss = iou_loss(pred_mask_patch, true_mask_patch).mean().item()
                    elif metric == 'l1':
                        loss = l1_loss(pred_mask_patch, true_mask_patch).item()
                    else:
                        raise ValueError(f"Unsupported metric {metric}. Choose 'iou' or 'l1'.")

                    # Write this step's metric to the CSV file
                    csv_writer.writerow([start_idx, t, loss])

                    total_loss += loss
                    num_samples += 1

                    # Encode by VAE using the predicted rgb and mask for the next iteration
                    cur_latent = encode_input(vae, pred_rgb.unsqueeze(0), pred_mask.unsqueeze(0)).squeeze(0)

                    # Read the true observation at the current step in prediction horizon
                    # Only action and the next true mask are needed for loss calculation
                    _, _, act, _, next_mask = test_dataset[start_idx + t + 1]

                    # Move necessary data to device
                    act.to(device)
                    next_mask.to(device)

    # Calculate the average metric over the entire test dataset
    average_metric = total_loss / num_samples
    print(f"Average {metric.upper()} over the test dataset: {average_metric:.4f}")
    return average_metric


def visualize_masks(true_mask, pred_mask):
    """
    Function to visualize the ground truth mask and the predicted mask.
    """
    true_mask_np = true_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()

    true_mask_np = inflate_patch_mask(true_mask_np)
    pred_mask_np = inflate_patch_mask(pred_mask_np)

    plt.figure(figsize=(10, 5))

    # Plot the ground truth mask
    plt.subplot(1, 2, 1)
    plt.imshow(true_mask_np, cmap='gray')
    plt.title(f'Ground Truth Mask')
    plt.axis('off')

    # Plot the predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title(f'Predicted Mask')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    train = False  # Set to True for training, False for testing
    use_mlp_model = True  # Set to True to use the MLP version of the LDM
    device = 'cpu'  # Change to 'cuda' if using a GPU
    # For VAE
    channel_size = 4  # 4-channel rgb+mask
    image_size = 128  # same for width and height
    latent_dim = 64
    hidden_dims = [16, 32, 64, 128]
    vae_model_name = 'vae-4channel.pth'
    lr = 0.0003
    # For recurrent LDM
    action_dim = 4
    gru_hidden_dim = 128
    gru_layers = 1

    # Load the fixed VAE model
    vae = VAE(
        in_channels=channel_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        target_output_size=2,
        original_height=image_size,
        original_width=image_size,
    )

    vae_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), vae_model_name)
    vae.load_state_dict(torch.load(vae_model_path))
    vae.eval()
    print(f'VAE is loaded.')

    model_config = ModelConfig(
        dynamics={
            'hidden_sizes': [64, 64],
            'activation': 'relu',
            'lr': lr,
        }
    )

    if use_mlp_model:
        ldm = LatentDynamicsModelMLP(latent_dim=latent_dim, act_dim=action_dim, model_cfgs=model_config)
    else:
        ldm = LatentDynamicsModel(latent_dim=latent_dim, act_dim=action_dim, gru_hidden_dim=gru_hidden_dim,
                                  num_gru_layers=gru_layers, lr=lr)

    # Initialize datasets
    train_dataset, test_dataset = get_train_test_datasets()

    if train:
        batch_size = 64
        train_epochs = 20
        train_ldm(train_dataset, ldm, vae, batch_size, train_epochs, device)
        print(f'Training completed.')
    else:
        debug = False  # Set to True if want to visualize LDM's mask prediction
        model_path = 'ldm_mlp.pth' if use_mlp_model else 'ldm.pth'
        horizon = 10
        metric = 'iou'  # Choose between {'iou', 'l1'}
        ldm_test_file = f'{model_path.split(".")[0]}_test_{metric}_h{horizon}.csv'

        test_ldm(test_dataset, ldm, vae, model_path, device, horizon, metric=metric, output_csv=ldm_test_file, debug=debug)
        print(f'Testing completed.')
