
import torch
from torch.utils.data import DataLoader

import os
import csv
from typing import Union
from gymnasium.spaces import MultiBinary, MultiDiscrete

from omnisafe.typing import OmnisafeSpace
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel
from omnisafe.models.dynamics.sdm_mlp import SemanticDynamicsModelMLP
from omnisafe.utils.riverine_dataset import RiverDataset, get_train_test_datasets
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.patchification import get_patchified_mask
from omnisafe.utils.math import iou_loss, l1_loss


def train_sdm(
    train_dataset: RiverDataset,
    sdm: Union[SemanticDynamicsModel, SemanticDynamicsModelMLP],
    batch_size: int = 8,
    epochs: int = 200,
):
    """
    Train Semantic Dynamics Model
    Args:
        train_dataset:
        sdm:
        batch_size:
        epochs:

    Returns:

    """
    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Start training
    for epoch in range(epochs):
        running_loss: float = 0.

        for batch_idx, (rgb_cur, mask_cur, act, rgb_next, mask_next) in enumerate(train_loader):
            mask_cur = get_patchified_mask(mask_cur)
            mask_next = get_patchified_mask(mask_next)
            mask_act = torch.cat((mask_cur, act), dim=-1)

            if isinstance(sdm, SemanticDynamicsModel):
                delta = sdm.forward(mask_act)
                loss = sdm.loss_l1(mask_cur, act, delta, mask_next)
            elif isinstance(sdm, SemanticDynamicsModelMLP):
                pred_mask_next = sdm.forward(mask_act)
                loss = sdm.loss_l1(pred_mask_next, mask_next)
            else:
                raise NotImplementedError
            sdm.backprop(loss)

            # Print statistics
            running_loss += loss.item()
            if batch_idx % 10 == 9:  # Print every 10 batches
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.

    # Save the final model
    if isinstance(sdm, SemanticDynamicsModel):
        torch.save(sdm.state_dict(), 'sdm.pth')
    elif isinstance(sdm, SemanticDynamicsModelMLP):
        torch.save(sdm.state_dict(), 'sdm_mlp.pth')
    else:
        raise NotImplementedError


def test_sdm(
    test_dataset: RiverDataset,
    sdm: Union[SemanticDynamicsModel, SemanticDynamicsModelMLP],
    sdm_path: str,
    horizon: int = 5,
    step: int = 1,
    metric: str = 'iou',
    output_csv: str = 'sdm_test_metrics.csv',
):
    """
    Test Semantic Dynamics Model
    Args:
        test_dataset:
        sdm:
        sdm_path:
        horizon:
        step:
        metric:
        output_csv:

    Returns:

    """
    # Check path validity
    assert os.path.exists(sdm_path), f'SDM path {sdm_path} does not exist!'

    # Load the pre-trained model weights
    sdm.load_state_dict(torch.load(sdm_path))
    sdm.eval()

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
                cur_rgb, cur_mask, act, next_rgb, next_mask = test_dataset[start_idx]

                # Preprocess masks
                cur_mask, next_mask = get_patchified_mask(cur_mask), get_patchified_mask(next_mask)
                cur_mask_act = torch.cat((cur_mask, act), dim=-1)

                # Predict over the specified horizon
                # Predicted mask will be rounded to int (basically thresholded by 0.5)
                for t in range(horizon):
                    if isinstance(sdm, SemanticDynamicsModelMLP):
                        pred_mask = sdm.predict(cur_mask_act, round_to_int=True)
                    elif isinstance(sdm, SemanticDynamicsModel):
                        pred_mask = sdm.predict(cur_mask_act.unsqueeze(0), round_to_int=True).squeeze(0)

                    # Evaluate based on the chosen metric
                    if metric == 'iou':
                        loss = iou_loss(pred_mask, next_mask).mean().item()
                    elif metric == 'l1':
                        loss = l1_loss(pred_mask, next_mask).item()
                    else:
                        raise ValueError(f"Unsupported metric {metric}. Choose 'iou' or 'l1'.")

                    # Get the necessary data sample containing the true next mask
                    _, _, act, _, next_mask = test_dataset[start_idx + t + 1]

                    # Patchify the ground truth mask for loss calculation
                    next_mask = get_patchified_mask(next_mask)

                    # Prepare the mask_act input for the next iteration
                    cur_mask_act = torch.cat((pred_mask, act), dim=-1)

                    # Write this step's metric to the CSV file
                    csv_writer.writerow([start_idx, t, loss])

                    total_loss += loss
                    num_samples += 1

    # Calculate the average metric over the entire test dataset
    average_metric = total_loss / num_samples
    print(f"Average {metric.upper()} over the test dataset: {average_metric:.4f}")
    return average_metric


if __name__ == '__main__':
    # Constants
    patch_rows, patch_cols = 8, 8
    obs_space: OmnisafeSpace = MultiBinary(patch_rows * patch_cols)
    act_space: OmnisafeSpace = MultiDiscrete([3, 3, 3, 3])

    # Configurable parameters for both training and testing
    train = False
    use_mlp_model = False
    device = 'cpu'  # TODO, not used yet
    model_config = ModelConfig(
        dynamics={
            'hidden_sizes': [64, 64],
            'activation': 'relu',
            'lr': 0.0003,
        }
    )

    # Init SDM
    if use_mlp_model:
        sdm = SemanticDynamicsModelMLP(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_config,
        )
    else:
        sdm = SemanticDynamicsModel(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_config,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
        )

    # Init dataset
    train_dataset, test_dataset = get_train_test_datasets()

    if train:
        # Configurable params
        batch_size = 64
        train_epochs = 20

        # Start training
        train_sdm(train_dataset, sdm, batch_size, train_epochs)
        print(f'Train finished.')
    else:  # test
        # Configurable params
        model_path: str = 'sdm_mlp.pth' if use_mlp_model else 'sdm.pth'
        horizon: int = 10
        metric: str = 'l1'  # Choose between {'iou', 'l1'}
        sdm_test_file = f'{model_path.split(".")[0]}_test_{metric}_h{horizon}.csv'

        # Start testing
        test_sdm(
            test_dataset=test_dataset,
            sdm=sdm,
            sdm_path=model_path,
            horizon=horizon,
            step=1,
            metric=metric,
            output_csv=sdm_test_file
        )
        print(f'Test finished.')





