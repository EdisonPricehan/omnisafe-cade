
import torch
from torch.utils.data import DataLoader

import os
import csv
from typing import Union, Literal
from gymnasium.spaces import MultiBinary, MultiDiscrete

from omnisafe.typing import OmnisafeSpace
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel
from omnisafe.models.dynamics.sdm_mlp import SemanticDynamicsModelMLP
from omnisafe.utils.riverine_dataset import RiverDataset, get_train_test_datasets
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.patchification import get_patchified_mask
from omnisafe.utils.math import soft_iou_loss, l1_loss, iou


def train_sdm(
    train_dataset: RiverDataset,
    sdm: Union[SemanticDynamicsModel, SemanticDynamicsModelMLP],
    train_loss: Union['iou', 'l1', 'mse'] = 'iou',
    patch_size_x: int = 8,
    patch_size_y: int = 8,
    patch_step: int = 8,
    batch_size: int = 8,
    epochs: int = 200,
    model_save_path: str = '',
):
    """
    Train Semantic Dynamics Model
    Args:
        train_dataset:
        sdm:
        patch_size_x:
        patch_size_y:
        patch_step:
        batch_size:
        epochs:
        model_save_path:

    Returns:

    """
    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Start training
    for epoch in range(epochs):
        running_loss: float = 0.

        for batch_idx, (rgb_cur, mask_cur, act, rgb_next, mask_next) in enumerate(train_loader):
            mask_cur = get_patchified_mask(mask_cur, patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_step=patch_step)
            mask_next = get_patchified_mask(mask_next, patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_step=patch_step)
            mask_act = torch.cat((mask_cur, act), dim=-1)

            if isinstance(sdm, SemanticDynamicsModel):
                delta = sdm.forward(mask_act)
                # print(f'{delta=}')
                # print(f'{delta.min()=} {delta.max()=} {delta.abs().mean()=}')
                # exit(0)
                if train_loss == 'l1':
                    loss = sdm.loss_l1(mask_cur, act, delta, mask_next)
                else:
                    loss = sdm.loss_iou(mask_cur, act, delta, mask_next)
                # print(f'{loss=}')
            elif isinstance(sdm, SemanticDynamicsModelMLP):
                pred_mask_next = sdm.forward(mask_act)
                if train_loss == 'l1':
                    loss = l1_loss(pred_mask_next, mask_next)
                else:
                    loss = soft_iou_loss(pred_mask_next, mask_next)
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
    if model_save_path != '':
        if isinstance(sdm, SemanticDynamicsModel):
            torch.save(sdm.state_dict(), model_save_path)
            print(f'Model saved as {model_save_path}.')
        elif isinstance(sdm, SemanticDynamicsModelMLP):
            torch.save(sdm.state_dict(), model_save_path)
            print(f'Model saved as {model_save_path}.')
        else:
            raise NotImplementedError


def sdm_testing(
    test_dataset: RiverDataset,
    sdm: Union[SemanticDynamicsModel, SemanticDynamicsModelMLP],
    sdm_path: str,
    patch_size_x: int = 8,
    patch_size_y: int = 8,
    patch_step: int = 8,
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
        patch_size_x:
        patch_size_y:
        patch_step:
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
                cur_mask = get_patchified_mask(cur_mask, patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_step=patch_step)
                next_mask = get_patchified_mask(next_mask, patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_step=patch_step)
                cur_mask_act = torch.cat((cur_mask, act), dim=-1)

                # Predict over the specified horizon
                # Predicted mask will be rounded to int (basically thresholded by 0.5)
                for t in range(horizon):
                    if isinstance(sdm, SemanticDynamicsModelMLP):
                        pred_mask = sdm.predict(cur_mask_act, round_to_int=True)
                    elif isinstance(sdm, SemanticDynamicsModel):
                        pred_mask = sdm.predict(cur_mask_act.unsqueeze(0), round_to_int=True).squeeze(0)

                    # Evaluate based on the chosen metric
                    if metric == 'iou':  # Use the binarized mask to calculate iou (not a loss)
                        loss = iou(pred_mask, next_mask).mean().item()
                    elif metric == 'l1':
                        loss = l1_loss(pred_mask, next_mask).item()
                    else:
                        raise ValueError(f"Unsupported metric {metric}. Choose 'iou' or 'l1'.")

                    # Get the necessary data sample containing the true next mask
                    _, _, act, _, next_mask = test_dataset[start_idx + t + 1]

                    # Patchify the ground truth mask for loss calculation
                    next_mask = get_patchified_mask(next_mask, patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_step=patch_step)

                    # Prepare the mask_act input for the next iteration
                    cur_mask_act = torch.cat((pred_mask, act), dim=-1)

                    # Write this step's metric to the CSV file
                    csv_writer.writerow([start_idx, t, loss])

                    total_loss += loss
                    num_samples += 1

    # Calculate the average metric over the entire test dataset
    average_metric = total_loss / num_samples
    print(f"Average {metric.upper()} (loss) over the test dataset: {average_metric:.4f}")
    return average_metric


if __name__ == '__main__':
    # Constants
    env: str = 'riverine'
    image_size: int = 128  # dimension of raw image (assuming square)
    patch_rows, patch_cols = 16, 16  # dimension of coarsened mask
    patch_size_x, patch_size_y = image_size // patch_rows, image_size // patch_cols
    patch_step = patch_size_x  # no overlap among patches

    # Construct observation and action spaces
    obs_space: OmnisafeSpace = MultiBinary(patch_rows * patch_cols)
    act_space: OmnisafeSpace = MultiDiscrete([3, 3, 3, 3])

    # Configurable parameters for both training and testing
    train = False
    train_loss = 'iou'
    use_mlp_model = True
    model_path = f'{env}_sdm_mlp_{train_loss}.pth' if use_mlp_model else f'{env}_sdm_{train_loss}.pth'
    device = 'cpu'  # TODO, not used yet
    model_config = ModelConfig(
        dynamics={
            'hidden_sizes': [128, 64],
            'activation': 'relu',
            'lr': 0.001,
        }
    )

    # Init SDM or SDM-MLP
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

    # Init datasets for training and testing
    train_dataset, test_dataset = get_train_test_datasets()

    if train:
        # Configurable params
        batch_size = 64
        train_epochs = 30

        # Start training
        train_sdm(
            train_dataset=train_dataset,
            sdm=sdm,
            train_loss=train_loss,
            patch_size_x=patch_size_x,
            patch_size_y=patch_size_y,
            patch_step=patch_step,
            batch_size=batch_size,
            epochs=train_epochs,
            model_save_path=model_path,
        )
        print(f'Train finished.')

    else:  # test
        # Configurable params
        horizon: int = 10  # Prediction horizon
        metric: str = 'l1'  # Choose between {'iou', 'l1'}
        sdm_test_file = f'{model_path.split(".")[0]}_test_{metric}_h{horizon}.csv'

        # Start testing
        sdm_testing(
            test_dataset=test_dataset,
            sdm=sdm,
            sdm_path=model_path,
            patch_size_x=patch_size_x,
            patch_size_y=patch_size_y,
            patch_step=patch_step,
            horizon=horizon,
            step=1,
            metric=metric,
            output_csv=sdm_test_file
        )
        print(f'Test finished, results saved to {sdm_test_file}.')





