import torch
from torch.utils.data import DataLoader

import os
import csv
from typing import Union
from gymnasium.spaces import MultiBinary, Discrete

from omnisafe.typing import OmnisafeSpace
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel
from omnisafe.models.dynamics.sdm_mlp import SemanticDynamicsModelMLP
from omnisafe.utils.cliff_dataset import CliffCircularDataset, get_train_test_datasets
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.math import soft_iou_loss, l1_loss, mse_loss


def train_sdm_cliffcircular(
    train_dataset: CliffCircularDataset,
    sdm: Union[SemanticDynamicsModel, SemanticDynamicsModelMLP],
    train_loss: Union['iou', 'l1', 'mse'] = 'iou',
    batch_size: int = 8,
    epochs: int = 200,
    save_model: bool = True,
):
    """
    Train SDM for CliffCircular environment.
    Args:
        train_dataset: CliffCircularDataset.
        sdm: SemanticDynamicsModel or MLP model.
        train_loss: Either 'iou' or 'l1'.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        save_model: Whether save the final model.
    """
    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Start training
    for epoch in range(epochs):
        running_loss: float = 0.

        for batch_idx, (obs_cur, act, obs_next) in enumerate(train_loader):
            # Concatenate current observation with action
            obs_act = torch.cat((obs_cur, act), dim=-1)

            if isinstance(sdm, SemanticDynamicsModel):
                delta = sdm.forward(obs_act)
                if train_loss == 'l1':
                    loss = sdm.loss_l1(obs_cur, act, delta, obs_next)
                elif train_loss == 'iou':
                    loss = sdm.loss_iou(obs_cur, act, delta, obs_next)
                elif train_loss == 'mse':
                    loss = sdm.loss_mse(obs_cur, act, delta, obs_next)
                else:
                    raise NotImplementedError

            elif isinstance(sdm, SemanticDynamicsModelMLP):
                pred_obs_next = sdm.forward(obs_act)
                if train_loss == 'l1':
                    loss = l1_loss(pred_obs_next, obs_next)
                elif train_loss == 'iou':
                    loss = soft_iou_loss(pred_obs_next, obs_next)
                elif train_loss == 'mse':
                    loss = mse_loss(pred_obs_next, obs_next)
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            sdm.backprop(loss)

            # Print statistics
            running_loss += loss.item()
            if batch_idx % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.

    # Save the final model
    if save_model:
        if isinstance(sdm, SemanticDynamicsModel):
            torch.save(sdm.state_dict(), f'sdm_{train_loss}_cliff.pth')
        elif isinstance(sdm, SemanticDynamicsModelMLP):
            torch.save(sdm.state_dict(), f'sdm_mlp_{train_loss}_cliff.pth')
        else:
            raise NotImplementedError


def sdm_cliffcircular_test(
    test_dataset: CliffCircularDataset,
    sdm: Union[SemanticDynamicsModel, SemanticDynamicsModelMLP],
    sdm_path: str,
    horizon: int = 5,
    step: int = 1,
    metric: str = 'iou',
    output_csv: str = 'sdm_test_metrics.csv',
):
    """
    Test SDM for CliffCircular environment.
    Args:
        test_dataset: CliffCircularDataset.
        sdm: Pre-trained SDM model.
        sdm_path: Path to the saved model.
        horizon: Number of prediction steps to test.
        step: Step size for iterating through the dataset.
        metric: Either 'iou' or 'l1'.
        output_csv: File to save test metrics.
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
                cur_obs, act, next_obs = test_dataset[start_idx]

                # Predict over the specified horizon
                for t in range(horizon):
                    obs_act = torch.cat((cur_obs, act), dim=-1)

                    if isinstance(sdm, SemanticDynamicsModelMLP):
                        pred_obs = sdm.predict(obs_act, round_to_int=True)
                    elif isinstance(sdm, SemanticDynamicsModel):
                        pred_obs = sdm.predict(obs_act.unsqueeze(0), round_to_int=True).squeeze(0)
                    else:
                        raise NotImplementedError

                    # Evaluate based on the chosen metric
                    if metric == 'iou':
                        loss = soft_iou_loss(pred_obs, next_obs).mean().item()
                    elif metric == 'l1':
                        loss = l1_loss(pred_obs, next_obs).item()
                    else:
                        raise ValueError(f"Unsupported metric {metric}. Choose 'iou' or 'l1'.")

                    # Get the next observation and action
                    cur_obs, act, next_obs = test_dataset[start_idx + t + 1]

                    # Write this step's metric to the CSV file
                    csv_writer.writerow([start_idx, t, loss])

                    total_loss += loss
                    num_samples += 1

    # Calculate the average metric over the entire test dataset
    average_metric = total_loss / num_samples
    print(f"Average {metric.upper()} loss over the test dataset: {average_metric:.4f}")
    return average_metric


if __name__ == '__main__':
    # Constants
    patch_rows, patch_cols = 5, 5  # Dim of agent's observation in CliffCircular env

    # Configurable parameters for both training and testing
    train = False
    save_model = True
    train_loss = 'mse'
    use_mlp_model = False  # Set True to use MLP model
    batch_size = 64
    train_epochs = 30
    horizon = 10
    test_metric = 'iou'
    device = 'cpu'  # Use 'cuda' if available
    model_config = ModelConfig(
        dynamics={
            'hidden_sizes': [64, 64],
            'activation': 'relu',
            # 'lr': 0.0003,
            'lr': 0.001,
        }
    )

    # Construct observation and action spaces
    obs_space: OmnisafeSpace = MultiBinary(patch_rows * patch_cols)
    act_space: OmnisafeSpace = Discrete(5)

    # Init SDM model
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
        # Start training
        train_sdm_cliffcircular(
            train_dataset=train_dataset,
            sdm=sdm,
            train_loss=train_loss,
            batch_size=batch_size,
            epochs=train_epochs,
            save_model=save_model,
        )
        print(f'Training finished.')

    else:
        model_path = f'sdm_mlp_{train_loss}_cliff.pth' if use_mlp_model else f'sdm_{train_loss}_cliff.pth'
        output_csv = f'sdm_mlp_test_{train_loss}_cliff_metric_{test_metric}.csv' \
            if use_mlp_model else f'sdm_test_{train_loss}_cliff_metric_{test_metric}.csv'

        # Start testing
        sdm_cliffcircular_test(
            test_dataset=test_dataset,
            sdm=sdm,
            sdm_path=model_path,
            horizon=horizon,
            metric=test_metric,
            output_csv=output_csv,
        )
        print(f'Testing finished.')
