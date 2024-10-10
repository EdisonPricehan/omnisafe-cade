import csv
import torch

from omnisafe.utils.riverine_dataset import RiverDataset, get_train_test_datasets
from omnisafe.utils.patchification import get_patchified_mask
from omnisafe.utils.math import soft_iou_loss, l1_loss, iou


def baseline_test(
    test_dataset: RiverDataset,
    horizon: int = 5,
    step: int = 1,
    metric: str = 'iou',
    output_csv: str = 'baseline_test_metrics.csv',
):
    """
    Test the baseline model where the current true mask is assumed to be the next mask.

    Args:
        test_dataset: Dataset containing test trajectories.
        horizon: Number of prediction steps to evaluate.
        step: Step size for sliding window evaluation.
        metric: Metric to use ('iou' or 'l1').
        output_csv: CSV file to store the metric values.

    Returns:
        float: Average metric value over the test dataset.
    """
    # Initialize variables for metric accumulation
    total_loss = 0.0
    num_samples = 0

    # Prepare the CSV file for output
    with (open(output_csv, mode='w', newline='') as csv_file):
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['start_index', 'step', 'metric_value'])

        # Iterate over the test data
        with torch.no_grad():
            for start_idx in range(0, len(test_dataset) - horizon, step):
                cur_rgb, cur_mask, act, next_rgb, next_mask = test_dataset[start_idx]

                # Preprocess masks
                cur_mask = get_patchified_mask(cur_mask, patch_size_x=8, patch_size_y=8, patch_step=8)
                next_mask = get_patchified_mask(next_mask, patch_size_x=8, patch_size_y=8, patch_step=8)

                for t in range(horizon):
                    # Baseline assumption: current mask is treated as next mask
                    pred_mask = cur_mask

                    # Compute the baseline loss based on the selected metric
                    if metric == 'iou':
                        loss = iou(pred_mask, next_mask).mean().item()
                    elif metric == 'l1':
                        loss = l1_loss(pred_mask, next_mask).item()
                    else:
                        raise ValueError(f"Unsupported metric {metric}. Choose 'iou' or 'l1'.")

                    # Write this step's metric to the CSV file
                    csv_writer.writerow([start_idx, t, loss])

                    total_loss += loss
                    num_samples += 1

                    # Prepare for the next iteration
                    _, _, _, _, next_mask = test_dataset[start_idx + t + 1]
                    next_mask = get_patchified_mask(next_mask, patch_size_x=8, patch_size_y=8, patch_step=8)

    # Calculate the average metric over the entire test dataset
    average_metric = total_loss / num_samples
    print(f"Average {metric.upper()} loss for baseline over the test dataset: {average_metric:.4f}")
    return average_metric


if __name__ == '__main__':
    # Initialize dataset
    train_dataset, test_dataset = get_train_test_datasets()

    # Constants
    env: str = 'riverine'

    # Configurable parameters for the baseline
    horizon: int = 10
    metric: str = 'iou'  # Choose between {'iou', 'l1'}
    patch_size: int = 8
    baseline_test_file = f'{env}_baseline_test_{metric}_h{horizon}.csv'

    # Run the baseline test
    baseline_test(
        test_dataset=test_dataset,
        horizon=horizon,
        step=1,
        metric=metric,
        output_csv=baseline_test_file
    )
    print(f'Baseline test finished.')
