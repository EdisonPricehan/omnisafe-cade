import os
import csv
from typing import Tuple

import torch
from torch.utils.data import Dataset


class CliffCircularDataset(Dataset):
    def __init__(self, data_dir, trajectories, transform=None):
        """
        Args:
            data_dir (str): Root directory containing all demo trajectories.
            trajectories (list): List of trajectory folder names to include (e.g., ['episode_0', 'episode_1']).
            transform (callable, optional): Optional transform to be applied on a sample (e.g., data augmentation).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        # Collect all data entries from the specified trajectory files
        for traj in trajectories:
            traj_file = os.path.join(data_dir, traj)

            # Read the CSV file using the built-in csv module
            with open(traj_file, newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                episode_data = list(csv_reader)  # Read all rows as a list of lists

            # Iterate over rows to collect current observation, action, and next observation
            for i in range(len(episode_data) - 1):  # Skip the last row, as it has no "next observation"
                current_obs = self._parse_array_string(episode_data[i][0])
                action = float(episode_data[i][1])
                next_obs = self._parse_array_string(episode_data[i + 1][0])

                # Append to dataset
                self.data.append({
                    'current_obs': current_obs,
                    'action': action,
                    'next_obs': next_obs
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        current_obs = sample['current_obs']
        action = sample['action']
        next_obs = sample['next_obs']

        if self.transform:
            current_obs = self.transform(current_obs)
            next_obs = self.transform(next_obs)

        # Convert to torch tensors
        current_obs = torch.tensor(current_obs, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        return current_obs, action, next_obs

    def _parse_array_string(self, array_str):
        """
        Parse a string that represents an array (e.g., '[1 1 0 0]') and convert it into a list of floats.
        """
        array_str = array_str.strip('[]')  # Remove the square brackets
        array_list = [float(item) for item in array_str.split()]  # Convert the space-separated values to floats
        return array_list


def get_train_test_datasets() -> Tuple[CliffCircularDataset, CliffCircularDataset]:
    # Dataset paths
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(cur_dir, '../datasets/cliff')

    # Define episodes for training and testing manually
    train_trajectories = [f'episode_{i}.csv' for i in range(80)]
    test_trajectories = [f'episode_{i}.csv' for i in range(80, 100)]

    # Construct the train and test dataset
    train_dataset = CliffCircularDataset(data_dir=dataset_dir, trajectories=train_trajectories)
    test_dataset = CliffCircularDataset(data_dir=dataset_dir, trajectories=test_trajectories)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = get_train_test_datasets()

    # Some example tests
    print(f'Datapoints num in train: {len(train_dataset)}')
    print(f'Datapoints num in test: {len(test_dataset)}')
    print(f'Datapoint data len: {len(train_dataset[0])}')
    print(f'{train_dataset[0][0].shape=}')
    print(f'{train_dataset[0][1].shape=}')
    print(f'{train_dataset[0][2].shape=}')






