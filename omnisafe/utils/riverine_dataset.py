import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple


class RiverDataset(Dataset):
    def __init__(self, data_dir, trajectories, transform=None):
        """
        Args:
            data_dir (str): Root directory containing all demo trajectories.
            trajectories (list): List of trajectory folder names to include (e.g., ['demo0', 'demo1']).
            transform (callable, optional): Optional transform to be applied on a sample (e.g., image augmentations).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        # Collect all data entries from the specified trajectories
        for traj in trajectories:
            traj_folder = os.path.join(data_dir, traj)
            traj_csv_path = os.path.join(traj_folder, "traj.csv")

            # Read the CSV file using the csv module
            with open(traj_csv_path, mode='r') as file:
                reader = csv.reader(file)
                rows = list(reader)

                # Iterate over rows to collect data pairs
                for i in range(len(rows) - 1):
                    current_rgb_path = os.path.join(traj_folder, rows[i][0])
                    current_mask_path = os.path.join(traj_folder, rows[i][1])
                    action = list(map(float, rows[i][2:6]))  # Convert action values to float
                    next_rgb_path = os.path.join(traj_folder, rows[i + 1][0])
                    next_mask_path = os.path.join(traj_folder, rows[i + 1][1])

                    self.data.append({
                        'current_rgb': current_rgb_path,
                        'current_mask': current_mask_path,
                        'action': action,
                        'next_rgb': next_rgb_path,
                        'next_mask': next_mask_path
                    })
        # TODO deal with the joint between 2 trajectories as they are not a natural transition

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load images
        current_rgb = Image.open(sample['current_rgb']).convert('RGB')
        current_mask = Image.open(sample['current_mask']).convert('L')  # Assuming masks are grayscale
        next_rgb = Image.open(sample['next_rgb']).convert('RGB')
        next_mask = Image.open(sample['next_mask']).convert('L')

        # Apply transformations to convert images to tensors
        if self.transform:
            current_rgb = self.transform(current_rgb)
            current_mask = self.transform(current_mask)
            next_rgb = self.transform(next_rgb)
            next_mask = self.transform(next_mask)

        # Convert action to a tensor
        action = torch.tensor(sample['action'], dtype=torch.float)  # Action is already a float array

        return current_rgb, current_mask, action, next_rgb, next_mask


def get_train_test_datasets() -> Tuple[RiverDataset, RiverDataset]:
    # Dataset paths
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(cur_dir, '../datasets/demos')

    # Transform to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # Converts to float tensor with range [0, 1], and channel first
    ])

    # Example usage
    train_trajectories = ['demo0', 'demo1', 'demo2', 'demo3', 'demo4', 'demo5', 'demo6', 'demo7', 'demo8', 'demo10']
    test_trajectories = ['demo9']

    # Construct the train and test dataset
    train_dataset = RiverDataset(data_dir=dataset_dir, trajectories=train_trajectories, transform=transform)
    test_dataset = RiverDataset(data_dir=dataset_dir, trajectories=test_trajectories, transform=transform)

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


