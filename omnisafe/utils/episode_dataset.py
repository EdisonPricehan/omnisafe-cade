import torch
from torch.utils.data import Dataset

from typing import Tuple, Optional


class EpisodeDataset(Dataset):
    """
    A dataset class that splits data by episodes with padding for RL training with recurrent actor/critic
    """
    def __init__(
        self,
        tensors: Tuple[torch.Tensor, ...],
        done_tensor: torch.Tensor,
        padding: bool = False,
        max_seq_len: Optional[int] = None):
        self.padding = padding
        self.max_seq_len = max_seq_len
        if padding:
            assert max_seq_len is not None, f'padding enabled requires an integer max_seq_len argument.'

        self.padded_episodes = self.process_episodes(tensors, done_tensor)

    def process_episodes(self, tensors: Tuple[torch.Tensor, ...], done_tensor: torch.Tensor):
        all_padded_episodes = [[] for _ in tensors]

        # Iterate over each sequence (corresponding to interactions from one environment)
        start = 0
        for i, done in enumerate(done_tensor):
            if done:
                # Extract the episode from each tensor in tensors
                for j, tensor in enumerate(tensors):
                    episode = tensor[start:i + 1]
                    padded_episode = self.pad_episode(episode) if self.padding else episode
                    all_padded_episodes[j].append(padded_episode)
                start = i + 1

        # Handle any remaining episodes after the last "done"
        if start < len(done_tensor):
            for j, tensor in enumerate(tensors):
                episode = tensor[start:]
                padded_episode = self.pad_episode(episode) if self.padding else episode
                all_padded_episodes[j].append(padded_episode)

        if self.padding:
            # Stack the episodes for each tensor to form the final padded episodes
            return [torch.stack(episodes) for episodes in all_padded_episodes]
        else:
            # Episodes can be of various lengths
            return all_padded_episodes

    def pad_episode(self, episode):
        if len(episode) < self.max_seq_len:
            padding_size = self.max_seq_len - len(episode)
            if episode.dim() == 1:
                padding = torch.zeros(padding_size)
            elif episode.dim() == 2:
                padding = torch.zeros(padding_size, episode.size(-1))
            else:
                raise NotImplementedError
            episode = torch.cat([episode, padding], dim=0)
        return episode

    def __len__(self) -> int:
        return len(self.padded_episodes[0])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        """

        Args:
            idx: corresponds to the index of the episode in the dataset

        Returns:
            A tuple of all types of data in the idx episode
        """
        return tuple(tensor[idx] for tensor in self.padded_episodes)

