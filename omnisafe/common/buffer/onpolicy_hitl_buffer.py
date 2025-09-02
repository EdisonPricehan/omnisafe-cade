import torch
import ast
import pandas as pd

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace
from omnisafe.utils.model import get_obs_dim, get_act_dim


class OnPolicyHitlBuffer(BaseBuffer):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = DEVICE_CPU,
    ):
        """
        The buffer for on-policy Human-In-The-Loop (HITL) algorithms.
        :param obs_space: Observation space.
        :param act_space: Action space.
        :param size: Maximum size of the buffer.
        :param device: Where the buffer is stored, by default CPU.
        """
        super().__init__(obs_space, act_space, size, device)

        self.data['reward_pred'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['cost_pred'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['logp'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['act_agent'] = torch.zeros((size, get_act_dim(act_space, execution_dim=True)),
                                             dtype=torch.float32, device=device)
        self.data['act_overlaid'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['next_obs'] = torch.zeros((size, get_obs_dim(obs_space)), dtype=torch.float32, device=device)
        self.data['next_obs_pred'] = torch.zeros((size, get_obs_dim(obs_space)), dtype=torch.float32, device=device)

        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size: int = size
        # Track per-episode lengths for cumulative datasets
        self.episode_lengths: list[int] = []

    def store(self, **data: torch.Tensor) -> None:
        """Store single step data into the buffer.

        .. warning::
            The total size of the data must be less than the buffer size.

        Args:
            data (torch.Tensor): The data to store.
        """
        assert self.ptr < self.max_size, 'No more space in the buffer!'  # TODO might consider cyclic buffer
        for key, value in data.items():
            self.data[key][self.ptr] = value
        self.ptr += 1

    def set(self, df: pd.DataFrame) -> None:
        """
        Initialize the buffer with data from a pandas DataFrame.

        Assumes that the column names in the DataFrame match the keys in self.data,
        and each value is either a scalar or a list (1D).

        Args:
            df (pd.DataFrame): DataFrame loaded from CSV containing buffer data.

        Returns:
            None
        """
        num_rows = len(df)
        assert num_rows <= self.max_size, f"Buffer overflow: trying to load {num_rows} rows into max size {self.max_size}"

        for key in self.data:
            assert key in df.columns, f"Missing key '{key}' in DataFrame"

            column_data = df[key]

            # Convert to tensor, handle list or scalar
            if isinstance(column_data.iloc[0], list):
                stacked = torch.tensor(column_data.tolist(), dtype=self.data[key].dtype)
            else:
                stacked = torch.tensor(column_data.values, dtype=self.data[key].dtype)

            self.data[key][:num_rows] = stacked

        self.ptr = num_rows  # update the buffer pointer

    def add(self, data: dict[str, torch.Tensor]) -> None:
        """Append a batch of data (multiple steps) into the buffer.

        Args:
            data: Dictionary of tensors with the same keys as self.data. First dimension must be the batch length.
        Raises:
            AssertionError: If remaining capacity is insufficient or keys mismatch / shape mismatch.
        """
        if not data:
            return
        # Infer length from one of the mandatory keys
        first_key = next(iter(data))
        batch_len = data[first_key].shape[0]
        assert self.ptr + batch_len <= self.max_size, (
            f'Insufficient capacity in buffer: have {self.remaining_capacity} slots, need {batch_len}. '
            f'Consider allocating a larger buffer.'
        )
        for k, v in data.items():
            assert k in self.data, f'Key {k} not found in buffer.'
            assert v.shape[0] == batch_len, f'Inconsistent batch length for key {k}: {v.shape[0]} vs {batch_len}.'
            # Broadcast / shape check except first dim
            expected_shape = self.data[k][self.ptr:self.ptr + batch_len].shape
            assert v.shape == expected_shape, (
                f'Shape mismatch for key {k}: expected {expected_shape}, got {v.shape}.')
            self.data[k][self.ptr:self.ptr + batch_len].copy_(v)
        self.ptr += batch_len

    def register_episode(self, length: int) -> None:
        """Register a newly added episode of given length (number of rows)."""
        if length > 0:
            self.episode_lengths.append(length)

    def last_episode_mask(self) -> torch.Tensor:
        """Return boolean mask over current buffer selecting ONLY last registered episode.
        If no episode registered, returns all True (whole buffer treated as one episode)."""
        total = self.ptr
        mask = torch.zeros((total,), dtype=torch.bool, device=self._device)
        if not self.episode_lengths or total == 0:
            mask[:total] = True
            return mask
        last_len = self.episode_lengths[-1]
        start = max(total - last_len, 0)
        mask[start:total] = True
        return mask

    def get(self, reset: bool = True, return_last_episode_mask: bool = False) -> dict[str, torch.Tensor]:
        """
        Retrieve all data from the buffer and optionally reset it.

        :param reset: Whether to clear the buffer after getting the data.
        :param return_last_episode_mask: Whether to return a mask for the last episode.
        :return: The dictionary containing all the data in the buffer.
        """
        data = {
            'obs': self.data['obs'][:self.ptr],
            'act': self.data['act'][:self.ptr],
            'act_agent': self.data['act_agent'][:self.ptr],
            'logp': self.data['logp'][:self.ptr],
            'act_overlaid': self.data['act_overlaid'][:self.ptr],
            'reward': self.data['reward'][:self.ptr],
            'reward_pred': self.data['reward_pred'][:self.ptr],
            'cost': self.data['cost'][:self.ptr],
            'cost_pred': self.data['cost_pred'][:self.ptr],
            'done': self.data['done'][:self.ptr],
            'next_obs': self.data['next_obs'][:self.ptr],
            'next_obs_pred': self.data['next_obs_pred'][:self.ptr],
            'episode_lengths': torch.tensor(self.episode_lengths, dtype=torch.int32, device=self._device),
        }

        if return_last_episode_mask:
            data['last_episode_mask'] = self.last_episode_mask()

        if reset:
            self.clear()

        return data

    @property
    def data_size(self) -> int:
        return self.ptr

    @property
    def remaining_capacity(self) -> int:
        return self.max_size - self.ptr

    def full(self) -> bool:
        return self.ptr >= self.max_size

    def clear(self) -> None:
        self.path_start_idx, self.ptr = 0, 0
        self.episode_lengths.clear()


def save_buffer_to_csv(data: dict[str, torch.Tensor], filename: str) -> None:
    """
    Save buffer's dict data to csv file.

    Args:
        data: The dictionary data to be saved.
        filename: Path to the output csv file.

    Returns:
        None
    """
    assert filename != '', f'Empty filename is not allowed!'

    df_dict = {}
    for k, v in data.items():
        arr = v.detach().cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] == 1:
            df_dict[k] = arr.squeeze().tolist()
        else:
            df_dict[k] = [arr[i].tolist() for i in range(arr.shape[0])]
    pd.DataFrame(df_dict).to_csv(filename, index=False)


def load_buffer_from_csv(filename: str, buffer: OnPolicyHitlBuffer) -> int:
    """
    Load csv data as dict then initialize the buffer with the data (in-place).

    Args:
        filename: Path to csv file.
        buffer: An initialized but empty (or to-be-overwritten) buffer.

    Returns:
        number_of_rows_loaded (int). Buffer is updated in-place.
    """
    df = pd.read_csv(filename)
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.startswith("[").all():
            df[col] = df[col].map(ast.literal_eval)

    buffer.set(df)
    # Reset episode tracking to a single episode of loaded length
    buffer.episode_lengths.clear()
    buffer.register_episode(len(df))

    return len(df)


def append_buffer_from_csv(filename: str, buffer: OnPolicyHitlBuffer) -> int:
    """Append data from a CSV file into an existing buffer without clearing existing data.

    Args:
        filename: Path to csv file.
        buffer: Existing buffer (may already contain data). Must have enough remaining capacity.

    Returns:
        Number of rows appended.
    """
    df = pd.read_csv(filename)
    if df.empty:
        return 0

    # Parse list-like string columns
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].astype(str).str.startswith('[').all():
            try:
                df[col] = df[col].map(ast.literal_eval)
            except Exception:
                pass

    num_rows = len(df)
    assert num_rows <= buffer.remaining_capacity, (
        f'Not enough capacity to append {num_rows} rows; remaining {buffer.remaining_capacity}.')

    # Build episode batch dict matching buffer.data shapes
    episode_data: dict[str, torch.Tensor] = {}
    for key in buffer.data.keys():
        assert key in df.columns, f"Missing column {key} in {filename}"
        col = df[key]
        if isinstance(col.iloc[0], list):
            t = torch.tensor(col.tolist(), dtype=buffer.data[key].dtype, device=buffer.data[key].device)
        else:
            t = torch.tensor(col.values, dtype=buffer.data[key].dtype, device=buffer.data[key].device)
        episode_data[key] = t

    buffer.add(episode_data)
    buffer.register_episode(num_rows)
    return num_rows


if __name__ == '__main__':
    # Test buffer load from csv file
    from gymnasium.spaces import MultiDiscrete, MultiBinary

    csv_filename: str = '../../../examples/evaluations/riverine/medium_hitlTrue_seed000_difficulty1_episode0.csv'
    obs_space: OmnisafeSpace = MultiBinary(256)
    act_space: OmnisafeSpace = MultiDiscrete([3, 3, 3, 3])
    buffer: OnPolicyHitlBuffer = OnPolicyHitlBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=1000,
    )

    rows = load_buffer_from_csv(filename=csv_filename, buffer=buffer)
    print(f'buffer.data_size={buffer.data_size}, rows_loaded={rows}')
    print(f'act first 5={buffer.data["act"][:5]}')
