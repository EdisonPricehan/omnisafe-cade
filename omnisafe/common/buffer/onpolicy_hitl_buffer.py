import torch
import ast
import pandas as pd

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace
from omnisafe.utils.model import get_obs_dim, get_act_dim


class OnPolicyHITLBuffer(BaseBuffer):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = DEVICE_CPU,
    ):
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

    def get(self, reset: bool = True) -> dict[str, torch.Tensor]:
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
        }

        if reset:
            self.clear()

        return data

    @property
    def data_size(self) -> int:
        return self.ptr

    def full(self) -> bool:
        return self.max_size == self.ptr

    def clear(self) -> None:
        self.path_start_idx, self.ptr = 0, 0


def save_buffer_to_csv(data: dict[str, torch.Tensor], filename: str) -> None:
    """
    Save buffer's dict data to csv file
    Args:
        data:
        filename:

    Returns:

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


def load_buffer_from_csv(filename: str, buffer: OnPolicyHITLBuffer) -> OnPolicyHITLBuffer:
    """
    Load csv data as dict then initialize the buffer with the data
    Args:
        filename:
        buffer:

    Returns:

    """
    df = pd.read_csv(filename)
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.startswith("[").all():
            df[col] = df[col].map(ast.literal_eval)

    buffer.set(df)

    return buffer


if __name__ == '__main__':
    # Test buffer load from csv file
    from gymnasium.spaces import MultiDiscrete, MultiBinary

    csv_filename: str = '../../../examples/evaluations/riverine/medium_hitlTrue_seed000_difficulty1_episode0.csv'
    obs_space: OmnisafeSpace = MultiBinary(256)
    act_space: OmnisafeSpace = MultiDiscrete([3, 3, 3, 3])
    buffer: OnPolicyHITLBuffer = OnPolicyHITLBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=1000,
    )

    buffer = load_buffer_from_csv(filename=csv_filename, buffer=buffer)
    print(f'{buffer.data_size=}')
    print(f'{buffer.data["act"][:5]=}')





