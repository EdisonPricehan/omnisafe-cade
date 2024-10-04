
from mlagents_envs.base_env import CameraPose
from mlagents_envs.envs.env_utils import make_unity_env
from mlagents_envs.side_channel.agent_reset_channel import AgentResetChannel
from mlagents_envs import logging_util

import omnisafe
from omnisafe.algorithms.algo_wrapper import AlgoWrapper as Agent
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace
from omnisafe.typing import DEVICE_CPU

from gymnasium.spaces import Box, MultiBinary

from typing import Any, Union, ClassVar, List, Tuple, Optional
import os
import numpy as np
import random
from patchify import patchify, unpatchify

import torch
from torch import nn

from encoder.vae import VAE
from encoder.dataset import InputChannelConfig

channel_config = InputChannelConfig.RGB_MASK  # Encode 4 channel (rgb+mask) observation
latent_dim = 16  # 16 is proven to perform best among (1024, 512, 256, 128, 64, 32, 16, 8)
hidden_dims = [32, 64, 128, 256, 512, 1024]  # channel sizes of CNN

# Pre-trained VAE model
vae_model_name = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-all-rgb_mask-16.pth'

# Path to the compiled Unity environment
# env_path = '/home/edison/Research/unity-saferl-envs/medium/riverine_medium_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/medium_dr/riverine_medium_dr_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/medium_dr_ext_reset/riverine_medium_dr_ext_reset_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/easy_dr/riverine_easy_dr_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/hard_dr/riverine_hard_dr_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/medium_dr_lvp/riverine_medium_dr_lvp_env.x86_64'
# env_path = None

# Observation image size in both x and y axes (hardcoded in Unity)
image_size = 128

seed = 0

# env_id = 'Medium'
# env_id = 'Easy'
# env_id = 'Hard'

logger = logging_util.get_logger(__name__)
logger.setLevel(logging_util.INFO)

channel_reset = AgentResetChannel()  # TODO get this from made unity gym env

worker_id: int = 0  # temp workaround for duplicate unity env creation (both in env_register and real training)


def load_vae_model() -> nn.Module:
    assert os.path.exists(vae_model_name), f'{vae_model_name} does not exist!'

    vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(vae_model_name, map_location=torch.device('cpu')))
    print(f'VAE model {vae_model_name} is loaded!')

    return vae_model


@env_register
class RiverineEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ['easy', 'medium', 'hard']

    need_action_scale_wrapper = False
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    _num_envs = 1
    _coordinate_observation_space: OmnisafeSpace
    _obs_len = 16  # VAE encoded vector
    _obs_pos_len = _obs_len + 4  # pose: (x, y, z, yaw)

    # Patchification params for riverine env
    _patch_size_x: int = 16  # pixels num of a patch in x axis
    _patch_size_y: int = 16  # pixels num of a patch in y axis
    _patch_step: int = 16  # step size when traversing the whole image to get patches, best to be the same as above

    def __init__(
        self,
        env_id: str = '',
        env_path: Optional[str] = None,
        use_vae: bool = False,
        water_perc_thr: float = 0.7,
        device: torch.device = DEVICE_CPU,
        max_idle_steps: int = 50,
        **kwargs,
    ) -> None:
        # Check env path validity
        if env_id == '':
            assert env_path is not None, f'Need either env_id or env_path to load Unity env.'
            assert os.path.exists(env_path), f'{env_path} does not exist!'
            env_id = env_path.split('/')[-1].split('.')[0].split('_')[1]
            print(f'{env_id=}')
        else:
            assert env_id in self._support_envs, f'Currently only support envs: {self._support_envs}'
            if env_path is not None:
                print(f'Since env_id is given, env_path will not be used.')
            env_path = f'/home/edison/Research/unity-saferl-envs/{env_id}_dr/riverine_{env_id}_dr_env.x86_64'
            assert os.path.exists(env_path), f'Unity env {env_path} does not exist!'

        super().__init__(env_id)
        self._device = torch.device(device)

        global worker_id

        self.env = make_unity_env(
            env_path=env_path,
            worker_id=worker_id,
            seed=seed,
            max_idle_steps=max_idle_steps,
        )
        worker_id += 1

        self.use_vae = use_vae
        self._vae_model = None

        # Load VAE model
        if use_vae:
            self._vae_model = load_vae_model()

        assert water_perc_thr >= 0, f'Water percentage threshold for patchification should be non-negative, given {water_perc_thr}'
        self.water_perc_thr = water_perc_thr

        # Set observation space
        self.raw_observation_space = self.env.observation_space
        if use_vae:
            high = np.array([2.0] * latent_dim)
            self._observation_space = Box(-high, high, dtype=np.float32)
        else:
            self.patch_dim_x, self.patch_dim_y = self.get_patch_dim()
            self._observation_space = MultiBinary(self.patch_dim_x * self.patch_dim_y)

        # Set action space
        self._action_space = self.env.action_space

        print(f'Riverine, obs space: {self._observation_space}')
        print(f'Riverine, action space: {self._action_space}')
        self._coordinate_observation_space = self._observation_space

        self.is_first_reset = True
        self.reset_pose = CameraPose()

        self.rgb: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.rgb_mask: Optional[np.ndarray] = None

    def get_cost_from_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.env.cur_cost

    def step(self, action: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # obs is a list of RGB image and water mask arrays
        obs, rew, cost, term, trunc, info = self.env.step(action.tolist())
        self.rgb, self.mask, self.rgb_mask = self.env.render()

        if self.use_vae:  # VAE encoding
            assert self._vae_model is not None
            obs = self.get_vae_embedding()
        else:  # Patchification
            obs = self.get_patchified_mask(obs)

        obs, rew, cost, term, trunc = (torch.as_tensor(x, dtype=torch.float32, device=self._device)
                                       for x in (obs, rew, cost, term, trunc))

        new_info = {'final_observation': obs}
        if hasattr(info['step'], 'done_reason'):
            # print(f'riverine env done reason: {info["step"].done_reason}')
            new_info['done_reason'] = info['step'].done_reason[0]
        return obs, rew, cost, term, trunc, new_info

    # Deprecated
    @staticmethod
    def round_clamp_action(float_action: torch.Tensor):
        return torch.clamp(torch.round(float_action), min=0, max=2).to(torch.int)

    def reset(self, seed: Union[int, None] = None, options: Union[dict[str, Any], None] = None)\
            -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)

        if not self.is_first_reset:
            # TODO
            # channel_reset.set_reset_pose(False, self.reset_pose.x, self.reset_pose.y, self.reset_pose.z,
            #                              self.reset_pose.yaw)
            channel_reset.set_reset_pose(True)  # use random reset for now
            print(f'Current reset pose: {self.reset_pose}')

        obs = self.env.reset()
        # print(f'{obs=}')
        self.rgb, self.mask, self.rgb_mask = self.env.render()

        if self.use_vae:
            assert self._vae_model is not None
            obs = self.get_vae_embedding()
        else:
            obs = self.get_patchified_mask(obs)

        # assert len(obs) == self._obs_pos_len, f'Reset obs length is {len(obs)}'
        # assert len(obs) == self._obs_len, f'Reset obs length is {len(obs)}'

        # if self.is_first_reset:
        #     pose = obs[self._obs_len:]
        #     print(f'First pose obs is {pose}')
        #     assert len(pose) == self._obs_pos_len - self._obs_len, f'Pose length is {len(pose)}'
        #     self.reset_pose.x = pose[0]
        #     self.reset_pose.y = pose[1]
        #     self.reset_pose.z = pose[2]
        #     self.reset_pose.yaw = pose[3]
        #     print(f'First reset pose: {self.reset_pose}')
        #     self.is_first_reset = False
        #
        # obs = obs[:self._obs_len]

        return torch.Tensor(obs), {}

    def get_vae_embedding(self) -> np.ndarray:
        """
        Encode 4-channel rgb+mask observation to a latent vector using pre-trained VAE model
        Returns:

        """
        _, _, rgb_mask = self.env.render()
        obs = torch.Tensor(rgb_mask).permute((2, 0, 1)).unsqueeze(0)  # N=1 x C=4 x H x W

        obs = self._vae_model.encode(obs)[0][0].detach().numpy()  # 1d vector with hidden_dim length

        return obs

    def get_patchified_mask(self, obs: List[np.ndarray]) -> np.ndarray:
        # TODO use function from math module
        """
        Patchify water mask observation then condensed to a flattened vector
        Args:
            obs:

        Returns:

        """
        assert len(obs) >= 2, f'obs len should be at least 2, given {len(obs)}.'

        # Get the 1-channel 2D water mask
        mask_arr = obs[1][..., 0]

        # Patchify this mask to shape (patch_row_num, patch_col_num, patch_size_x, patch_size_y)
        patchified_mask = patchify(mask_arr, (self._patch_size_x, self._patch_size_y), step=self._patch_step)

        # Calculate the percentage of 255 (water) pixels in each (patch_size_x, patch_size_y) patch
        percentage_255 = np.mean(patchified_mask == 255, axis=(2, 3))
        # print(f'{percentage_255.shape=}')

        # Create a 2D binary array where each element is 255 if the percentage exceeds 50%, otherwise 0
        condensed_mask = np.where(percentage_255 > self.water_perc_thr, 255, 0).flatten()
        # print(f'{condensed_mask=}')

        return condensed_mask

    def get_patch_dim(self) -> Tuple[int, int]:
        """
        Get the patch numbers on x and y axes based on the patchification params
        Returns:

        """
        patch_dim_x: int = ((image_size - self._patch_size_x) // self._patch_step) + 1
        patch_dim_y: int = ((image_size - self._patch_size_y) // self._patch_step) + 1
        return patch_dim_x, patch_dim_y

    def inflate_patch_mask(self, obs: np.ndarray) -> Optional[np.ndarray]:
        # TODO use function from math module
        """
        Inflate the coarsened water mask to the original size for parallel display
        Args:
            obs: coarsened water mask for RL training

        Returns:

        """
        if self.use_vae is True:
            print(f'Mask inflation is on supported for patchification method, not VAE encoding.')
            return None

        if len(obs.shape) == 1:
            obs_2d = obs.reshape((self.patch_dim_x, self.patch_dim_y))
        elif len(obs.shape) == 2:
            assert obs.shape[0] == self.patch_dim_x and obs.shape[1] == self.patch_dim_y, f'obs dim not match, {obs.shape}'
            obs_2d = obs.copy()
        else:
            raise NotImplementedError

        inflated_mask = np.zeros((self.patch_dim_x, self.patch_dim_y, self._patch_size_x, self._patch_size_y))
        for row in range(self.patch_dim_x):
            for col in range(self.patch_dim_y):
                inflated_mask[row, col] = np.full((self._patch_size_x, self._patch_size_y), obs_2d[row, col])

        inflated_mask_2d = unpatchify(inflated_mask, (image_size, image_size))

        return inflated_mask_2d

    def set_seed(self, seed: int) -> None:
        logger.warning('Setting env seed is not supported!')

    def sample_action(self) -> torch.Tensor:
        return torch.as_tensor(self._action_space.sample())

    def render(self) -> Any:
        return self.env.render()

    def render_available(self) -> bool:
        return self.rgb is not None and self.mask is not None and self.rgb_mask is not None

    def close(self) -> None:
        self.env.close()

    @property
    def coordinate_observation_space(self) -> OmnisafeSpace:
        return self._coordinate_observation_space


def evaluate(algo: str, env_id: str, seed_id: int, eval_time: int):
    LOG_DIR = f'./runs/{algo}-{{Medium}}/seed-{str(seed_id)}'
    global env_path, seed
    env_path = f'/home/edison/Research/unity-saferl-envs/{env_id.lower()}_dr/riverine_{env_id.lower()}_dr_env.x86_64'
    seed = seed_id

    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt' and '200' in item.name:
            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name='track',
                width=128,
                height=128,
            )
            result_path = f'./eval-V3/{algo}/{env_id}/seed{seed}'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            evaluator.render(num_episodes=eval_time, save_replay_path=result_path, max_render_steps=1000)
    scan_dir.close()


if __name__ == '__main__':
    # Play with Safe Riverine Environment with keyboard
    from mlagents_envs.key2action import Key2Action
    import matplotlib.pyplot as plt

    # Set the env name
    # env_id = 'medium'
    # env_id = 'easy'
    env_id = 'hard'

    env = RiverineEnv(
        env_id=env_id,
        use_vae=False,
        max_idle_steps=50000,
    )

    obs, _ = env.reset()

    k2a = Key2Action()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # Turn off the axes for each subplot
    for row in axes:
        for ax in row:
            ax.axis('off')
    # Interactive plot
    plt.ion()
    # Change the save key to Shift + S to avoid conflict
    plt.rcParams['keymap.save'] = ['shift+s']

    while not env.render_available():
        obs, _ = env.reset()

    rgb_canvas = axes[0, 0].imshow(env.rgb)
    mask_canvas = axes[1, 0].imshow(env.mask)
    mixed_canvas = axes[0, 1].imshow(env.rgb_mask)
    patchified_mask_canvas = axes[1, 1].imshow(env.inflate_patch_mask(obs.numpy()), cmap='gray')

    plt.tight_layout()

    try:
        i = 0
        while i < 10000:
            # get next action either manually or randomly
            action = k2a.get_multi_discrete_action()  # no action if no keyboard input

            obs, reward, cost, terminated, truncated, info = env.step(torch.Tensor(action))

            rgb, mask, mixed = env.render()

            rgb_canvas.set_data(rgb)
            mask_canvas.set_data(mask)
            mixed_canvas.set_data(mixed)
            patchified_mask_canvas.set_data(env.inflate_patch_mask(obs.numpy()))

            plt.tight_layout()
            fig.canvas.draw()  # Force canvas to draw
            fig.canvas.flush_events()  # Flush the GUI events for the figure
            plt.pause(0.001)

            if terminated or truncated:
                assert 'done_reason' in info, f'{info=}'
                done_reason = info['done_reason']
                print(f'{done_reason=}')
                env.reset()
    except KeyboardInterrupt:
        print(f'Interrupted by user.')
    finally:
        env.close()
        plt.close()









