from mlagents_envs.base_env import CameraPose
from mlagents_envs.envs.unity_gym_env import DoneReason
from mlagents_envs.envs.env_utils import make_unity_env
from mlagents_envs.side_channel.agent_reset_channel import AgentResetChannel
from mlagents_envs import logging_util


# import omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace
from omnisafe.typing import DEVICE_CPU
from omnisafe.utils.patchification import inflate_patch_mask, get_patchified_mask

from gymnasium.spaces import Box, MultiBinary, Discrete, MultiDiscrete

from typing import Any, Union, ClassVar, List, Tuple, Optional
import os
import numpy as np
import random
from patchify import patchify, unpatchify

import torch
from torch import nn

# VAE-related
# from encoder.vae import VAE
# from encoder.dataset import InputChannelConfig
# channel_config = InputChannelConfig.RGB_MASK  # Encode 4 channel (rgb+mask) observation
latent_dim = 16  # 16 is proven to perform best among (1024, 512, 256, 128, 64, 32, 16, 8)
# hidden_dims = [32, 64, 128, 256, 512, 1024]  # channel sizes of CNN
# # Pre-trained VAE model
# vae_model_name = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-all-rgb_mask-16.pth'

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

logger = logging_util.get_logger(__name__)
logger.setLevel(logging_util.INFO)

channel_reset = AgentResetChannel()  # TODO get this from made unity gym env

worker_id: int = 0  # temp workaround for duplicate unity env creation (both in env_register and real training)


# Deprecated
def load_vae_model() -> nn.Module:
    """Deprecated. Used to load pre-trained VAE model for encoding observations."""
    assert os.path.exists(vae_model_name), f'{vae_model_name} does not exist!'

    vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(vae_model_name, map_location=torch.device('cpu')))
    print(f'VAE model {vae_model_name} is loaded!')

    return vae_model


def get_done_reason(reason_value: int) -> str:
    try:
        return DoneReason(reason_value).name
    except ValueError:
        return 'Unknown done reason.'


@env_register
class RiverineEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ['easy', 'medium', 'hard']

    need_action_scale_wrapper = False
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    _num_envs = 1
    _coordinate_observation_space: OmnisafeSpace

    # VAE param
    _obs_len = 16  # VAE encoded vector
    _obs_pos_len = _obs_len + 4  # pose: (x, y, z, yaw)

    # Patchification params for riverine env
    patch_size_x: int = 8  # pixels num of a patch in x axis
    patch_size_y: int = 8  # pixels num of a patch in y axis
    patch_step: int = 8  # step size when traversing the whole image to get patches, best to be the same as above

    def __init__(
        self,
        env_id: str = '',
        env_path: Optional[str] = None,
        use_vae: bool = False,
        water_perc_thr: float = 0.5,
        use_discrete_action: bool = False,
        block_backward_action: bool = True,
        device: Union[torch.device, str] = DEVICE_CPU,
        max_idle_steps: int = 50,
        **kwargs,
    ) -> None:
        """
        Initialize the Riverine environment.
        :param env_id: choose from 'easy', 'medium', 'hard' or leave empty to use env_path
        :param env_path: environment path to the compiled Unity environment.
        :param use_vae: whether to use VAE for observation encoding.
        :param water_perc_thr: the water percentage threshold for patchification.
        :param use_discrete_action: whether to use discrete action space instead of multi-discrete.
        :param block_backward_action: whether to block backward action in discrete action space.
        :param device: device to run the environment on, e.g., 'cpu' or 'cuda:0'.
        :param max_idle_steps: maximum number of idle steps before the environment is reset.
        :param kwargs:
        """
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

        # Init Unity environment with the unique worker_id
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
            print(f'Patch dim x: {self.patch_dim_x}, patch dim y: {self.patch_dim_y}')
            self._observation_space = MultiBinary(self.patch_dim_x * self.patch_dim_y)

        # Set action space
        self.use_discrete_action = use_discrete_action
        self.block_backward_action = block_backward_action
        if use_discrete_action:
            self._action_space = Discrete(8) if block_backward_action else Discrete(9)
        else:
            self._action_space = MultiDiscrete([3, 3, 2, 3]) if block_backward_action else self.env.action_space

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

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Convert discrete action to multi-discrete action if needed
        if self.use_discrete_action:
            assert action.squeeze().dim() == 0, f'Action should be a scalar for discrete action space, got {action.shape}'
            action = torch.tensor(discrete_to_multi_discrete_action(action.cpu().item()))
            # if not np.all(np.array(action) == 1):
            #     print(f'Converted discrete action to multi-discrete: {action}')

        # obs is a list of RGB image and water mask arrays
        obs, rew, cost, term, trunc, info = self.env.step(action.tolist())
        self.rgb, self.mask, self.rgb_mask = self.env.render()

        if self.use_vae:  # VAE encoding
            assert self._vae_model is not None
            obs_vec = self.get_vae_embedding()
        else:  # Patchification
            obs_vec = get_patchified_mask(
                mask=obs[1],
                is_uint8=True,
                patch_size_x=self.patch_size_x,
                patch_size_y=self.patch_size_y,
                patch_step=self.patch_step,
                binary_threshold=0.5,
                patch_threshold=self.water_perc_thr,
            )

            # Update immediate cost
            # if not term:
            #     obs_mask = np.reshape(obs_vec, (self.patch_dim_x, self.patch_dim_y))
            #     cost = self.get_water_iou_cost(obs_mask)

        # Upscale reward so that per-step max reward is 1
        # if abs(rew) > 1e-6:
        #     rew = 1

        obs_vec, rew, cost, term, trunc = (torch.as_tensor(x, dtype=torch.float32, device=self._device)
                                       for x in (obs_vec, rew, cost, term, trunc))

        new_info = {'final_observation': obs_vec}
        if hasattr(info['step'], 'done_reason'):
            # print(f'riverine env done reason: {info["step"].done_reason}')
            new_info['done_reason'] = info['step'].done_reason[0]
        return obs_vec, rew, cost, term, trunc, new_info

    # Deprecated
    @staticmethod
    def round_clamp_action(float_action: torch.Tensor):
        return torch.clamp(torch.round(float_action), min=0, max=2).to(torch.int)

    def reset(self, seed: Union[int, None] = None, options: Union[dict[str, Any], None] = None) \
        -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)

        if not self.is_first_reset:
            # TODO manual reset is not available for now
            # channel_reset.set_reset_pose(False, self.reset_pose.x, self.reset_pose.y, self.reset_pose.z,
            #                              self.reset_pose.yaw)
            channel_reset.set_reset_pose(random_reset=True)  # use random reset for now
            print(f'Current reset pose: {self.reset_pose}')

        obs = self.env.reset()
        self.rgb, self.mask, self.rgb_mask = self.env.render()

        if self.use_vae:
            assert self._vae_model is not None
            obs = self.get_vae_embedding()
        else:
            obs = get_patchified_mask(
                mask=obs[1],
                is_uint8=True,
                patch_size_x=self.patch_size_x,
                patch_size_y=self.patch_size_y,
                patch_step=self.patch_step,
                binary_threshold=0.5,
                patch_threshold=self.water_perc_thr,
            )

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
        Encode 4-channel rgb+mask observation to a latent vector using pre-trained VAE model.

        Returns:
            The encoded observation as a 1D numpy array with length equal to latent_dim.
        """
        _, _, rgb_mask = self.env.render()

        obs = torch.Tensor(rgb_mask).permute((2, 0, 1)).unsqueeze(0)  # N=1 x C=4 x H x W

        obs = self._vae_model.encode(obs)[0][0].detach().numpy()  # 1d vector with hidden_dim length

        return obs

    def get_patch_dim(self) -> Tuple[int, int]:
        """
        Get the patch numbers on x and y axes based on the patchification params.

        Returns:
            A tuple containing the number of patches in x and y axes.
        """
        patch_dim_x: int = ((image_size - self.patch_size_x) // self.patch_step) + 1
        patch_dim_y: int = ((image_size - self.patch_size_y) // self.patch_step) + 1
        return patch_dim_x, patch_dim_y

    def get_water_iou_cost(self, mask_obs: np.ndarray) -> float:
        """
        Calculate the IoU-based cost for the water mask observation against a predefined trapezoidal mask.

        Args:
            mask_obs (np.ndarray): Observation mask (H x W) as 2D array.

        Returns:
            float: IoU-based cost, calculated as (1 - IoU) / 20.
        """
        assert mask_obs is not None, f'mask obs is None'
        assert len(mask_obs.shape) == 2, f'mask obs has wrong dimension {mask_obs.shape}'

        mask_trapezoid = self.create_trapezoidal_mask(
            top_width=4,
            down_width=10,
            trapezoid_height=14,
        )  # TODO these values can be percentages

        # print(f'mask obs:')
        # print(f'{mask_obs}')
        #
        # print(f'mask trap:')
        # print(f'{mask_trapezoid}')

        # Compute intersection and union directly
        intersection = np.logical_and(mask_obs == 1, mask_trapezoid == 255).sum()
        union = np.logical_or(mask_obs == 1, mask_trapezoid == 255).sum()

        # Avoid division by zero
        iou = intersection / (union + 1e-6)

        # Calculate IoU-based cost
        iou_cost = 1.0 - iou
        return iou_cost / 20  # Downscale immediate cost

    def create_trapezoidal_mask(
        self,
        top_width: int,
        down_width: int,
        trapezoid_height: int,
    ) -> np.ndarray:
        """
        Create a trapezoidal mask for a square observation, with the trapezoid's bottom side aligned to the bottom of the observation.

        Args:
            top_width (int): Width of the trapezoid at the top.
            down_width (int): Width of the trapezoid at the bottom.
            trapezoid_height (int): Height of the trapezoid.

        Returns:
            np.ndarray: A binary mask (2D array) with the trapezoidal region filled with 1s.
        """
        assert top_width < self.patch_dim_x, f'{top_width} should be less than obs side len {self.patch_dim_x}'
        assert down_width < self.patch_dim_x, f'{down_width} should be less than obs side len {self.patch_dim_x}'
        assert trapezoid_height < self.patch_dim_x, f'{trapezoid_height} should be less than obs side len {self.patch_dim_x}'
        assert top_width <= down_width, f'{top_width} should not be greater than {down_width}'

        # Initialize the mask with zeros
        mask = np.zeros((self.patch_dim_x, self.patch_dim_x), dtype=np.uint8)

        # Calculate the vertical positions for the trapezoid
        bottom_y = self.patch_dim_x  # Bottom edge of the observation
        top_y = bottom_y - trapezoid_height  # Top edge of the trapezoid

        # Calculate the horizontal positions for the top and bottom edges of the trapezoid
        top_left = (self.patch_dim_x - top_width) // 2
        top_right = top_left + top_width
        bottom_left = (self.patch_dim_x - down_width) // 2
        bottom_right = bottom_left + down_width

        # Fill in the trapezoidal area
        for y in range(top_y, bottom_y):
            # Interpolate the width of the trapezoid at the current height
            alpha = (y - top_y) / trapezoid_height
            current_left = int((1 - alpha) * top_left + alpha * bottom_left)
            current_right = int((1 - alpha) * top_right + alpha * bottom_right + 1)
            current_right = min(current_right, self.patch_dim_x)

            # Fill the row in the trapezoidal range
            mask[y, current_left:current_right] = 255

        return mask

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


def discrete_to_multi_discrete_action(action: int) -> List[int]:
    """
    Convert a discrete action to a multi-discrete action.

    Args:
        action (int): Discrete action value in range [0, 8], where 0 means no movement.

    Returns:
        List[int]: Multi-discrete action as a list of 4 integers, at 4 distinct axes.
    """
    assert 0 <= action <= 8, f'Action {action} is out of range [0, 8]'

    multi_discrete_action = [1, 1, 1, 1]  # Default action (no movement)

    if action == 0:  # No movement
        return multi_discrete_action

    axis_idx = (action - 1) // 2
    direction = 0 if action % 2 == 1 else 2  # 0 for positive direction, 2 for negative direction
    multi_discrete_action[int(axis_idx)] = int(direction)

    return multi_discrete_action


if __name__ == '__main__':
    # Play with Safe Riverine Environment with keyboard
    from mlagents_envs.key2action import Key2Action
    import matplotlib.pyplot as plt

    # Set the env difficulty level
    env_id = 'medium'
    # env_id = 'easy'
    # env_id = 'hard'

    # Init the env
    env = RiverineEnv(
        env_id=env_id,
        use_vae=False,
        water_perc_thr=0.5,
        use_discrete_action=True,  # Test discrete action space
        device='cpu',
        max_idle_steps=50000,
    )

    # Init keyboard control
    k2a = Key2Action()

    # Init the figure canvas
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Turn off the axes for each subplot
    for row in axes:
        for ax in row:
            ax.axis('off')

    # Interactive plot
    plt.ion()

    # Change the save key to Shift + S to avoid conflict
    plt.rcParams['keymap.save'] = ['shift+s']

    # Make sure env is ready
    obs, _ = env.reset()
    while not env.render_available():
        print(f'Reset again until rendered observations are available.')
        obs, _ = env.reset()
    print(f'Rendering is available!')

    rgb_canvas = axes[0, 0].imshow(env.rgb)
    mask_canvas = axes[1, 0].imshow(env.mask)
    mixed_canvas = axes[0, 1].imshow(env.rgb_mask)
    inflated_patch_mask = inflate_patch_mask(
        obs=obs.numpy(),
        image_size=image_size,
        patch_dim_x=env.patch_dim_x,
        patch_dim_y=env.patch_dim_y,
        patch_size_x=env.patch_size_x,
        patch_size_y=env.patch_size_y,
    )
    patchified_mask_canvas = axes[1, 1].imshow(inflated_patch_mask, cmap='gray')

    plt.tight_layout()

    try:
        i = 0
        while i < 10000:
            # get next action manually
            if env.use_discrete_action:
                action = k2a.get_discrete_action()
            else:
                action = k2a.get_multi_discrete_action()  # default action if no keyboard input

            obs, reward, cost, terminated, truncated, info = env.step(torch.Tensor(action))

            # Print meaningful action, reward, and cost
            if env.use_discrete_action:
                if action[0] != 0:
                    print(f'Action: {action[0]}, reward: {reward:.2f}, cost: {cost:.2f}')
            else:
                if not np.all(np.array(action) == 1):
                    print(f'Action: {action}, reward: {reward:.2f}, cost: {cost:.2f}')

            rgb, mask, mixed = env.render()

            rgb_canvas.set_data(rgb)
            mask_canvas.set_data(mask)
            mixed_canvas.set_data(mixed)
            patchified_mask_canvas.set_data(
                inflate_patch_mask(
                    obs=obs.numpy(),
                    image_size=image_size,
                    patch_dim_x=env.patch_dim_x,
                    patch_dim_y=env.patch_dim_y,
                    patch_size_x=env.patch_size_x,
                    patch_size_y=env.patch_size_y,
                ))

            plt.tight_layout()
            fig.canvas.draw()  # Force canvas to draw
            fig.canvas.flush_events()  # Flush the GUI events for the figure
            plt.pause(0.001)

            if terminated or truncated:
                assert 'done_reason' in info, f'{info=}'
                done_reason = info['done_reason']
                print(f'Done reason: {get_done_reason(done_reason)}')
                env.reset()
    except KeyboardInterrupt:
        print(f'Interrupted by user.')
    finally:
        env.close()
        plt.close()
