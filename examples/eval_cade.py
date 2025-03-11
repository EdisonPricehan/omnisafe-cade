import os
import json
import csv
import pandas as pd
import numpy as np
import torch
import time
from pynput import keyboard
from torch.distributions import Categorical
from typing import Optional, List, Tuple, Dict, Any
from gymnasium.spaces import Discrete, MultiDiscrete
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import omnisafe
from omnisafe.envs.core import make, CMDP
from omnisafe.utils.config import Config
from omnisafe.typing import OmnisafeSpace
from omnisafe.models.actor_critic import ConstraintActorCriticDynamics

from gymnasium.envs.toy_text.cliffcircular import CliffCircularEnv
from omnisafe.envs.riverine_env import RiverineEnv


class EvalCADE:
    render_mode: Optional[str] = 'rgb_array'
    # render_mode: Optional[str] = 'human'

    keyboard_step: bool = False  # Use space bar to step environment

    def __init__(
        self,
        env_id: str,
        model_dir: str,
        model_name: str,
        eval_episodes: int,
        cade_method: str,
        save_path: Optional[str] = None,
        difficulty: int = 1,
        disp: bool = False,
        enable_safety_layer: bool = False,
        safety_layer_use_reward: bool = False,
        prediction_horizon: int = 0,
        horizon_cost_threshold: float = 0.,
    ):
        # Init model params
        self.end_id: str = env_id
        self.model_dir: str = model_dir
        self.model_name: str = model_name
        self.cade_method: str = cade_method
        self.eval_episodes: int = eval_episodes
        self.disp: bool = disp
        self.enable_safety_layer: bool = enable_safety_layer
        self.safety_layer_use_reward: bool = safety_layer_use_reward
        self.save_path: Optional[str] = save_path
        self.difficulty: int = difficulty
        self.H: int = prediction_horizon
        self.H_cost_thr: float = horizon_cost_threshold

        # Define stat file path
        seed: str = self.model_dir.split('/')[-1].split('-')[1]
        if self.enable_safety_layer:
            stat_file_name: str = f'{self.end_id}_{self.cade_method}_safety{self.enable_safety_layer}_seed{seed}_difficulty{self.difficulty}.csv'
        else:
            stat_file_name: str = f'{self.end_id}_{self.cade_method}_seed{seed}_difficulty{self.difficulty}.csv'
        self.stat_file_path: str = os.path.join(self.save_path, stat_file_name)

        # Init env
        env_kwarg: Dict[str, Any] = {
            'env_id': env_id,
            'render_mode': self.render_mode,
        }

        assert 0 <= difficulty <= 2

        if 'cliff' in self.end_id.lower():  # cliffcircular env
            env_kwarg['extra_cliff_num'] = difficulty
        else:  # riverine env
            if difficulty == 0:
                env_kwarg['env_id'] = 'easy'
            elif difficulty == 1:
                env_kwarg['env_id'] = 'medium'
            elif difficulty == 2:
                env_kwarg['env_id'] = 'hard'

        self.env: CMDP = make(**env_kwarg)
        self.obs_space: OmnisafeSpace = self.env.observation_space
        self.act_space: OmnisafeSpace = self.env.action_space

        # Load model configs
        self.cfgs: Config = self.load_cfgs()

        # Modify horizon
        if self.enable_safety_layer:
            if self.H > 0 and self.H_cost_thr > 0:
                self.cfgs.model_cfgs.dynamics.horizon = self.H
                self.cfgs.model_cfgs.dynamics.horizon_cost_threshold = self.H_cost_thr

        # Load model
        self.cad = self.load_model()

        # Set nominal (default) action
        if isinstance(self.act_space, Discrete):
            self.nominal_action = torch.tensor([[0]])  # CliffCircular
        elif isinstance(self.act_space, MultiDiscrete):
            self.nominal_action = torch.tensor([[1] * self.act_space.nvec.shape[0]])  # SRE
        else:
            print(f'Nominal action for {type(self.act_space)} is not supported.')
            raise NotImplementedError
        print(f'Nominal (no_op) action in this env: {self.nominal_action}')

        # Init keyboard env stepper
        if self.keyboard_step:
            self.space_pressed: bool = False
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()

    def load_cfgs(self) -> Config:
        """
        Load the config from the save directory.

        Args:

        Raises:
            FileNotFoundError: If the config file is not found.
        """
        cfg_path = os.path.join(self.model_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory {self.model_dir}.',
            ) from error
        return Config.dict2config(kwargs)

    def load_model(self) -> ConstraintActorCriticDynamics:
        assert os.path.exists(self.model_dir), f'Model dir {self.model_dir} does not exist!'

        model_path: str = os.path.join(self.model_dir, 'torch_save', self.model_name)
        assert os.path.exists(model_path), f'Model path {model_path} does not exist!'

        model_params = torch.load(model_path, map_location='cpu')

        cad: ConstraintActorCriticDynamics = ConstraintActorCriticDynamics(
            obs_space=self.obs_space,
            act_space=self.act_space,
            model_cfgs=self.cfgs.model_cfgs,
            epochs=1,  # Not used, for linear lr decay
        )

        # for name, module in cad.named_modules():
        #     print(f'{name=} {module=}')
        #     print('-'*40)

        cad.load_state_dict(model_params['actor_critic'])

        return cad

    def on_press(self, key):
        # Check if the spacebar is pressed
        if key == keyboard.Key.space:
            self.space_pressed = True

    def evaluate(self) -> Tuple[List[float], List[float]]:
        obs, info = self.env.reset()

        latent = None
        cur_episodes: int = 0
        ep_rew_list: List[float] = []
        ep_cost_list: List[float] = []
        ep_rew: float = 0.
        ep_cost: float = 0.
        last_action = self.nominal_action

        try:
            while cur_episodes < self.eval_episodes:
                # Wait for keyboard step
                while self.keyboard_step and not self.space_pressed:
                    time.sleep(0.1)
                if self.keyboard_step:
                    self.space_pressed = False

                # Reshape obs
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                elif obs.dim() == 3:
                    obs = obs.squeeze(0)
                # print(f'{obs.shape=}')

                # Step CAD
                act, logp, act_overlaid, reward_pred, cost_pred, latent = self.cad.step(
                    obs=obs,
                    last_act=last_action,
                    lagrangian_multiplier=1.0,  # equally weigh reward and cost
                    latent=latent,
                    deterministic=False,
                    enable_safety_layer=self.enable_safety_layer,
                    safety_layer_use_reward=self.safety_layer_use_reward,
                )

                # Step SDM and update cost_pred
                next_obs_pred = self.cad.sdm.predict(torch.cat([obs, act], dim=-1), round_to_int=True)
                with torch.no_grad():
                    cost_pred = self.cad.cost_critic(next_obs_pred)[0]  # only use the first cost critic

                # Step environment
                # print(f'{act=}')
                next_obs, reward, cost, terminated, truncated, info = self.env.step(act[0])
                last_action.copy_(act)
                ep_rew += reward.item()
                ep_cost += cost.item()

                # print(f'Pred reward:   {reward_pred.item():.2f},   Pred cost:   {cost_pred.item():.2f} \n'
                #       f'Actual reward: {reward.item():.2f},   Actual cost: {cost.item():.2f} \n')

                if self.disp:
                    self.display(
                        obs=obs,
                        next_obs_pred=next_obs_pred,
                        next_obs=next_obs,
                        act=act,
                        act_distribution=self.cad.actor._current_dist,
                        reward=reward,
                        reward_pred=reward_pred,
                        cost=cost,
                        cost_pred=cost_pred,
                    )

                obs = next_obs
                if terminated or truncated:
                    obs, info = self.env.reset()
                    latent = None
                    last_action = self.nominal_action

                    print(f'Episode {cur_episodes} finished with reward {ep_rew:.2f} and cost {ep_cost:.2f}.')

                    # Save to file
                    if self.save_path is not None:
                        self.save_to_file(ep_rew, ep_cost, overwrite=(cur_episodes == 0))

                    ep_rew_list.append(ep_rew)
                    ep_cost_list.append(ep_cost)
                    cur_episodes += 1
                    ep_rew = 0.
                    ep_cost = 0.
        except KeyboardInterrupt:
            print(f'Program interrupted by user.')
            self.close_display()
            if self.keyboard_step:
                self.listener.stop()

        # mean_ep_reward: float = ep_reward_sum / cur_episodes
        # mean_ep_cost: float = ep_cost_sum / cur_episodes
        # print(f'{cur_episodes} episodes finished, mean ep reward: {mean_ep_reward:.2f}, mean ep cost: {mean_ep_cost:.2f}')

        self.close_display()
        if self.keyboard_step:
            self.listener.stop()

        self.env.close()
        time.sleep(1)

        return ep_rew_list, ep_cost_list

    def save_to_file(self, ep_rew: float, ep_cost: float, overwrite: bool = False):
        assert self.save_path is not None

        if not os.path.exists(self.stat_file_path):
            with open(self.stat_file_path, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episodic Rewards", "Episodic Costs"])
        else:
            if overwrite:
                with open(self.stat_file_path, 'w', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Episodic Rewards", "Episodic Costs"])
                    writer.writerow([ep_rew, ep_cost])
            else:
                with open(self.stat_file_path, 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ep_rew, ep_cost])

    def display(
        self,
        obs: torch.Tensor,
        next_obs_pred: torch.Tensor,
        next_obs: torch.Tensor,
        act: torch.Tensor,
        act_distribution: Categorical,
        reward: torch.Tensor,
        reward_pred: torch.Tensor,
        cost: torch.Tensor,
        cost_pred: torch.Tensor,
    ):
        # print(f'{obs.shape=}')
        # print(f'{act.shape=}')
        # print(f'{next_obs.shape=}')
        # print(f'{next_obs_pred.shape=}')
        # print(f'{reward.shape=}')
        # print(f'{reward_pred.shape=}')
        # print(f'{cost.shape=}')
        # print(f'{cost_pred.shape=}')

        # To numpy
        obs = 1 - obs.squeeze().numpy().reshape(5, 5)
        next_obs = 1 - next_obs.squeeze().numpy().reshape(5, 5)
        next_obs_pred = 1 - next_obs_pred.squeeze().numpy().reshape(5, 5)
        act = act.squeeze().item()  # single int action
        reward = reward.item()
        reward_pred = reward_pred.squeeze().item()
        cost = cost.item()
        cost_pred = cost_pred.squeeze().item()
        act_probs = act_distribution.probs.detach().numpy()[0]  # normalized probabilities

        # print(f'{obs.shape=}')
        # print(f'{act=}')
        # print(f'{next_obs.shape=}')
        # print(f'{next_obs_pred.shape=}')
        # print(f'{reward=}')
        # print(f'{reward_pred=}')
        # print(f'{cost=}')
        # print(f'{cost_pred=}')
        # print(f'{act_probs=}')

        # Initialize figure and axes on first call
        if not hasattr(self, 'fig'):
            plt.ion()  # Turn on interactive mode
            self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8))

            # Create permanent plot elements
            # Observation plots (top row)
            self.obs_img = self.axes[0, 0].imshow(obs, cmap='gray', vmin=0, vmax=1)
            self.axes[0, 0].set_title("Current Observation")

            self.next_obs_img = self.axes[0, 2].imshow(next_obs, cmap='gray', vmin=0, vmax=1)
            self.axes[0, 2].set_title("True Next Observation")

            # Prediction plot (bottom right)
            self.next_obs_pred_img = self.axes[1, 2].imshow(next_obs_pred,
                                                            cmap='gray', vmin=0, vmax=1)
            self.axes[1, 2].set_title("Predicted Next Observation")

            # Action probability plot (bottom left)
            self.axes[1, 0].set_title("Action Probabilities")
            self.axes[1, 0].set_xticks([])
            self.axes[1, 0].set_yticks([])
            self.axes[1, 0].set_xlim(-1, 1)
            self.axes[1, 0].set_ylim(-1, 1)

            # Reward/cost comparison plots (middle column)
            self.axes[0, 1].set_title("Reward Comparison")
            self.axes[1, 1].set_title("Cost Comparison")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

        # Update all plots
        # 1. Update observation grids
        self.obs_img.set_data(obs)
        self.next_obs_img.set_data(next_obs)
        self.next_obs_pred_img.set_data(next_obs_pred)

        # 2. Update action probability plot
        ax = self.axes[1, 0]
        ax.clear()
        colors = ['red', 'blue', 'green', 'purple']
        labels = ['Up', 'Right', 'Down', 'Left']

        # Plot directional arrows using probabilities[1-4]
        ax.quiver(0, 0, 0, act_probs[1], scale=1, scale_units='xy',
                  angles='xy', color=colors[0], label=labels[0])
        ax.quiver(0, 0, act_probs[2], 0, scale=1, scale_units='xy',
                  angles='xy', color=colors[1], label=labels[1])
        ax.quiver(0, 0, 0, -act_probs[3], scale=1, scale_units='xy',
                  angles='xy', color=colors[2], label=labels[2])
        ax.quiver(0, 0, -act_probs[4], 0, scale=1, scale_units='xy',
                  angles='xy', color=colors[3], label=labels[3])

        # Plot no-op probability as circle
        circle = plt.Circle((0, 0), act_probs[0] / 2, color='orange', alpha=0.5, label='No Op')
        ax.add_patch(circle)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper right')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Action Probabilities")

        # 3. Update reward comparison
        ax = self.axes[0, 1]
        ax.clear()
        ax.bar(['Actual', 'Predicted'], [reward, reward_pred], color=['blue', 'orange'])
        ax.set_ylabel('Reward')
        # y_min = min(reward, reward_pred) - 0.1
        # y_max = max(reward, reward_pred) + 0.1
        # ax.set_ylim(y_min if y_min < y_max else y_max - 0.2,
        #             y_max if y_max > y_min else y_min + 0.2)
        ax.set_ylim(0, 1.5)
        ax.set_title("Reward Comparison")

        # 4. Update cost comparison
        ax = self.axes[1, 1]
        ax.clear()
        ax.bar(['Actual', 'Predicted'], [cost, cost_pred], color=['blue', 'orange'])
        ax.set_ylabel('Cost')
        # y_min = min(cost, cost_pred) - 0.1
        # y_max = max(cost, cost_pred) + 0.1
        # ax.set_ylim(y_min if y_min < y_max else y_max - 0.2,
        #             y_max if y_max > y_min else y_min + 0.2)
        ax.set_ylim(0, 1)
        ax.set_title("Cost Comparison")

        # Force redraw and brief pause
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close_display(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig
            del self.axes


def get_cade_statistics(
    env_name: str,
    difficulty: int,
    safety_layer_enabled: bool = False,
    digit: int = 2
) -> Tuple[List[float], ...]:
    dir_name: str = f'evaluations/{env_name}'
    assert os.path.exists(dir_name)

    mgae_rewards = []
    mgae_costs = []
    lagrange_rewards = []
    lagrange_costs = []
    safety_rewards = []
    safety_costs = []

    for item in os.scandir(dir_name):
        if item.is_file():
            if f'difficulty{difficulty}' in item.name:
                if safety_layer_enabled:
                    if 'safetyTrue' not in item.name:
                        continue
                if 'mgae' in item.name:
                    print(f'{item.name=}')
                    df = pd.read_csv(os.path.join(dir_name, item.name))
                    mgae_rewards += df['Episodic Rewards'].to_list()
                    mgae_costs += df['Episodic Costs'].to_list()
                elif 'lagrange' in item.name:
                    print(f'{item.name=}')
                    df = pd.read_csv(os.path.join(dir_name, item.name))
                    lagrange_rewards += df['Episodic Rewards'].to_list()
                    lagrange_costs += df['Episodic Costs'].to_list()
                elif 'safety_layer' in item.name:
                    print(f'{item.name=}')
                    df = pd.read_csv(os.path.join(dir_name, item.name))
                    safety_rewards += df['Episodic Rewards'].to_list()
                    safety_costs += df['Episodic Costs'].to_list()
                else:
                    pass

    mgae_rew_mean, mgae_cost_mean = np.mean(mgae_rewards), np.mean(mgae_costs)
    mgae_rew_std, mgae_cost_std = np.std(mgae_rewards), np.std(mgae_costs)
    lagrange_rew_mean, lagrange_cost_mean = np.mean(lagrange_rewards), np.mean(lagrange_costs)
    lagrange_rew_std, lagrange_cost_std = np.std(lagrange_rewards), np.std(lagrange_costs)
    safety_rew_mean, safety_cost_mean = np.mean(safety_rewards), np.mean(safety_costs)
    safety_rew_std, safety_cost_std = np.std(safety_rewards), np.std(safety_costs)

    print(f'MGAE reward: {mgae_rew_mean:.{digit}f} +- {mgae_rew_std:.{digit}f} \n'
          f'MGAE cost: {mgae_cost_mean:.{digit}f} +- {mgae_cost_std:.{digit}f} \n'
          f'Lagrange reward: {lagrange_rew_mean:.{digit}f} +- {lagrange_rew_std:.{digit}f} \n'
          f'Lagrange cost: {lagrange_cost_mean:.{digit}f} +- {lagrange_cost_std:.{digit}f} \n'
          f'Safety reward: {safety_rew_mean:.{digit}f} +- {safety_rew_std:.{digit}f} \n'
          f'Safety cost: {safety_cost_mean:.{digit}f} +- {safety_cost_std:.{digit}f} \n')

    return mgae_rewards, mgae_costs, lagrange_rewards, lagrange_costs, safety_rewards, safety_costs


def get_cade_stat_all(
    env_name: str,
    safety_layer_enabled: bool = False,
    digit: int = 2
):
    """
    Get stat across 3 difficulty levels of all methods all seeds
    Args:
        env_name:
        safety_layer_enabled:
        digit:

    Returns:

    """
    mgae_rewards = []
    mgae_costs = []
    lagrange_rewards = []
    lagrange_costs = []
    safety_rewards = []
    safety_costs = []

    for difficulty in [0, 1, 2]:
        mr, mc, lr, lc, sr, sc = get_cade_statistics(
            env_name=env_name,
            difficulty=difficulty,
            safety_layer_enabled=safety_layer_enabled,
            digit=digit,
        )
        mgae_rewards += mr
        mgae_costs += mc
        lagrange_rewards += lr
        lagrange_costs += lc
        safety_rewards += sr
        safety_costs += sc

    mgae_rew_mean, mgae_cost_mean = np.mean(mgae_rewards), np.mean(mgae_costs)
    mgae_rew_std, mgae_cost_std = np.std(mgae_rewards), np.std(mgae_costs)
    lagrange_rew_mean, lagrange_cost_mean = np.mean(lagrange_rewards), np.mean(lagrange_costs)
    lagrange_rew_std, lagrange_cost_std = np.std(lagrange_rewards), np.std(lagrange_costs)
    safety_rew_mean, safety_cost_mean = np.mean(safety_rewards), np.mean(safety_costs)
    safety_rew_std, safety_cost_std = np.std(safety_rewards), np.std(safety_costs)

    print('All 3 difficulty levels stat:')
    print(f'MGAE reward: {mgae_rew_mean:.{digit}f} +- {mgae_rew_std:.{digit}f} \n'
          f'MGAE cost: {mgae_cost_mean:.{digit}f} +- {mgae_cost_std:.{digit}f} \n'
          f'Lagrange reward: {lagrange_rew_mean:.{digit}f} +- {lagrange_rew_std:.{digit}f} \n'
          f'Lagrange cost: {lagrange_cost_mean:.{digit}f} +- {lagrange_cost_std:.{digit}f} \n'
          f'Safety reward: {safety_rew_mean:.{digit}f} +- {safety_rew_std:.{digit}f} \n'
          f'Safety cost: {safety_cost_mean:.{digit}f} +- {safety_cost_std:.{digit}f} \n')


if __name__ == '__main__':
    # Define model evaluation params
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-04-14-10-11'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-06-21-26-28'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-07-12-18-30'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-07-13-07-28'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-07-16-31-19'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-07-17-12-27'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-10-21-03-36'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-11-15-42-23'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-11-16-08-30'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-12-16-07-56'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-12-21-43-33'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-14-15-05-51'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-14-15-30-08'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-14-16-13-05'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-14-19-56-03'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-16-14-42-34'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-16-15-30-23'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-042-2025-02-17-16-46-14'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-17-17-17-58'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-17-20-41-30'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-17-21-00-59'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-17-21-35-44'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-18-13-14-00'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-18-15-21-04'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-18-20-16-15'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-18-21-30-06'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-18-15-59-58'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-19-13-04-10'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-002-2025-02-20-21-56-08'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-002-2025-02-21-14-04-26'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-002-2025-02-21-13-45-17'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-002-2025-02-21-14-52-15'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-002-2025-02-21-15-15-22'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-042-2025-02-21-22-09-16'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-100-2025-02-22-14-55-50'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-26-14-29-15'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-005-2025-02-26-16-18-57'
    # model_dir: str = './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-005-2025-02-26-16-35-45'

    # Eval single model
    # # Init CAD evaluator
    # cad_eval = EvalCAD(
    #     model_dir=model_dir,
    #     model_name=model_name,
    #     eval_episodes=eval_episodes,
    #     disp=False,
    #     enable_safety_layer=False,
    #     safety_layer_use_reward=False,
    # )
    #
    # # Start evaluation
    # cad_eval.evaluate()

    # env_id: str = 'CliffCircular-v1'  # or medium
    env_id: str = 'medium'  # or CliffCircular-v1
    save_path: str = 'evaluations/cliffcircular' if 'CliffCircular' in env_id else 'evaluations/riverine'
    model_name: str = 'epoch-1500.pt' if 'CliffCircular' in env_id else 'epoch-350.pt'
    eval_episodes: int = 30  # Evaluation episodes number
    difficulty: int = 0  # [0, 2], different difficulty levels of env
    evaluate: bool = False  # Eval if True, read csv data and get stat if False
    safety_layer_enabled: bool = True
    merge_across_envs: bool = True  # Get stats across all difficulty levels

    if env_id == 'CliffCircular-v1':
        cade_dir_dict: Dict[str, List[str]] = {
            'mgae': ['./runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-28-09-19-56',
                     './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-042-2025-02-28-09-04-11',
                     './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-100-2025-02-28-10-05-29'],
            'lagrange': ['./runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-28-09-37-35',
                         './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-042-2025-02-28-08-50-27',
                         './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-100-2025-02-28-10-19-06'],
            'safety_layer': ['./runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2025-02-28-11-12-02',
                             './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-042-2025-02-28-11-27-30',
                             './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-100-2025-02-28-10-38-55'],
        }
    else:
        cade_dir_dict: Dict[str, List[str]] = {
            'mgae': ['./runs/FOCOPS_CACD-{medium}/seed-000-2025-03-01-13-10-24',
                     './runs/FOCOPS_CACD-{medium}/seed-042-2025-03-01-16-15-11',
                     './runs/FOCOPS_CACD-{medium}/seed-100-2025-03-02-13-54-29'],
            'lagrange': ['./runs/FOCOPS_CACD-{medium}/seed-000-2025-03-01-15-06-43',
                         './runs/FOCOPS_CACD-{medium}/seed-042-2025-03-02-12-45-53',
                         './runs/FOCOPS_CACD-{medium}/seed-100-2025-03-02-15-10-38'],
            'safety_layer': ['./runs/FOCOPS_CACD-{medium}/seed-000-2025-03-02-20-33-01',
                             './runs/FOCOPS_CACD-{medium}/seed-042-2025-03-02-22-20-30',
                             './runs/FOCOPS_CACD-{medium}/seed-100-2025-03-03-09-49-51'],
        }

    if evaluate:
        # Eval a list of models
        for cade, model_dir_list in cade_dir_dict.items():
            cade_ep_rew_list = []
            cade_ep_cost_list = []

            for model_dir in model_dir_list:
                # Init CADE evaluator
                cade_eval = EvalCADE(
                    env_id=env_id,
                    model_dir=model_dir,
                    model_name=model_name,
                    eval_episodes=eval_episodes,
                    cade_method=cade,
                    save_path=save_path,
                    difficulty=difficulty,
                    disp=False,
                    enable_safety_layer=safety_layer_enabled,
                    safety_layer_use_reward=False,
                    prediction_horizon=3,
                    horizon_cost_threshold=0.3,
                )

                # Start evaluation
                ep_rew_list, ep_cost_list = cade_eval.evaluate()
                cade_ep_rew_list += ep_rew_list
                cade_ep_cost_list += ep_cost_list

            print(f'Method: {cade}, level: {difficulty}, reward: {np.mean(cade_ep_rew_list):.2f} +- {np.std(cade_ep_rew_list):.2f}')
            print(f'Method: {cade}, level: {difficulty}, cost: {np.mean(cade_ep_cost_list):.2f} +- {np.std(cade_ep_cost_list):.2f}')
    else:
        if merge_across_envs:
            get_cade_stat_all(
                env_name='cliffcircular' if 'Cliff' in env_id else 'riverine',
                safety_layer_enabled=safety_layer_enabled,
                digit=2,
            )
        else:
            get_cade_statistics(
                env_name='cliffcircular' if 'Cliff' in env_id else 'riverine',
                difficulty=difficulty,
                safety_layer_enabled=safety_layer_enabled,
                digit=2
            )






