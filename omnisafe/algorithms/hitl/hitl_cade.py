import os
import sys
import json
import time
import csv
import re
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Literal, Union, get_args
from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary
import matplotlib.pyplot as plt
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.models.actor_critic import ConstraintActorDynamicsEstimator
from omnisafe.common.buffer.onpolicy_hitl_buffer import OnPolicyHITLBuffer, save_buffer_to_csv, load_buffer_from_csv
from omnisafe.utils.math import discount_cumsum, kld_multi_categorical
from omnisafe.utils.key2action import Key2ActionDrone, Key2ActionBoat
from omnisafe.utils.patchification import get_patchified_mask

# For real world deployment
from omnisafe.algorithms.hitl.perception_infer import PerceptionInfer
from splashdrone4.keyboard_control import KeyboardControl


# Types of HITL losses, 'None' means no HITL
LossType = Literal['None', 'IWR', 'HG-DAgger', 'BT', 'DPO', 'Indirect']

# Custom types
Loss2RewStep: type = Dict[str, Tuple[List[float], List[int]]]


class HitlCade:
    def __init__(
        self,
        model_dir: str,
        model_name: str,
        env_id: Optional[str] = None,
        segmentation_engine_path: Optional[str] = None,
        eval_episodes: int = 1,
        deterministic: bool = True,
        save_path: Optional[str] = None,
        save_buffer: bool = True,
        difficulty: int = 1,
        buffer_size: int = 1000,
        retrain_epoch: int = 3,
        loss_type: LossType = 'None',  # None means no hitl loss
        save_ckpts: bool = False,
        render_mode: Optional[str] = 'human',
        enable_hitl: bool = True,
        enable_retrain: bool = True,
        device: Union[torch.device, str] = 'cpu'  # Device to run the model on, e.g., 'cuda:0' or 'cpu'
    ):
        """
        Human-in-the-loop Constrained Actor Dynamics Estimator (HITL-CADE) evaluation and improvement process.

        Args:
            env_id: environment id. For Safe Riverine Environment, choose from {easy, medium, hard}.
            model_dir: model directory containing the CADE pytorch model, which is under the torch_save dir.
            model_name: exact model name that has "pt" suffix.
            eval_episodes: number of episodes to evaluate the loaded CADE model.
            deterministic: whether to use deterministic policy or not.
            save_path: folder to store the episodic statistics of evaluation results.
            save_buffer: whether to save per-step result of evaluations.
            difficulty: Explicit integer of difficulty level, has to be in range [0, 2], where 0 is easy and 2 is hard.
            buffer_size: Buffer size of per-episode data.
            retrain_epoch: number of epochs to retrain the CADE using the buffered episodic data.
            loss_type: human-in-the-loop loss type.
            save_ckpts: whether to save the checkpoints of retrained CADE model
            render_mode: render mode of the environment.
            enable_hitl: Whether to enable human-in-the-loop during evaluation.
            enable_retrain: Whether to enable retraining of CADE during evaluation.
        """
        # Init parameters
        self.env_id: Optional[str] = env_id
        self.model_dir: str = model_dir
        self.model_name: str = model_name
        self.eval_episodes: int = eval_episodes
        self.deterministic: bool = deterministic
        self.save_path: Optional[str] = save_path
        self.save_buffer: bool = save_buffer
        self.difficulty: int = difficulty
        self.buffer_size: int = buffer_size
        self.retrain_epoch: int = retrain_epoch
        self.loss_type: LossType = loss_type
        self.save_ckpts: bool = save_ckpts
        self.render_mode: Optional[str] = render_mode
        self.enable_hitl: bool = enable_hitl
        self.enable_retrain: bool = enable_retrain
        self.device: Union[torch.device, str] = device

        # Variables
        self.ep_num: int = 0

        # Define stat file path for all eval episodes
        # Stat includes: episodic reward, episodic cost, episodic steps
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

            if 'seed' in self.model_dir:
                self.seed: str = self.model_dir.split('/')[-1].split('-')[1]
            else:
                self.seed: str = '000'  # Default seed if not specified in model_dir
            env_name: str = 'real' if self.env_id is None else self.env_id  # 'real' for real-world riverine environment
            # TODO unify the filenames for sim and real
            stat_file_name: str = f'{env_name}_hitl{self.enable_hitl}_seed{self.seed}_difficulty{self.difficulty}_loss{self.loss_type}.csv'
            self.stat_file_path: str = os.path.join(self.save_path, stat_file_name)

        # Init env
        if self.env_id is not None:
            self.setup_env()
            logger.info(f'Environment is created.')
        else:
            logger.info('Real world riverine environment is used.')
            self.env = None
            self.obs_space: OmnisafeSpace = HitlCade.gen_obs_space()
            self.act_space: OmnisafeSpace = HitlCade.gen_act_space()

            # Init keyboard controller of Splashdrone4
            self.keyboard_control = KeyboardControl(save_data=True, data_len=buffer_size, debug=True)

            # Init semantic segmentation inference engine
            self.segmentation_engine = PerceptionInfer(engine_path=segmentation_engine_path)
            logger.info(f'Segmentation engine loaded from {segmentation_engine_path}.')

        # Init the buffer
        self.buffer = OnPolicyHITLBuffer(
            obs_space=self.obs_space,
            act_space=self.act_space,
            size=self.buffer_size,
            device=self.device,
        )

        # Load model configs
        self.cfgs: Config = self.load_cfgs()

        # Load model
        self.cade = self.load_model()
        logger.info(f'CADE model is loaded.')

        # Set nominal (default) action
        assert isinstance(self.act_space, MultiDiscrete)
        self.nominal_action = torch.tensor([[1] * self.act_space.nvec.shape[0]]).to(self.device)
        logger.info(f'Nominal action: {self.nominal_action}')

        # Set human-in-the-loop interruption for simulation
        if self.enable_hitl and self.env is not None:
            self.k2a = Key2ActionDrone()  # TODO only support drone for now
            logger.info(f'Human-in-the-loop keyboard interruption is enabled.')

    @staticmethod
    def gen_obs_space() -> OmnisafeSpace:
        return MultiBinary(16 * 16)

    @staticmethod
    def gen_act_space() -> OmnisafeSpace:
        return MultiDiscrete([3, 3, 3, 3])

    def setup_env(self):
        """
        Setup simulation environment.
        Returns:

        """
        assert self.env_id is not None, 'Environment ID must be specified to setup the environment.'

        # Lazy import when env_id is not None
        from omnisafe.envs.core import make, CMDP
        from omnisafe.envs.riverine_env import RiverineEnv  # for Unity Safe Riverine Environment of drone
        # from vrx_gym.river_follow_env import WamvGazeboEnv  # for Gazebo VRX WAM-V environment of boat

        env_kwarg: Dict[str, Any] = {
            'env_id': env_id,
            'render_mode': self.render_mode,
            'max_idle_steps': 500000,  # allow time for human intervention
        }
        assert 0 <= difficulty <= 2, f'Difficulty {difficulty} is not in range [0, 2].'
        if difficulty == 0:
            env_kwarg['env_id'] = 'easy'
        elif difficulty == 1:
            env_kwarg['env_id'] = 'medium'
        elif difficulty == 2:
            env_kwarg['env_id'] = 'hard'

        self.env: CMDP = make(**env_kwarg)
        self.obs_space: OmnisafeSpace = self.env.observation_space
        self.act_space: OmnisafeSpace = self.env.action_space

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

    def load_model(self) -> ConstraintActorDynamicsEstimator:
        """
        Load CADE model.

        Returns:
            cade: The loaded CADE model.

        """
        assert os.path.exists(self.model_dir), f'Model dir {self.model_dir} does not exist!'

        model_path: str = os.path.join(os.path.dirname(__file__), self.model_dir, 'torch_save', self.model_name)
        assert os.path.exists(model_path), f'Model path {model_path} does not exist!'

        model_params = torch.load(model_path, map_location=self.device)

        cade: ConstraintActorDynamicsEstimator = ConstraintActorDynamicsEstimator(
            obs_space=self.obs_space,
            act_space=self.act_space,
            model_cfgs=self.cfgs.model_cfgs,
            epochs=1,  # Not used, for linear lr decay
        )

        cade.load_state_dict(model_params['actor_critic'])  # TODO might need to change the name here

        cade = cade.to(self.device)

        # check if all submodules of CADE are on correct device
        for name, param in cade.named_parameters():
            assert param.device == torch.device(self.device), \
                f'Parameter {name} is on {param.device}, expected {self.device}'

        return cade

    def save_model(self, name: str) -> None:
        """
        Save current CADE model.

        Args:
            name: Model name with suffix.

        Returns:
            None
        """
        assert os.path.exists(self.model_dir), f'Model dir {self.model_dir} does not exist!'

        model_path: str = os.path.join(self.model_dir, 'torch_save', name)

        torch.save({'actor_critic': self.cade.state_dict()}, model_path)  # TODO might need to change the name here

    def save_checkpoint(self):
        """
        Save current checkpoint with the name.

        Returns:

        """
        if not self.save_ckpts:
            logger.warning('save_checkpoint should be enabled when initializing HITL CADE.')
            return

        if self.env is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name: str = f'real-episode-{self.ep_num:03}-hitl-{self.enable_hitl}-loss-{self.loss_type}-{timestamp}.pt'
        else:
            ckpt_name: str = f'sim-episode-{self.ep_num:03}-hitl-{self.enable_hitl}-loss-{self.loss_type}.pt'

        self.save_model(name=ckpt_name)

        logger.info(f'Checkpoint is saved as {ckpt_name}.')

    def save_ep_stats_to_file(
        self,
        ep_rew: float,
        ep_cost: float,
        ep_steps: float,
        overwrite: bool = False,
    ) -> None:
        """
        Save episodic statistics into csv file.

        Args:
            ep_rew: Episodic reward.
            ep_cost: Episodic cost.
            ep_steps: Episodic steps.
            overwrite: Whether to overwrite the existing file or not.

        Returns:
            None
        """
        assert self.save_path is not None, f'Episodic stats save path should not be None.'
        assert self.env is not None, f'Episodic stats are only savable in simulation.'

        if not os.path.exists(self.stat_file_path) or overwrite:
            with open(self.stat_file_path, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(['Episodic Rewards', 'Episodic Costs', 'Episodic Steps'])
                writer.writerow([ep_rew, ep_cost, ep_steps])
        else:
            with open(self.stat_file_path, 'a', newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ep_rew, ep_cost, ep_steps])

    def save_buffer_to_file(self):
        if not self.save_buffer:
            logger.warning('Enable save_buffer when initializing HITL CADE.')
            return

        if self.env is None:  # Real world
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename: str = f'real_hitl{self.enable_hitl}_loss{self.loss_type}_episode{self.ep_num:03}_{timestamp}.csv'
        else:  # Simulation
            filename: str = f'{self.env_id}_hitl{self.enable_hitl}_loss{self.loss_type}_episode{self.ep_num:03}.csv'

        filename = os.path.join(self.save_path, filename)
        data = self.buffer.get(reset=False)  # don't reset here, reset after retraining
        save_buffer_to_csv(data=data, filename=filename)
        logger.info(f'Buffer data of episode {self.ep_num} is saved to {filename}.')

    def close(self) -> None:
        """
        Close the environment, optionally close keyboard reader.

        Returns:
            None
        """
        if self.env is not None:
            self.env.close()

            if self.enable_hitl:
                self.k2a.listener.stop()

    def img2obs(self, img: np.ndarray, show_mask: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Convert rgb image to binary water mask via trained semantic segmentation model (tensorrt),
        then convert to observation tensor, which is flattened patchified water mask.

        Args:
            img: Input image in numpy array format with shape [H, W, C].

        Returns:
            obs: Observation tensor with shape [1, 16 x 16].
        """
        assert img.ndim == 3, f'Image must be a 3D array, got {img.ndim}D.'

        img = img.transpose((2, 0, 1)).astype(np.float32) / 255  # Normalize image to [0, 1]

        _, mask = self.segmentation_engine.infer(img, mask_path=None)

        if show_mask:
            cv2.imshow('Mask', mask)
            cv2.waitKey(1)

        obs = get_patchified_mask(
            mask=mask,
            is_uint8=True,
            patch_size_x=8,
            patch_size_y=8,
            patch_step=8,
            binary_threshold=0.5,
            patch_threshold=0.5,
        )
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, W x H]

        return obs, mask

    def evaluate(self) -> Tuple[List[float], List[float]]:
        """
        Note: this function should ONLY be called in simulation.
        Evaluate the loaded policy, optionally retrain it while inferencing if human corrections are available.

        Returns:
            Tuple of per-episode reward list and per-episode cost list.
        """
        assert self.env is not None
        obs, info = self.env.reset()

        latent = None
        ep_rew_list: List[float] = []  # per-episode reward return
        ep_cost_list: List[float] = []  # per-episode cost return
        ep_rew: float = 0.
        ep_cost: float = 0.
        step: int = 0
        last_action = self.nominal_action.clone()

        try:
            while self.ep_num < self.eval_episodes:
                # Reshape obs
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                elif obs.dim() == 3:
                    obs = obs.squeeze(0)

                # Step CADE
                # obs.shape=torch.Size([1, 256]), last_action.shape=torch.Size([1, 4]), latent.shape=torch.Size([1, 64])
                agent_act, logp, act_overlaid, reward_pred, cost_pred, latent = self.cade.step(
                    obs=obs,
                    last_act=last_action,
                    lagrangian_multiplier=1.0,  # equally weigh reward and cost
                    latent=latent,
                    deterministic=self.deterministic,
                    enable_safety_layer=False,  # TODO enable this for human demonstration
                    safety_layer_use_reward=False,
                )

                # Wait for human confirmation (whether to intervene or not)
                human_intervened: bool = False
                if self.enable_hitl:
                    human_action = self.k2a.get_multi_discrete_action()
                    while human_action == [1, 1, 1, 1]:  # Waiting for human confirmation (take over control or not)
                        # print(f'Waiting for human actions ...')
                        time.sleep(0.5)
                        human_action = self.k2a.get_multi_discrete_action()
                    if human_action is not None:  # Human take over control
                        human_intervened = True
                        act = torch.tensor([human_action])
                    else:  # Human confirms/acknowledges agent's action
                        act = agent_act
                else:
                    act = agent_act

                # Step SDM and update cost_pred
                next_obs_pred = self.cade.sdm.predict(torch.cat([obs, act], dim=-1), round_to_int=True)
                with torch.no_grad():
                    cost_pred = self.cade.cost_critic(next_obs_pred)[0]  # only use the first cost critic

                # Step environment
                next_obs, reward, cost, terminated, truncated, info = self.env.step(act[0])  # TODO check env None
                step += 1
                last_action.copy_(act)
                ep_rew += reward.item()
                ep_cost += cost.item()

                # print(f'Pred reward:   {reward_pred.item():.1f}, Actual reward: {reward.item():.1f} \n'
                #       f'Pred cost:   {cost_pred.item():.1f}, Actual cost: {cost.item():.1f} \n'
                #       f'Agent action: {agent_act}, Human action: {human_action if human_intervened else None} \n')

                # Save data into buffer
                self.buffer.store(
                    obs=obs.squeeze(),
                    act=act.squeeze(),
                    act_agent=agent_act.squeeze(),
                    logp=logp,
                    act_overlaid=torch.tensor(human_intervened, dtype=torch.float32),
                    reward=reward,
                    reward_pred=reward_pred,
                    cost=cost,
                    cost_pred=cost_pred,
                    done=terminated or truncated,
                    next_obs=next_obs,
                    next_obs_pred=next_obs_pred,
                )

                obs = next_obs
                if terminated or truncated or self.buffer.full():  # TODO Buffer full is an early stop trick
                    obs, info = self.env.reset()
                    latent = None
                    last_action.copy_(self.nominal_action)

                    logger.info(f'Episode {self.ep_num} finished with reward {ep_rew:.2f}, cost {ep_cost:.2f}, step: {step}.')

                    # Save per-episode stats to file
                    if self.save_path is not None:
                        self.save_ep_stats_to_file(
                            ep_rew=ep_rew,
                            ep_cost=ep_cost,
                            ep_steps=step,
                            overwrite=(self.ep_num == 0),
                        )

                    # Save per-step episodic stats to file
                    if self.save_buffer:
                        self.save_buffer_to_file()

                    # Update stats
                    ep_rew_list.append(ep_rew)
                    ep_cost_list.append(ep_cost)
                    self.ep_num += 1
                    step = 0
                    ep_rew = 0.
                    ep_cost = 0.

                    # Retrain CADE
                    if self.enable_hitl and self.enable_retrain:
                        data = self.buffer.get()
                        self.retrain(data=data, epoch=self.retrain_epoch)

                    # Clear the buffer
                    # TODO Might allow buffer to store multiple episodes data?
                    self.buffer.clear()

        except KeyboardInterrupt:
            print(f'Program interrupted by user.')
        except Exception as e:
            print(f'Unexpected error occurred: {e}.')
        finally:
            self.close()

            ep_rew_mean, ep_rew_std = np.mean(ep_rew_list), np.std(ep_rew_list)
            ep_cost_mean, ep_cost_std = np.mean(ep_cost_list), np.std(ep_cost_list)
            logger.info(
                f'Evaluated {self.ep_num} episodes, '
                f'ep_rew: {ep_rew_mean:.1f}+-{ep_rew_std:.1f}, '
                f'ep_cost: {ep_cost_mean:.1f}+-{ep_cost_std:.1f}')

            time.sleep(1)  # Give some time for Unity to close
            return ep_rew_list, ep_cost_list

    def deploy(self):
        """
        Note: this function should ONLY be called in real world deployment.
        Deploy the HITL CADE algorithm for human-in-the-loop evaluation and improvement.

        Returns:
            Tuple of per-episode reward list and per-episode cost list.
        """
        assert self.env is None, 'Deploy mode does not support environment interaction.'

        latent = None
        step: int = 0
        last_action = self.nominal_action.clone()

        try:
            # Main loop
            while True:
                img, wp_yaw, ep_reset, g2g, overlaid, action_taken = self.keyboard_control.step(action=None)

                # Wait for human approval that current observation is stable for policy inference
                while not g2g:
                    logger.debug('Waiting for good-to-go signal from human ...')
                    img, wp_yaw, ep_reset, g2g, overlaid, action_taken = self.keyboard_control.step(action=None)

                # Let policy do inference if good to go
                obs, mask = self.img2obs(img, show_mask=True)
                agent_act, logp, act_overlaid_policy, reward_pred, cost_pred, latent = self.cade.step(
                    obs=obs,
                    last_act=last_action,
                    lagrangian_multiplier=1.0,  # equally weigh reward and cost
                    latent=latent,
                    deterministic=self.deterministic,
                    enable_safety_layer=False,  # TODO enable this for human demonstration
                    safety_layer_use_reward=False,
                )

                # Wait for human approval or correction of policy-chosen action, blocking call
                _, _, ep_reset, g2g, overlaid, act = self.keyboard_control.step(action=agent_act[0].cpu().tolist())

                # Log data to h5 file
                self.keyboard_control.log_data(
                    wp_yaw=wp_yaw,
                    image=img,
                    mask=mask,
                    action=np.array(act),
                    overlaid=overlaid,
                )

                act = torch.tensor(act).unsqueeze(0).to(self.device)  # [1, 4]
                logger.info(f'Episode {self.ep_num} Step {step}: Action {act.cpu().tolist()}')

                # Step SDM and update cost_pred
                next_obs_pred = self.cade.sdm.predict(torch.cat([obs, act], dim=-1), round_to_int=True)
                with torch.no_grad():
                    cost_pred = self.cade.cost_critic(next_obs_pred)[0]  # only use the first cost critic

                # Save to buffer
                self.buffer.store(
                    obs=obs.squeeze(),
                    act=act.squeeze(),
                    act_agent=agent_act.squeeze(),
                    logp=logp,
                    act_overlaid=torch.tensor(overlaid, dtype=torch.float32),
                    # reward=torch.tensor(0.),  # Not available in real world
                    reward_pred=reward_pred,
                    # cost=torch.tensor(0.),  # Not available in real world
                    cost_pred=cost_pred,
                    done=ep_reset,
                    # next_obs=next_obs,  # Not available until the next step in real world
                    next_obs_pred=next_obs_pred,
                )

                if ep_reset or self.buffer.full():  # Episode terminated by human
                    logger.info(f'Episode {self.ep_num} terminated with {step} steps.')

                    latent = None
                    last_action.copy_(self.nominal_action)
                    step = 0

                    # Save per-step episodic stats to file
                    if self.save_buffer:
                        self.save_buffer_to_file()

                    # Retrain CADE
                    if self.enable_hitl and self.enable_retrain:
                        data = self.buffer.get()
                        self.retrain(data=data, epoch=self.retrain_epoch)

                    self.ep_num += 1

                    # Clear the buffer
                    self.buffer.clear()  # TODO Might allow buffer to store multiple episodes data?
                else:  # Episode not terminated, continue navigation
                    step += 1
        except KeyboardInterrupt:
            print(f'Program interrupted by user.')
        except Exception as e:
            print(f'Unexpected error occurred: {e}.')
        finally:
            self.keyboard_control.close()

    def retrain(
        self,
        data: dict[str, torch.Tensor],
        epoch: Optional[int] = 3,
    ) -> None:
        """
        Retrain policy based on the most recent episode with human correction (intervention + demonstration).

        Args:
            data: Data collected in the most recent episode with human correction.
            epoch: Number of epochs to retrain the CADE using the buffered episodic data.
                   If None, use self.retrain_epoch.

        Returns:
            None
        """
        obs = data['obs']
        act = data['act']
        act_agent = data['act_agent']
        act_overlaid = data['act_overlaid']
        logp = data['logp']  # logp of agent actions
        done = data['done'].bool()

        # Filter out done data samples (usually end of an episode in simulation)
        if self.env is not None:
            obs = obs[~done]
            act = act[~done]
            act_agent = act_agent[~done]
            act_overlaid = act_overlaid[~done]
            logp = logp[~done]

        # Update SDM in real world (no need to train in sim since sdm already converged)
        if self.env is None:
            obs_cur = obs[:-1]
            act_cur = act[:-1]
            obs_next = obs[1:]
            self.update_sdm(obs=obs_cur, act=act_cur, next_obs=obs_next)
            logger.info('SDM retraining is done.')

        # Check if there are any human corrections
        if not act_overlaid.any():
            logger.info(f'No human corrections in episode {self.ep_num}, skipping policy retraining.')
            return

        # Get initial policy and reward prediction (before epoch 0)
        with torch.no_grad():
            init_distribution, init_reward_pred = self.cade.forward_actor_reward(obs, act)
        assert isinstance(init_distribution, List), f'Currently only support multi-discrete action space.'

        logger.info(f'Starting retraining for episode {self.ep_num} ...')

        for e in range(epoch if epoch is not None else self.retrain_epoch):
            # Zero gradients
            self.cade.gru_optimizer.zero_grad()
            self.cade.actor_optimizer.zero_grad()
            self.cade.reward_critic_optimizer.zero_grad()

            # Calculate loss based on different HITL loss types
            if self.loss_type == 'Indirect':
                # First update reward estimator
                reward_loss = self.calc_reward_estimator_loss(obs, act, act_agent, act_overlaid)
                reward_loss.backward()
                self.cade.reward_critic_optimizer.step()
                logger.info('Training of reward estimator is done.')

                # Then update policy using the learned reward as advantage
                distribution, reward_pred = self.cade.forward_actor_reward(obs, act)
                reward_adv = self.calc_reward_adv(reward_pred[0], baseline_return=25.)
                loss = self.policy_loss_by_hitl_reward(distribution, init_distribution, act, logp, reward_adv)
            else:
                distribution, reward_pred = self.cade.forward_actor_reward(obs, act)
                if self.loss_type == 'HG-DAgger':
                    loss = self.weighted_bc_loss(distribution, act, act_overlaid, hg_dagger=True)
                elif self.loss_type == 'IWR':
                    loss = self.weighted_bc_loss(distribution, act, act_overlaid, hg_dagger=False)
                elif self.loss_type == 'BT':
                    loss = self.bt_loss(distribution, act, act_agent, act_overlaid)
                elif self.loss_type == 'DPO':
                    loss = self.dpo_loss(distribution, init_distribution, act, act_agent, act_overlaid, beta=1.0)
                else:
                    raise NotImplementedError(f'Loss {self.loss_type} is not supported.')

            # Calculate gradients from policy loss
            loss.backward()

            # Update recurrent and policy parameters
            self.cade.gru_optimizer.step()
            self.cade.actor_optimizer.step()

        logger.info(f'Retraining for episode {self.ep_num} of loss {self.loss_type} for {epoch} epochs is done.')

        self.save_checkpoint()

    def calc_reward_estimator_loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        act_agent: torch.Tensor,
        act_overlaid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bradley-Terry preference loss of reward estimator in CADE, on human corrected data samples.

        Args:
            obs: Observations, [num_samples, obs_dim]
            act: Actually executed action, including both agent and human actions, [num_samples, num_branches]
            act_agent: Agent action, including both intended and executed actions, [num_samples, num_branches]
            act_overlaid: Mask of human intervention, [num_samples]

        Returns:
            loss: Calculated loss.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=act.device)

        # Filter human intervened samples
        filtered = self.filter([],  # no policy distribution needed for reward loss
                               act,
                               act_agent,
                               act_overlaid)
        if filtered is None:
            return total_loss

        _, reward_pred_actual = self.cade.forward_actor_reward(obs, act)
        _, reward_pred_intended = self.cade.forward_actor_reward(obs, act_agent)

        *_, final_mask = filtered

        # Only consider entries where human intervened and action differs from agent's original action
        filtered_reward_pred_actual = reward_pred_actual[0][final_mask]
        filtered_reward_pred_intended = reward_pred_intended[0][final_mask]

        # Bradley-Terry loss
        loss = -torch.log(torch.sigmoid(filtered_reward_pred_actual - filtered_reward_pred_intended + 1e-8)).mean()

        return loss

    def calc_reward_adv(
        self,
        reward_pred: torch.Tensor,
        baseline_return: float,  # TODO not used
    ) -> torch.Tensor:
        """
        Single reward as reward advantage, with normalization.
        TODO can be replaced by more advanced reward-to-go or GAE.

        Args:
            reward_pred:
            baseline_return:

        Returns:

        """
        mean, std = torch.mean(reward_pred), torch.std(reward_pred)
        adv = (reward_pred - mean) / (std + 1e-8)
        return adv

    def filter(
        self,
        policy_distributions: List[Categorical],
        act: torch.Tensor,
        act_agent: torch.Tensor,
        act_overlaid: torch.Tensor,
    ) -> Optional[Tuple[List[Categorical], torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Filter out data where human intervention occurred AND action differs from agent's original action.

        Args:
            policy_distributions: List of policy branches
            act: Actually executed action, including both agent and human actions, [num_samples, num_branches]
            act_agent: Agent action, including both intended and executed actions, [num_samples, num_branches]
            act_overlaid: Mask of human intervention, [num_samples]

        Returns:
            None if no human intervention exists or no corrective human action exists, else return:
            filtered_policy_distributions: List of policy branches with filtered batch
            filtered_act: [num_filtered, num_branches]
            filtered_act_agent: [num_filtered, num_branches]
            final_mask: [num_filtered]
        """

        assert act.shape == act_agent.shape
        assert act.shape[0] == act_overlaid.shape[0]

        # Only consider entries where human intervened
        human_mask = act_overlaid.bool()  # [batch_size]
        if human_mask.sum() == 0:  # No human intervention exists
            return None

        act_human = act[human_mask]  # [batch_size, num_branches]
        act_agent_human = act_agent[human_mask]  # [batch_size, num_branches]

        # Find samples where human and agent actions differ (any branch)
        unequal_mask = (act_human != act_agent_human).any(dim=1)  # [batch_size]
        if unequal_mask.sum() == 0:
            return None

        # Final filtered indices
        final_mask = torch.zeros_like(act_overlaid, dtype=torch.bool)
        indices = torch.nonzero(human_mask).squeeze(1)
        final_indices = indices[unequal_mask]
        final_mask[final_indices] = True  # [batch_size]

        logger.info(
            f'Total steps: {act_overlaid.size(0)}, '
            f'human steps: {human_mask.sum()}, '
            f'human corrective steps: {final_mask.sum()}.')

        # Apply mask to all data
        filtered_act = act[final_mask].long()
        filtered_act_agent = act_agent[final_mask].long()

        # Apply to each branch in policy_distributions
        filtered_policy_distributions = [
            Categorical(logits=dist.logits[final_mask]) for dist in policy_distributions
        ]

        return filtered_policy_distributions, filtered_act, filtered_act_agent, final_mask

    def update_sdm(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Update Semantic Dynamics Model in CADE.
        Args:
            obs: Current observation.
            act: Action taken at current observation.
            next_obs: The next observation after taken act.

        Returns:

        """
        # Concat obs and act
        obs_act = torch.cat((obs, act), dim=-1)

        # Forward pass
        delta = self.cade.sdm(obs_act)

        # Calculate loss
        # loss = self._actor_critic.sdm.loss_l1(obs, act, delta, next_obs)
        loss = self.cade.sdm.loss_iou(obs, act, delta, next_obs)

        # Backpropagate
        self.cade.sdm.backprop(loss)

    def policy_loss_by_hitl_reward(
        self,
        policy_distributions: List[Categorical],
        init_policy_distributions: List[Categorical],
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        """
        FOCOPS policy loss.
        (https://proceedings.neurips.cc/paper_files/paper/2020/file/af5d5ef24881f3c3049a7b9bfe74d58b-Paper.pdf)
        Currently only support multi-discrete action space.

        Args:
            policy_distributions: List of policy distributions for mutli-discrete action space after update.
            init_policy_distributions: List of policy distributions before update.
            act: Actually executed action, including both agent and human actions, [num_samples, num_branches].
            logp: Log prob of agent actions, [number_samples,], summed over all action branches.
            adv: Reward advantage as calculated in calc_reward_adv(), [num_samples].

        Returns:
            loss: Calculated policy loss.
        """
        logp_ = self.cade.actor.log_prob(act)  # log prob of actually executed actions
        ratio = torch.exp(logp_ - logp)
        kl = kld_multi_categorical(policy_distributions, init_policy_distributions)

        # FOCOPS loss
        loss = ((kl - (1 / self.cfgs.algo_cfgs.focops_lam) * ratio * adv) *
                (kl.detach() <= self.cfgs.algo_cfgs.focops_eta).type(torch.float32))

        loss = loss.mean()

        return loss

    def weighted_bc_loss(
        self,
        policy_distributions: List[Categorical],
        act: torch.Tensor,
        act_overlaid: torch.Tensor,
        hg_dagger: bool = False,
    ) -> torch.Tensor:
        """
        Loss of weighted Behavior Cloning.
        Can be adapted to Intervention Weighted Regression (IWR, https://arxiv.org/pdf/2012.06733),
        or HG-DAgger (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793698)

        Args:
            policy_distributions: List of policy distributions for mutli-discrete action space after update.
            act: Actually executed action, including both agent and human actions, [num_samples, num_branches].
            act_overlaid: Mask of human intervention, [num_samples].
            hg_dagger: Whether only considers the loss where human has intervened.

        Returns:
            loss: Calculated policy loss.
        """
        # Behavior Cloning loss
        total_loss = 0.
        act_branches: int = len(policy_distributions)
        act = act.long()  # indices

        # Iterate over each action branch for multi-discrete action space
        for i, dist in enumerate(policy_distributions):
            logits = dist.logits  # [batch, action num in a single branch]
            loss = F.cross_entropy(logits, act[:, i], reduction='none')
            total_loss += loss

        # Average loss over all branches
        avg_loss = total_loss / act_branches  # [batch,]

        # Apply different weights to different samples (Intervention Weighted Regression)
        # Human-in-the-Loop Imitation Learning using Remote Teleoperation (https://arxiv.org/pdf/2012.06733)
        num_human_actions = act_overlaid.sum().item()
        num_policy_actions = act_overlaid.numel() - num_human_actions
        if num_human_actions == 0:
            # Uniform weights if no human actions
            weight_ratio = 1.
        else:
            # More proportion of human actions, less emphasis on human actions
            # Less proportion of human actions, more emphasis on human actions
            weight_ratio = num_policy_actions / num_human_actions
        weights = torch.where(act_overlaid == 1, weight_ratio, 0 if hg_dagger else 1)
        # print(f'{weights=}')

        # Weighted loss across the sample dimension
        avg_loss = avg_loss * weights

        return avg_loss.mean()

    def bt_loss(
        self,
        policy_distributions: List[Categorical],
        act: torch.Tensor,
        act_agent: torch.Tensor,
        act_overlaid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bradley-Terry preference loss (logp as reward in the preference model)
        (https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)

        Args:
            policy_distributions: List of policy distributions for mutli-discrete action space after update.
            act: Actually executed action, including both agent and human actions, [num_samples, num_branches].
            act_agent: Agent action, including both intended and executed actions, [num_samples, num_branches].
            act_overlaid: Mask of human intervention, [num_samples].

        Returns:
            loss: Calculated policy loss.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=act.device)
        act_branches = len(policy_distributions)

        # Filter human intervened samples
        filtered = self.filter(policy_distributions,
                               act,
                               act_agent,
                               act_overlaid)
        if filtered is None:
            return total_loss

        filtered_policy_distributions, filtered_act, filtered_act_agent, _ = filtered

        for i, dist in enumerate(filtered_policy_distributions):
            # Log-probs from current policy
            log_probs = F.log_softmax(dist.logits, dim=-1)  # [batch_size, num_actions_in_a_single_branch]

            # Get log probs of human and agent actions
            logp_human = log_probs.gather(1, filtered_act[:, i].unsqueeze(1)).squeeze(1)
            logp_agent = log_probs.gather(1, filtered_act_agent[:, i].unsqueeze(1)).squeeze(1)

            # Bradley-Terry preference loss
            preference_prob = torch.sigmoid(logp_human - logp_agent)
            branch_loss = -torch.log(preference_prob + 1e-8).mean()  # Avoid log(0)
            total_loss += branch_loss

        return total_loss / act_branches

    def dpo_loss(
        self,
        policy_distributions: List[Categorical],
        ref_policy_distributions: List[Categorical],
        act: torch.Tensor,
        act_agent: torch.Tensor,
        act_overlaid: torch.Tensor,
        beta: float = 1.,
    ) -> torch.Tensor:
        """Direct Preference Optimization (DPO) loss
        https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf

        Args:
            policy_distributions: List of policy distributions for mutli-discrete action space after update.
            ref_policy_distributions: List of policy distributions for mutli-discrete action space from reference model.
            act: Actually executed action, including both agent and human actions, [num_samples, num_branches].
            act_agent: Agent action, including both intended and executed actions, [num_samples, num_branches].
            act_overlaid: Mask of human intervention, [num_samples].
            beta: Scaling factor controlling divergence from the reference model.

        Returns:

        """
        assert beta > 0, f'Beta should be non-negative, given {beta}.'

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=act.device)
        act_branches = len(policy_distributions)

        # Filter human intervened samples
        filtered = self.filter(policy_distributions,
                               act,
                               act_agent,
                               act_overlaid)
        if filtered is None:
            return total_loss

        filtered_policy_distributions, filtered_act, filtered_act_agent, final_mask = filtered

        # Filter corresponding reference policy distributions
        filtered_ref_distributions = [
            Categorical(logits=dist.logits[final_mask])
            for dist in ref_policy_distributions
        ]

        for i in range(act_branches):
            # Get log probs of current and reference policies
            policy_log_probs = F.log_softmax(filtered_policy_distributions[i].logits, dim=-1)
            ref_log_probs = F.log_softmax(filtered_ref_distributions[i].logits, dim=-1)

            # Get log probs of human action and agent action under agent policy
            logp_human = policy_log_probs.gather(1, filtered_act[:, i].unsqueeze(1)).squeeze(1)
            logp_agent = policy_log_probs.gather(1, filtered_act_agent[:, i].unsqueeze(1)).squeeze(1)

            # Get log probs of human action and agent action under reference policy
            logp_ref_human = ref_log_probs.gather(1, filtered_act[:, i].unsqueeze(1)).squeeze(1)
            logp_ref_agent = ref_log_probs.gather(1, filtered_act_agent[:, i].unsqueeze(1)).squeeze(1)

            # Compute DPO preference term
            diff = beta * ((logp_human - logp_ref_human) - (logp_agent - logp_ref_agent))
            loss_branch = -F.logsigmoid(diff).mean()
            total_loss += loss_branch

        return total_loss / act_branches


def eval_multiple_models(model_dir: str) -> None:
    """
    Evaluate CADE models trained with different loss types.

    Args:
        model_dir: Directory storing checkpoints.

    Returns:
        None
    """
    assert os.path.exists(model_dir), f'{model_dir} does not exist.'

    loss_tuple = get_args(LossType)
    logger.info(f'{loss_tuple=}')

    for loss_type in loss_tuple:
        if loss_type == 'None':  # no HITL loss
            model_name: str = 'episode-000.pt'
        else:
            model_name: str = f'episode-000-loss-{loss_type}.pt'

        # Init and evaluate
        hitl_cade = HitlCade(
            env_id=env_id,
            model_dir=model_dir,
            model_name=model_name,
            eval_episodes=eval_episodes,
            deterministic=deterministic,
            save_path=save_path,
            save_buffer=save_buffer,
            difficulty=difficulty,
            retrain_epoch=retrain_epoch,  # NOT USED
            loss_type=loss_type,  # NOT USED
            save_ckpts=False,
            enable_hitl=False,
            enable_retrain=False,
        )

        logger.info(f'Start eval of loss {loss_type} ...')
        hitl_cade.evaluate()
        logger.info(f'Eval of loss {loss_type} finished.')


def extract_episode_id(filename: str, key: str = 'episode') -> int:
    """
    Extract integer episode id from string filename, used to sort files containing integer after the key string.

    Args:
        filename: Path to the file.
        key: Key string before the integer episode id.

    Returns:
        Extracted integer episode id, or -1 if not found.
    """
    base = os.path.basename(filename)
    pattern = rf"{key}[-_]?(\d+)"
    match = re.search(pattern, base)
    return int(match.group(1)) if match else -1


def integral_retrain(save_path: str, loss_type: LossType) -> None:
    """
    Integrally retrain the CADE models using episodes with human intervention data.
    For example, checkpoint N trained on episode N will serve as the start point of training on episode N+1, which
    results in checkpoint N+1.
    The initial CADE model is loaded from {model_dir}/{model_name}.

    Args:
        save_path: directory storing the evaluation episodes with human interventions.
        loss_type: hitl loss type.

    Returns:
        None
   """
    assert os.path.exists(save_path), f'Save path {save_path} does not exist.'
    assert loss_type in get_args(LossType), f'Loss type {loss_type} is not supported.'

    logger.info(f'Start integral retraining for loss {loss_type} ...')
    episodes_paths: List[str] = []
    for item in os.scandir(save_path):
        if not item.is_file():
            continue
        if 'hitlTrue' not in item.name:  # Retrained episode needs to have HITL
            continue
        if 'lossNone' not in item.name:  # Retrain for different loss types has to start with plain CADE policy
            continue
        ep_full_path: str = os.path.join(save_path, item.name)
        episodes_paths.append(ep_full_path)
    # print(f'{episodes_paths=}')

    sorted_episodes_paths: List[str] = sorted(episodes_paths, key=lambda path: extract_episode_id(path, 'episode'))
    logger.info(f'{sorted_episodes_paths=}')

    # Init HITL CADE
    hitl_cade = HitlCade(
        env_id=env_id,
        model_dir=model_dir,
        model_name=model_name,
        eval_episodes=eval_episodes,
        deterministic=deterministic,
        save_path=save_path,
        save_buffer=False,
        difficulty=difficulty,
        retrain_epoch=retrain_epoch,
        loss_type=loss_type,
        save_ckpts=True,  # Necessary
        enable_hitl=False,
        enable_retrain=True,
    )

    # Sequentially train CADE in integral manner for multiple episodes
    for ep_id, ep_path in enumerate(sorted_episodes_paths):
        hitl_cade.buffer = load_buffer_from_csv(filename=ep_path, buffer=hitl_cade.buffer)
        hitl_cade.ep_num = ep_id  # Will appear in the saved checkpoint's name
        hitl_cade.retrain(hitl_cade.buffer.get())

    hitl_cade.close()
    logger.info(f'Integral retraining of {len(sorted_episodes_paths)} episodes for loss {loss_type} is done.')


def eval_integral_retrained_ckpts(model_path: str, loss_type: LossType) -> None:
    """
    Evaluate CADE checkpoints that are integrally trained on several episodes.

    Args:
        model_path: directory storing the checkpoints.
        loss_type: hitl loss type.

    Returns:
        None
    """
    assert os.path.exists(model_path), f'{model_path} does not exist.'
    assert loss_type in get_args(LossType), f'Loss {loss_type} is not supported.'

    ckpt_paths: List[str] = []
    for item in os.scandir(os.path.join(model_path, 'torch_save')):
        if 'hitl-False' not in item.name:
            continue
        if f'loss-{loss_type}' not in item.name:
            continue
        ckpt_paths.append(os.path.join(model_path, item.name))
    # print(f'{ckpt_paths=}')

    sorted_ckpt_paths: List[str] = sorted(ckpt_paths, key=lambda path: extract_episode_id(path, 'episode'))
    logger.info(f'{sorted_ckpt_paths=}')

    for ckpt_id, ckpt_path in enumerate(sorted_ckpt_paths):
        model_name: str = os.path.basename(ckpt_path)
        logger.info(f'Start eval of checkpoint {model_name} for loss {loss_type} ...')

        # Init and evaluate
        hitl_cade = HitlCade(
            env_id=env_id,
            model_dir=model_dir,
            model_name=model_name,
            eval_episodes=eval_episodes,
            deterministic=deterministic,
            save_path=save_path,
            save_buffer=False,
            difficulty=difficulty,
            retrain_epoch=retrain_epoch,  # NOT USED
            loss_type=loss_type,  # NOT USED
            save_ckpts=False,
            enable_hitl=False,
            enable_retrain=False,
        )

        # Update stat file name with checkpoint info
        stat_file_name: str = f'{hitl_cade.env_id}_hitl{hitl_cade.enable_hitl}_seed{hitl_cade.seed}_difficulty{hitl_cade.difficulty}_ckpt{ckpt_id}_loss{loss_type}.csv'
        hitl_cade.stat_file_path = os.path.join(hitl_cade.save_path, stat_file_name)

        hitl_cade.evaluate()

        logger.info(f'Eval of checkpoint {ckpt_id} for loss {loss_type} is finished.')


def get_statistics(metrics_dir: str, ckpt_id: Optional[int] = None) -> Loss2RewStep:
    """
    Get episodic rewards of evaluation episodes of different loss types.

    Args:
        metrics_dir: directory storing statistic csv files of evaluation results.
        ckpt_id: A specific checkpoint's eval results are of interest.

    Returns:
        Dictionary of loss to list of episodic rewards.

    """
    assert os.path.exists(metrics_dir), f'Metrics dir {metrics_dir} does not exist.'

    stat_dict: Loss2RewStep = {}

    for item in os.scandir(metrics_dir):
        if (item.is_file() and
            'loss' in item.name and
            'episode' not in item.name
        ):
            if ckpt_id is not None and 'None' not in item.name and f'ckpt{ckpt_id}' not in item.name:
                continue

            file_path: str = os.path.join(metrics_dir, item.name)
            df = pd.read_csv(file_path)
            ep_rews = df['Episodic Rewards'].to_list()
            ep_steps = df['Episodic Steps'].to_list()

            loss_name = 'None'  # no HITL loss
            for loss in get_args(LossType):
                if loss in item.name:
                    loss_name = loss
                    break
            if loss_name == 'None':
                stat_dict['Baseline'] = ep_rews, ep_steps
            else:
                stat_dict[loss_name] = ep_rews, ep_steps

    return stat_dict


def plot_stat_loss_types(stat: Loss2RewStep, plot_ratio: bool = False) -> None:
    """
    Bar plot of episodic rewards during evaluation of different loss types.

    Args:
        stat: dictionary of loss type to list of episodic rewards.
        plot_ratio: whether to plot the episodic reward over episodic steps ratio, which is a measure of efficiency.

    Returns:
        None
    """
    # Plot parameters
    logger.info(stat.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    metric: str = ''
    for loss_name, (ep_rews, ep_steps) in stat.items():
        if plot_ratio:
            ratio = np.array(ep_rews) / np.array(ep_steps)
            ratio_mean, ratio_std = ratio.mean(), ratio.std()
            plt.bar(loss_name, ratio_mean, yerr=ratio_std, label=loss_name)
            metric = 'Episodic Reward Per Step'
        else:
            ep_rew_mean, ep_rew_std = np.array(ep_rews).mean(), np.array(ep_rews).std()
            plt.bar(loss_name, ep_rew_mean, yerr=ep_rew_std, label=loss_name)
            metric = 'Episodic Reward'

    ax.set_xlabel('HITL Losses')
    ax.set_ylabel(metric)
    ax.set_title(f'Comparison of {metric} for HITL Losses')

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_stat_ckpts(
    metrics_dir: str,
    loss_type: LossType,
    plot_ratio: bool = False,
) -> None:
    """
    Plot the evaluation results of checkpoints retrained with some loss.

    Args:
        metrics_dir: directory storing evaluation metrics.
        loss_type: hitl loss type.
        plot_ratio: whether to plot the episodic reward over episodic steps ratio, which is a measure of efficiency.

    Returns:
        None
    """
    assert os.path.exists(metrics_dir), f'{metrics_dir} does not exist.'
    assert loss_type in get_args(LossType), f'Loss {loss_type} is not supported.'

    ckpt_to_ep_rews_steps: Dict[int, Tuple[List[float], List[int]]] = {}

    for item in os.scandir(metrics_dir):
        if loss_type not in item.name:
            continue
        if 'ckpt' not in item.name:
            continue

        ckpt_eval_path: str = os.path.join(metrics_dir, item.name)
        ckpt_id: int = extract_episode_id(filename=ckpt_eval_path, key='ckpt')
        df = pd.read_csv(ckpt_eval_path)
        ep_rews = df['Episodic Rewards'].to_list()
        ep_steps = df['Episodic Steps'].to_list()
        ckpt_to_ep_rews_steps[ckpt_id] = (ep_rews, ep_steps)

    ckpt_to_ep_rews_steps = dict(sorted(ckpt_to_ep_rews_steps.items()))
    logger.info(f'{ckpt_to_ep_rews_steps=}')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot baseline (no hitl retraining)
    baseline_name: str = 'medium_hitlFalse_seed000_difficulty1_lossNone.csv'
    baseline_path: str = os.path.join(metrics_dir, baseline_name)
    assert os.path.exists(baseline_path), f'{baseline_path} does not exist!'
    df = pd.read_csv(baseline_path)
    ep_rews = df['Episodic Rewards'].to_list()
    ep_steps = df['Episodic Steps'].to_list()
    rew_step_ratio = np.array(ep_rews) / np.array(ep_steps)
    if plot_ratio:
        plt.plot(rew_step_ratio, label='Baseline', color='k', linestyle='-.')
    else:
        plt.plot(ep_rews, label='Baseline', color='k', linestyle='-.')

    # Plot HITL retrained checkpoints
    for ckpt_id, (ep_rews, ep_steps) in ckpt_to_ep_rews_steps.items():
        rew_step_ratio = np.array(ep_rews) / np.array(ep_steps)
        if plot_ratio:
            plt.plot(rew_step_ratio, label=f'Checkpoint {ckpt_id}')
        else:
            plt.plot(ep_rews, label=f'Checkpoint {ckpt_id}')

    ax.legend()
    ax.set_xticks(ticks=range(len(ckpt_to_ep_rews_steps)))
    ax.set_xlabel('Episode ID')
    ax.set_ylabel("Episodic Rewards")
    ax.set_title(f"Comparison of Episodic Rewards of Integrally Retrained Checkpoints for {loss_type} Loss")

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Configurable parameters
    env_id: str = 'medium'  # Choose from {easy, medium, hard}
    model_dir: str = './runs/FOCOPS_CACD-{medium}/seed-000-2025-03-01-15-06-43'  # Base CADE model
    save_path: str = 'evaluations/riverine'  # Loading and saving path of all files
    save_buffer: bool = False  # Whether save per-step data into file
    model_name: str = 'epoch-350.pt'  # Start point of CADE model
    eval_episodes: int = 5  # Evaluation episodes number
    deterministic: bool = True  # Determinism of the actor policy in CADE
    difficulty: int = 1  # [0, 2], different difficulty levels of env
    retrain_epoch: int = 5  # Number of epochs to retrain CADE
    loss_type: LossType = 'DPO'  # Choose from {'None', 'IWR', 'HG-DAgger', 'BT', 'DPO', 'Indirect'}
    enable_hitl: bool = False  # Will retrain policy if True
    enable_retrain: bool = True  # Whether retrain CADE if hitl is enabled
    save_ckpts: bool = True  # Whether save checkpoints of retrained CADE

    evaluate: bool = False  # Whether evaluate the trained policy or test the retrain function
    evaluate_single: bool = False  # Whether evaluate single CADE model or multiple CADE models
    retrain_single: bool = False  # Whether retrain from single episodes or multiple episodes (integral retrain)
    plot_comp: bool = True  # Whether plot statistic comparisons

    if evaluate:
        if evaluate_single:
            # Evaluate with HITL
            hitl_cade = HitlCade(
                env_id=env_id,
                model_dir=model_dir,
                model_name=model_name,
                eval_episodes=eval_episodes,
                deterministic=deterministic,
                save_path=save_path,
                save_buffer=save_buffer,
                difficulty=difficulty,
                retrain_epoch=retrain_epoch,
                loss_type=loss_type,
                save_ckpts=save_ckpts,
                enable_hitl=enable_hitl,
                enable_retrain=enable_retrain,
            )
            hitl_cade.evaluate()
        else:
            # Evaluate multiple HITL models
            # eval_multiple_models(model_dir=model_dir)

            # Evaluate integral retrained HITL checkpoints
            eval_integral_retrained_ckpts(model_path=model_dir, loss_type=loss_type)
    else:
        if retrain_single:
            # Test of HITL retraining
            csv_filename: str = './evaluations/riverine/medium_hitlTrue_seed000_difficulty1_episode0.csv'
            hitl_cade = HitlCade(
                env_id=env_id,
                model_dir=model_dir,
                model_name=model_name,
                eval_episodes=eval_episodes,
                deterministic=deterministic,
                save_path=save_path,
                save_buffer=save_buffer,
                difficulty=difficulty,
                retrain_epoch=retrain_epoch,
                loss_type=loss_type,
                save_ckpts=save_ckpts,
                enable_hitl=enable_hitl,
                enable_retrain=enable_retrain,
            )
            hitl_cade.buffer = load_buffer_from_csv(filename=csv_filename, buffer=hitl_cade.buffer)
            hitl_cade.retrain(hitl_cade.buffer.get(), epoch=retrain_epoch)
            hitl_cade.close()
        else:
            pass
            # integral_retrain(save_path=save_path, loss_type=loss_type)

        if plot_comp:
            # Plot statistics of the last integrally trained checkpoint as bar plot
            stat = get_statistics(metrics_dir=save_path, ckpt_id=eval_episodes - 1)
            plot_stat_loss_types(stat)

            # Plot statistics as line plot
            # plot_stat_ckpts(metrics_dir=save_path, loss_type=loss_type, plot_ratio=True)
