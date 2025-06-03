import os
import csv
import glob
import json
import time
import numpy as np
import pandas as pd
import torch
from typing import Union, Optional, List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium.spaces import Discrete, MultiDiscrete

import omnisafe
from omnisafe.utils.config import Config
from omnisafe.envs.core import make, CMDP
from omnisafe.typing import OmnisafeSpace
from omnisafe.models.actor_critic import ConstraintActorDynamicsEstimator
from cliffcircular.cliffcircular import CliffCircularEnv
from omnisafe.envs.riverine_env import RiverineEnv


def evaluate_model_recurrent(
    log_dir: str,
    render_mode: str,
    difficulty: int,
    save_path: Optional[str],
    eval_episodes: int = 30,
    enable_safety_layer: bool = False,
):
    # Load config
    cfg_path = os.path.join(log_dir, 'config.json')
    try:
        with open(cfg_path, encoding='utf-8') as file:
            kwargs = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'The config file is not found in the save directory {log_dir}.',
        ) from error
    cfgs = Config.dict2config(kwargs)
    print('Config is loaded.')

    # Init environment
    env_id: str = log_dir[log_dir.find('{') + 1: log_dir.find('}')]
    print(f'{env_id=}')
    env_kwarg: Dict[str, Any] = {
        'env_id': env_id,
        'render_mode': render_mode,
    }

    assert 0 <= difficulty <= 2

    if 'cliff' in env_id.lower():  # cliffcircular env
        env_kwarg['extra_cliff_num'] = difficulty
    else:  # riverine env
        if difficulty == 0:
            env_kwarg['env_id'] = 'easy'
        elif difficulty == 1:
            env_kwarg['env_id'] = 'medium'
        elif difficulty == 2:
            env_kwarg['env_id'] = 'hard'

    env: CMDP = make(**env_kwarg)
    obs_space: OmnisafeSpace = env.observation_space
    act_space: OmnisafeSpace = env.action_space
    print(f'Env is inited.')

    # Load model
    assert os.path.exists(log_dir), f'Model dir {log_dir} does not exist!'

    model_name: str = 'epoch-1500.pt' if 'CliffCircular' in env_id else 'epoch-350.pt'

    model_path: str = os.path.join(log_dir, 'torch_save', model_name)
    if not os.path.exists(model_path):  # Deal with corner case
        model_name = 'epoch-800.pt'
        model_path = os.path.join(log_dir, 'torch_save', model_name)

    print(f'{model_name=}')
    assert os.path.exists(model_path), f'Model path {model_path} does not exist!'

    model_params = torch.load(model_path, map_location='cpu')

    is_value_critic: bool = True  # TODO this depends on reward advantage type
    cade: ConstraintActorDynamicsEstimator = ConstraintActorDynamicsEstimator(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=cfgs.model_cfgs,
        epochs=1,  # Not used, for linear lr decay
        is_value_critic=is_value_critic,
    )

    # for name, module in cade.named_modules():
    #     print(f'{name=} {module=}')
    #     print('-'*40)

    cade.load_state_dict(model_params['actor_critic'])
    print('Model is loaded.')

    # Set nominal (default) action
    if isinstance(act_space, Discrete):
        nominal_action = torch.tensor([[0]])  # CliffCircular
    elif isinstance(act_space, MultiDiscrete):
        nominal_action = torch.tensor([[1] * act_space.nvec.shape[0]])  # SRE
    else:
        print(f'Nominal action for {type(act_space)} is not supported.')
        raise NotImplementedError
    print(f'Nominal (no_op) action in this env: {nominal_action}')

    # Start evaluation
    print('Start evaluation ...')
    obs, info = env.reset()

    latent = None
    cur_episodes: int = 0
    cur_steps: int = 0
    ep_rew_list: List[float] = []
    ep_cost_list: List[float] = []
    ep_steps_list: List[int] = []
    ep_rew: float = 0.
    ep_cost: float = 0.
    last_action = nominal_action.clone()

    while cur_episodes < eval_episodes:

        # Reshape obs
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 3:
            obs = obs.squeeze(0)
        # print(f'{obs.shape=}')

        # Step CAD
        act, logp, act_overlaid, reward_pred, cost_pred, latent = cade.step(
            obs=obs,
            last_act=last_action,
            lagrangian_multiplier=1.0,  # equally weigh reward and cost
            latent=latent,
            deterministic=False,
            enable_safety_layer=enable_safety_layer,
            safety_layer_use_reward=False,
        )

        # Step SDM and update cost_pred
        next_obs_pred = cade.sdm.predict(torch.cat([obs, act], dim=-1), round_to_int=True)
        with torch.no_grad():
            cost_pred = cade.cost_critic(next_obs_pred)[0]  # only use the first cost critic

        # Step environment
        # print(f'{act=}')
        next_obs, reward, cost, terminated, truncated, info = env.step(act[0])
        last_action.copy_(act)
        ep_rew += reward.item()
        ep_cost += cost.item()
        cur_steps += 1

        # print(f'Pred reward:   {reward_pred.item():.2f},   Pred cost:   {cost_pred.item():.2f} \n'
        #       f'Actual reward: {reward.item():.2f},   Actual cost: {cost.item():.2f} \n')

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()
            latent = None
            last_action.copy_(nominal_action)
            cur_episodes += 1

            print(f'Episode {cur_episodes} finished with reward {ep_rew:.2f} and cost {ep_cost:.2f}, {cur_steps} steps.')

            ep_rew_list.append(ep_rew)
            ep_cost_list.append(ep_cost)
            ep_steps_list.append(cur_steps)
            ep_rew = 0.
            ep_cost = 0.
            cur_steps = 0

    # Save to file
    if save_path is not None:
        with open(save_path, 'w', newline="") as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(["Episodic Rewards", "Episodic Costs", "Episode Lengths"])

            # Write the data row by row
            for reward, cost, length in zip(ep_rew_list, ep_cost_list, ep_steps_list):
                writer.writerow([reward, cost, length])

    # mean_ep_reward: float = ep_reward_sum / cur_episodes
    # mean_ep_cost: float = ep_cost_sum / cur_episodes
    # print(f'{cur_episodes} episodes finished, mean ep reward: {mean_ep_reward:.2f}, mean ep cost: {mean_ep_cost:.2f}')

    env.close()
    time.sleep(1)

    return ep_rew_list, ep_cost_list, ep_steps_list


def read_csv_files(file_pattern: str) -> Tuple[List[List[float]], List[str]]:
    """
    Reads multiple CSV files matching the given file pattern and extracts relevant data.

    Args:
    file_pattern (str): Pattern to match the files (e.g., '*0*.csv' or '*[01]*.csv')

    Returns:
    list of lists: Returns a list where each inner list contains episodic rewards from one file.
    """
    file_paths = glob.glob(file_pattern, root_dir=eval_dir)  # Find all files matching the pattern
    all_rewards = []
    all_advs = []

    # Loop through files and extract the info in order
    for adv, _ in adv_dir_dict.items():
        for file in file_paths:
            if adv not in file:
                continue
            if adv == 'gae' and 'rtg' in file:
                continue
            file_abs_path = os.path.join(os.path.dirname(__file__), eval_dir, file)
            df = pd.read_csv(file_abs_path)
            all_rewards.append(df['Episodic Rewards'].tolist())
            all_advs.append(adv)

    # print(f'{all_advs=}')

    return all_rewards, all_advs


def get_cade_statistics(
    env_name: str,
    difficulty: int,
):
    dir_name: str = f'evaluations/{env_name}'
    assert os.path.exists(dir_name)

    mgae_rewards = []
    mgae_costs = []
    lagrange_rewards = []
    lagrange_costs = []
    safety_rewards = []
    safety_costs = []

    scan_dir = os.scandir(dir_name)
    for item in scan_dir:
        if item.is_file():
            if f'difficulty{difficulty}' in item.name:
                if 'mgae' in item.name:
                    df = pd.read_csv(os.path.join(dir_name, item.name))
                    mgae_rewards += df['Episodic Rewards'].to_list()
                    mgae_costs += df['Episodic Costs'].to_list()
                    print(f'{item.name=}')
                elif 'lagrange' in item.name:
                    df = pd.read_csv(os.path.join(dir_name, item.name))
                    lagrange_rewards += df['Episodic Rewards'].to_list()
                    lagrange_costs += df['Episodic Costs'].to_list()
                    print(f'{item.name=}')
                elif 'safety' in item.name:
                    df = pd.read_csv(os.path.join(dir_name, item.name))
                    safety_rewards += df['Episodic Rewards'].to_list()
                    safety_costs += df['Episodic Costs'].to_list()
                    print(f'{item.name=}')
                else:
                    pass

    mgae_rew_mean, mgae_cost_mean = np.mean(mgae_rewards), np.mean(mgae_costs)
    mgae_rew_std, mgae_cost_std = np.std(mgae_rewards), np.std(mgae_costs)
    lagrange_rew_mean, lagrange_cost_mean = np.mean(lagrange_rewards), np.mean(lagrange_costs)
    lagrange_rew_std, lagrange_cost_std = np.std(lagrange_rewards), np.std(lagrange_costs)
    safety_rew_mean, safety_cost_mean = np.mean(safety_rewards), np.mean(safety_costs)
    safety_rew_std, safety_cost_std = np.std(safety_rewards), np.std(safety_costs)

    print(f'MGAE reward: {mgae_rew_mean:.1f} +- {mgae_rew_std:.1f} \n'
          f'MGAE cost: {mgae_cost_mean:.1f} +- {mgae_cost_std:.1f} \n'
          f'Lagrange reward: {lagrange_rew_mean:.1f} +- {lagrange_rew_std:.1f} \n'
          f'Lagrange cost: {lagrange_cost_mean:.1f} +- {lagrange_cost_std:.1f} \n'
          f'Safety reward: {safety_rew_mean:.1f} +- {safety_rew_std:.1f} \n'
          f'Safety cost: {safety_cost_mean:.1f} +- {safety_cost_std:.1f} \n')


def violin_plot(
    env_difficulty_level: int,
    fig_save_path: Optional[str] = None,
):
    file_pattern: str = f'*{env_difficulty_level}*.csv'
    rewards_data, adv_labels = read_csv_files(file_pattern)

    # Adjust names
    adv_labels[adv_labels.index('subm')] = 'MGAE'
    adv_labels[adv_labels.index('plain')] = 'TD'
    adv_labels[adv_labels.index('gae')] = 'GAE'
    adv_labels[adv_labels.index('gae-rtg')] = 'GAE-RTG'
    adv_labels[adv_labels.index('vtrace')] = 'V-trace'

    # print(f'{rewards_data=}')
    # print(f'{file_paths=}')
    # print(f'{adv_labels=}')

    # Create a violin plot using Seaborn
    sns.violinplot(data=rewards_data)

    # Set plt style
    plt.style.use("seaborn-v0_8-darkgrid")
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 15,
            }
    font_ticks = {'family': 'STIXGeneral',
                  'weight': 'bold',
                  'size': 12}
    font_title = {'family': 'STIXGeneral',
                  'weight': 'bold',
                  'size': 18}

    # Set labels for the x-axis (one for each file or agent)
    plt.xticks(ticks=range(len(adv_labels)), labels=adv_labels, fontdict=font_ticks)

    if difficulty == 0:
        plt.title(f'{env_name}-Easy', fontdict=font_title)
    else:
        plt.title(f'{env_name}-Hard', fontdict=font_title)
    plt.ylabel("Episodic Rewards", fontdict=font)
    plt.xlabel("Reward Advantage Estimation Method", fontdict=font)
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path, dpi=250)

    plt.show()


if __name__ == '__main__':
    env_name: str = 'riverine'  # ['cliffcircular', 'riverine']

    # Advantage name to the trained model dir
    if env_name == 'cliffcircular':
        adv_dir_dict: Dict[str, str] = {
            'plain': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-22-32-09',
            'gae': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-21-55-13',
            'gae-rtg': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-22-07-04',
            'vtrace': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-22-20-05',
            'subm': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-21-40-27',
            'REINFORCE': './runs/FOCOPS_CADE-{CliffCircular-v1}/seed-000-2025-05-09-21-58-48',
        }
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
        adv_dir_dict: Dict[str, str] = {
            'plain': './runs/FOCOPS_CACD-{medium}/seed-000-2024-11-01-22-01-02',
            'gae': './runs/FOCOPS_CACD-{medium}/seed-000-2024-11-03-13-35-00',
            'gae-rtg': './runs/FOCOPS_CACD-{medium}/seed-000-2024-11-03-12-38-53',
            'vtrace': './runs/FOCOPS_CACD-{medium}/seed-000-2024-11-01-23-38-10',
            'subm': './runs/FOCOPS_CACD-{medium}/seed-000-2024-10-31-16-12-21',
            'REINFORCE': './runs/FOCOPS_CADE-{medium}/seed-000-2025-05-10-14-04-28',
        }

    render_mode: str = 'rgb_array'  # ['rgb_array', 'human']
    # render_mode: str = 'human'  # ['rgb_array', 'human']
    difficulty: int = 0  # [0, 1, 2]
    eval_episodes: int = 1
    # eval_dir: str = f'evaluations/{env_name}'
    eval_dir: str = f'evaluations/{env_name}/rew_adv_comp'

    evaluate: bool = True  # Testing the trained models if True, plot the result otherwise
    evaluate_cade: bool = False  # Evaluate CADE if True, otherwise evaluate advantage
    enable_safety_layer: bool = False

    if evaluate:
        if evaluate_cade:
            for cade, log_dir_list in cade_dir_dict.items():
                for log_dir in log_dir_list:
                    seed: str = log_dir.split('/')[-1].split('-')[1]
                    results_filename: str = f'{env_name}_{cade}_seed{seed}_difficulty{difficulty}.csv'
                    save_path: Optional[str] = os.path.join(eval_dir, results_filename)

                    print(f'Evaluating {cade} on difficulty level {difficulty} in {env_name} env ...')
                    evaluate_model_recurrent(
                        log_dir=log_dir,
                        render_mode=render_mode,
                        difficulty=difficulty,
                        save_path=save_path,
                        eval_episodes=eval_episodes,
                        enable_safety_layer=enable_safety_layer,
                    )
        else:
            for adv, log_dir in adv_dir_dict.items():
                results_filename: str = f'{env_name}_{adv}_difficulty{difficulty}.csv'
                save_path: Optional[str] = os.path.join(eval_dir, results_filename)

                print(f'Evaluating {adv} on difficulty level {difficulty} in {env_name} env ...')
                evaluate_model_recurrent(
                    log_dir=log_dir,
                    render_mode=render_mode,
                    difficulty=difficulty,
                    save_path=save_path,
                    eval_episodes=eval_episodes,
                    enable_safety_layer=enable_safety_layer,
                )
                exit(0)

        print(f'All evaluations are finished.')

    else:  # plot or get statistics
        if evaluate_cade:
            get_cade_statistics(env_name=env_name, difficulty=difficulty)
        else:
            fig_save_path: str = f'{env_name}_adv_comp_difficulty{difficulty}.png'
            fig_save_path_abs: str = os.path.join(os.path.dirname(__file__), eval_dir, fig_save_path)

            violin_plot(
                env_difficulty_level=difficulty,
                fig_save_path=fig_save_path_abs,
            )
