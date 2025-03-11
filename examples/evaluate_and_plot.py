import os
import csv
import glob

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

import omnisafe

from gymnasium.envs.toy_text.cliffcircular import CliffCircularEnv
from omnisafe.envs.riverine_env import RiverineEnv


def evaluate_model(
    log_dir: str,
    render_mode: str,
    difficulty: int,
    save_path: Optional[str],
    eval_episodes: int = 30,
    enable_safety_layer: bool = False,
):
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(log_dir, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            # ckpt_number: str = '800' if env_name == 'cliffcircular' else '400'
            ckpt_number: str = '1500' if env_name == 'cliffcircular' else '400'

            if ckpt_number not in item.name:
                continue
            else:
                print(f'Start evaluating {item.name} ...')

            evaluator.load_saved(
                save_dir=log_dir,
                model_name=item.name,
                render_mode=render_mode,
                camera_name='track',
                width=5,
                height=5,
                difficulty=difficulty,
                enable_safety_layer=enable_safety_layer,
            )

            # evaluator.render(num_episodes=1)
            episodic_rewards, episodic_costs, episode_lengths = evaluator.evaluate(num_episodes=eval_episodes)

            if save_path is not None:
                with open(save_path, 'w', newline="") as file:
                    writer = csv.writer(file)

                    # Write the header
                    writer.writerow(["Episodic Rewards", "Episodic Costs", "Episode Lengths"])

                    # Write the data row by row
                    for reward, cost, length in zip(episodic_rewards, episodic_costs, episode_lengths):
                        writer.writerow([reward, cost, length])

    scan_dir.close()


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
    env_name: str = 'cliffcircular'  # ['cliffcircular', 'riverine']

    # Advantage name to the trained model dir
    if env_name == 'cliffcircular':
        adv_dir_dict: Dict[str, str] = {
            'plain': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-22-32-09',
            'gae': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-21-55-13',
            'gae-rtg': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-22-07-04',
            'vtrace': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-22-20-05',
            'subm': './runs/FOCOPS_CACD-{CliffCircular-v1}/seed-000-2024-10-23-21-40-27',
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
        }

    render_mode: str = 'rgb_array'  # ['rgb_array', 'human']
    # render_mode: str = 'human'  # ['rgb_array', 'human']
    difficulty: int = 1  # [0, 1, 2]
    eval_episodes: int = 30
    eval_dir: str = f'evaluations/{env_name}'

    evaluate: bool = True  # Testing the trained models if True, plot the result otherwise
    evaluate_cade: bool = True  # Evaluate CADE if True, otherwise evaluate advantage
    enable_safety_layer: bool = True

    if evaluate:
        if evaluate_cade:
            for cade, log_dir_list in cade_dir_dict.items():
                for log_dir in log_dir_list:
                    seed: str = log_dir.split('/')[-1].split('-')[1]
                    results_filename: str = f'{env_name}_{cade}_seed{seed}_difficulty{difficulty}.csv'
                    save_path: Optional[str] = os.path.join(eval_dir, results_filename)

                    print(f'Evaluating {cade} on difficulty level {difficulty} in {env_name} env ...')
                    evaluate_model(
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
                evaluate_model(
                    log_dir=log_dir,
                    render_mode=render_mode,
                    difficulty=difficulty,
                    save_path=save_path,
                    eval_episodes=eval_episodes,
                )

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
