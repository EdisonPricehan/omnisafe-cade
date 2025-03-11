import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import List
import os
import numpy as np


def plot(
    exp_file_list: List[str],
    env_name: str,
    algo_name: str = 'FOCOPS_CACD',
):
    abs_file_list: List[str] = get_abs_paths(
        file_list=exp_file_list,
        env_name=env_name,
        algo_name=algo_name,
    )

    mgae_ep_rew = []
    mgae_ep_cost = []
    lagrange_ep_rew = []
    lagrange_ep_cost = []
    safety_layer_ep_rew = []
    safety_layer_ep_cost = []

    for file in abs_file_list:
        config = get_config_json(file)
        data = get_progress_data(file)

        if config['algo_cfgs']['use_lagrangian']:
            lagrange_ep_rew.append(data['Metrics/EpRet'])
            lagrange_ep_cost.append(data['Metrics/EpCost'])
        else:
            if config['algo_cfgs']['use_safety_layer']:
                safety_layer_ep_rew.append(data['Metrics/EpRet'])
                safety_layer_ep_cost.append(data['Metrics/EpCost'])
            else:
                mgae_ep_rew.append(data['Metrics/EpRet'])
                mgae_ep_cost.append(data['Metrics/EpCost'])

    steps = get_progress_data(abs_file_list[0])['TotalEnvSteps']

    mgae_ep_rew = np.array(mgae_ep_rew)
    mgae_ep_cost = np.array(mgae_ep_cost)
    lagrange_ep_rew = np.array(lagrange_ep_rew)
    lagrange_ep_cost = np.array(lagrange_ep_cost)
    safety_layer_ep_rew = np.array(safety_layer_ep_rew)
    safety_layer_ep_cost = np.array(safety_layer_ep_cost)

    mgae_rew_mean, mgae_cost_mean = np.mean(mgae_ep_rew, axis=0), np.mean(mgae_ep_cost, axis=0)
    mgae_rew_std, mgae_cost_std = np.std(mgae_ep_rew, axis=0), np.std(mgae_ep_cost, axis=0)
    lagrange_rew_mean, lagrange_cost_mean = np.mean(lagrange_ep_rew, axis=0), np.mean(lagrange_ep_cost, axis=0)
    lagrange_rew_std, lagrange_cost_std = np.std(lagrange_ep_rew, axis=0), np.std(lagrange_ep_cost, axis=0)
    safety_layer_rew_mean, safety_layer_cost_mean = np.mean(safety_layer_ep_rew, axis=0), np.mean(safety_layer_ep_cost, axis=0)
    safety_layer_rew_std, safety_layer_cost_std = np.std(safety_layer_ep_rew, axis=0), np.std(safety_layer_ep_cost, axis=0)

    mgae_rew_mean, mgae_cost_mean = moving_average(mgae_rew_mean), moving_average(mgae_cost_mean)
    mgae_rew_std, mgae_cost_std = moving_average(mgae_rew_std), moving_average(mgae_cost_std)
    lagrange_rew_mean, lagrange_cost_mean = moving_average(lagrange_rew_mean), moving_average(lagrange_cost_mean)
    lagrange_rew_std, lagrange_cost_std = moving_average(lagrange_rew_std), moving_average(lagrange_cost_std)
    safety_layer_rew_mean, safety_layer_cost_mean = moving_average(safety_layer_rew_mean), moving_average(safety_layer_cost_mean)
    safety_layer_rew_std, safety_layer_cost_std = moving_average(safety_layer_rew_std), moving_average(safety_layer_cost_std)

    plt.style.use("seaborn-v0_8-darkgrid")
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 15,
            }
    font_title = {'family': 'STIXGeneral',
                  'weight': 'bold',
                  'size': 20,
                  }
    alpha = 0.3

    # Create a figure for the plot
    fig_rew = plt.figure(figsize=(10, 6))
    plt.plot(steps, mgae_rew_mean, label='MGAE')
    plt.fill_between(steps, mgae_rew_mean - mgae_rew_std, mgae_rew_mean + mgae_rew_std, alpha=alpha, label='')

    plt.plot(steps, safety_layer_rew_mean, label='MGAE + Safety Layer')
    plt.fill_between(steps, safety_layer_rew_mean - safety_layer_rew_std, safety_layer_rew_mean + safety_layer_rew_std, alpha=alpha, label='')

    plt.plot(steps, lagrange_rew_mean, label='MGAE + Lagrangian')
    plt.fill_between(steps, lagrange_rew_mean - lagrange_rew_std, lagrange_rew_mean + lagrange_rew_std, alpha=alpha,
                     label='')

    # plt.xlim([0, 200000])
    plt.xlim([0, 175000])

    # Add labels, title, and legend
    plt.xlabel('Environment Steps', fontdict=font)
    plt.ylabel('Episodic Reward', fontdict=font)

    # plt.title('SRE', fontdict=font_title)
    # plt.title(env_name, fontdict=font_title)

    plt.legend(loc='best', prop=font_title)
    plt.tight_layout()

    fig_rew.savefig(f'{env_name}_ep_rew_comp_training.png', dpi=300)

    # Show the plot
    # plt.show()

    fig_cost = plt.figure(figsize=(10, 6))
    plt.plot(steps, mgae_cost_mean, label='MGAE')
    plt.fill_between(steps, mgae_cost_mean - mgae_cost_std, mgae_cost_mean + mgae_cost_std, alpha=alpha, label='')

    plt.plot(steps, safety_layer_cost_mean, label='MGAE + Safety Layer')
    plt.fill_between(steps, safety_layer_cost_mean - safety_layer_cost_std, safety_layer_cost_mean + safety_layer_cost_std, alpha=alpha, label='')

    plt.plot(steps, lagrange_cost_mean, label='MGAE + Lagrangian')
    plt.fill_between(steps, lagrange_cost_mean - lagrange_cost_std, lagrange_cost_mean + lagrange_cost_std, alpha=alpha,
                     label='')

    # plt.xlim([0, 200000])
    plt.xlim([0, 175000])

    # Add labels, title, and legend
    plt.xlabel('Environment Steps', fontdict=font)
    plt.ylabel('Episodic Cost', fontdict=font)

    # plt.title('SRE', fontdict=font_title)
    # plt.title(env_name, fontdict=font_title)

    plt.legend(loc='best', prop=font_title)
    plt.tight_layout()

    fig_cost.savefig(f'{env_name}_ep_cost_comp_training.png', dpi=300)


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def get_progress_data(path: str):
    fp = os.path.join(path, 'progress.csv')
    assert os.path.exists(fp)

    data = pd.read_csv(fp)
    return data


def get_config_json(path: str):
    fp = os.path.join(path, 'config.json')
    assert os.path.exists(fp)

    with open(fp, 'r') as f:
        config = json.load(f)

    return config


def get_abs_paths(
    file_list: List[str],
    env_name: str,
    algo_name: str = 'FOCOPS_CACD',
) -> List[str]:
    abs_paths = []
    for f in file_list:
        algo_env = algo_name + '-' + '{' + env_name + '}'
        full_path = os.path.join('runs', algo_env, f)
        assert os.path.exists(full_path)

        abs_paths.append(full_path)
    return abs_paths


if __name__ == '__main__':
    # env_name: str = 'CliffCircular-v1'  # or medium
    env_name: str = 'medium'  # or CliffCircular-v1

    # For CliffCircular-v1
    filenames_cliffcircular: List[str] = [
        'seed-042-2025-02-28-08-50-27',
        'seed-042-2025-02-28-09-04-11',
        'seed-000-2025-02-28-09-19-56',
        'seed-000-2025-02-28-09-37-35',
        'seed-100-2025-02-28-10-05-29',
        'seed-100-2025-02-28-10-19-06',
        'seed-100-2025-02-28-10-38-55',
        'seed-000-2025-02-28-11-12-02',
        'seed-042-2025-02-28-11-27-30',
    ]

    # For SRE
    filenames_sre: List[str] = [
        'seed-100-2025-03-03-09-49-51',
        'seed-042-2025-03-02-22-20-30',
        'seed-000-2025-03-02-20-33-01',
        'seed-100-2025-03-02-15-10-38',
        'seed-100-2025-03-02-13-54-29',
        'seed-042-2025-03-02-12-45-53',
        'seed-042-2025-03-01-16-15-11',
        'seed-000-2025-03-01-15-06-43',
        'seed-000-2025-03-01-13-10-24',
    ]

    plot(
        exp_file_list=filenames_cliffcircular if 'Cliff' in env_name else filenames_sre,
        env_name=env_name,
    )


