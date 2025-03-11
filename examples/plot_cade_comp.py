import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Dict, Tuple


def plot():
    # Methods
    methods = ["MGAE", "MGAE+Lagrangian", "MGAE+SafetyLayer"]

    # Example data (means)
    episodic_rewards_mean = np.array([
        [100, 90],  # MGAE: [without safety, with safety]
        [110, 95],  # MGAE+Lagrangian
        [120, 100]  # MGAE+SafetyLayer
    ])

    episodic_costs_mean = np.array([
        [30, 20],  # MGAE: [without safety, with safety]
        [25, 15],  # MGAE+Lagrangian
        [20, 10]  # MGAE+SafetyLayer
    ])

    # Example data (standard deviations)
    episodic_rewards_std = np.array([
        [5, 6],  # MGAE
        [7, 5],  # MGAE+Lagrangian
        [6, 4]  # MGAE+SafetyLayer
    ])

    episodic_costs_std = np.array([
        [3, 2],  # MGAE
        [2, 2],  # MGAE+Lagrangian
        [1, 1]  # MGAE+SafetyLayer
    ])

    # Plot parameters
    bar_width = 0.3  # Width of each bar
    x = np.arange(len(methods))  # X positions for the groups

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars with error bars
    for i in range(len(methods)):
        # Left bars (Without Safety)
        ax.bar(x[i] - bar_width / 2, episodic_rewards_mean[i, 0], width=bar_width, color='blue',
               label="Reward" if i == 0 else "",
               yerr=episodic_rewards_std[i, 0], capsize=5, alpha=0.8)
        ax.bar(x[i] - bar_width / 2, episodic_costs_mean[i, 0], width=bar_width, color='red',
               label="Cost" if i == 0 else "",
               yerr=episodic_costs_std[i, 0], bottom=episodic_rewards_mean[i, 0], capsize=5, alpha=0.8)

        # Right bars (With Safety)
        ax.bar(x[i] + bar_width / 2, episodic_rewards_mean[i, 1], width=bar_width, color='blue',
               yerr=episodic_rewards_std[i, 1], capsize=5, alpha=0.8)
        ax.bar(x[i] + bar_width / 2, episodic_costs_mean[i, 1], width=bar_width, color='red',
               yerr=episodic_costs_std[i, 1], bottom=episodic_rewards_mean[i, 1], capsize=5, alpha=0.8)

    # Labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Values")
    ax.set_title("Comparison of Episodic Rewards and Costs with Error Bars")

    # Legend
    ax.legend(["Reward", "Cost"], loc="upper right")

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_scatter(
    mean_values: Dict[str, Dict[str, Tuple[float, float]]],
    std_values: Dict[str, Dict[str, Tuple[float, float]]],
    env_name: str,
):
    # Example data: Mean and Std for 3 Methods, Aggregated Across 3 Environments
    methods = ["MGAE", "MGAE+Lagrangian", "MGAE+SafetyLayer"]
    conditions = ["No Safety", "With Safety"]

    # Colors and markers for differentiation
    colors = {"MGAE": "blue", "MGAE+Lagrangian": "green", "MGAE+SafetyLayer": "purple"}
    markers = {"No Safety": "o", "With Safety": "s"}

    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.style.use("seaborn-v0_8-darkgrid")
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 15,
            }

    # Plot ellipsoids
    for method in methods:
        for condition in conditions:
            reward_mean, cost_mean = mean_values[method][condition]
            reward_std, cost_std = std_values[method][condition]

            # Scatter center point
            ax.scatter(reward_mean, cost_mean, marker=markers[condition], color=colors[method], s=100,
                       label=f"{method} - {condition}")

            # Draw error ellipse (scaled 1 std deviation)
            ellipse = Ellipse(xy=(reward_mean, cost_mean), width=2 * reward_std, height=2 * cost_std,
                              edgecolor=colors[method], facecolor=colors[method], alpha=0.3, linestyle='--', lw=1.5)
            ax.add_patch(ellipse)

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="lower right")

    plt.xlabel("Episodic Reward", fontdict=font)
    plt.ylabel("Episodic Cost", fontdict=font)
    # plt.title("Scatter Plot of Episodic Reward vs. Cost with Variability")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'./{env_name}_eval_scatter_comp.png', dpi=300)

    # plt.show()


if __name__ == '__main__':
    # plot()

    # CliffCircular means (episodic rewards, episodic costs)
    mean_values_cliffcircular = {
        "MGAE": {"No Safety": (16.56, 0.58), "With Safety": (17.26, 0.52)},
        "MGAE+Lagrangian": {"No Safety": (19.16, 0.47), "With Safety": (19, 0.51)},
        "MGAE+SafetyLayer": {"No Safety": (15.16, 0.64), "With Safety": (15.27, 0.62)}
    }

    # CliffCircular std deviations (rewards, costs)
    std_values_cliffcircular = {
        "MGAE": {"No Safety": (5.51, 0.56), "With Safety": (4.99, 0.52)},
        "MGAE+Lagrangian": {"No Safety": (2.45, 0.33), "With Safety": (2.66, 0.36)},
        "MGAE+SafetyLayer": {"No Safety": (7.07, 0.54), "With Safety": (6.99, 0.53)}
    }

    mean_values_riverine = {
        "MGAE": {"No Safety": (48.35, 0.97), "With Safety": (50.57, 0.96)},
        "MGAE+Lagrangian": {"No Safety": (35.56, 0.98), "With Safety": (37.03, 0.98)},
        "MGAE+SafetyLayer": {"No Safety": (53.02, 0.96), "With Safety": (57, 0.96)}
    }

    # CliffCircular std deviations (rewards, costs)
    std_values_riverine = {
        "MGAE": {"No Safety": (47.93, 0.13), "With Safety": (51.63, 0.15)},
        "MGAE+Lagrangian": {"No Safety": (38.65, 0.1), "With Safety": (39.75, 0.1)},
        "MGAE+SafetyLayer": {"No Safety": (50.46, 0.13), "With Safety": (53.91, 0.13)}
    }

    # env_name = 'cliffcircular'
    env_name = 'riverine'
    plot_scatter(
        mean_values=mean_values_cliffcircular if 'cliff' in env_name else mean_values_riverine,
        std_values=std_values_cliffcircular if 'cliff' in env_name else std_values_riverine,
        env_name=env_name,
    )
