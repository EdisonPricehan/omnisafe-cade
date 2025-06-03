import pandas as pd
import matplotlib.pyplot as plt


def plot_ep_ret_training(env_id: str) -> None:
    """
    Plot the metrics (episodic return) during training.

    :param env_id: {cliffcircular, riverine}
    :return:
    """
    # Load the CSV file
    is_cliff = False
    if 'cliff' in env_id:
        data = pd.read_csv('cliffcircular_adv_3seeds.csv')
        data_reinforce = pd.read_csv('cliffcircular_adv_3seeds_reinforce.csv')
        is_cliff = True
    else:
        assert 'riverine' in env_id
        data = pd.read_csv('riverine_adv_3seeds.csv')

    # Get the list of algorithms from the column headers
    adv_labels_orig = list(set(col.split('-seed-')[0] for col in data.columns if '-seed-' in col))
    if is_cliff:
        adv_labels_orig += list(set(col.split('-seed-')[0] for col in data_reinforce.columns if '-seed-' in col))

    # Sort labels by name
    adv_labels_orig = sorted(adv_labels_orig)
    print(f'{adv_labels_orig=}')

    # Adjust name
    adv_labels = adv_labels_orig.copy()
    adv_labels[adv_labels.index('mgae')] = 'MGAE'
    adv_labels[adv_labels.index('td')] = 'TD'
    adv_labels[adv_labels.index('gae')] = 'GAE'
    adv_labels[adv_labels.index('gae-rtg')] = 'GAE-RTG'
    adv_labels[adv_labels.index('vtrace')] = 'V-trace'

    print(f'{adv_labels=}')
    print(f'{adv_labels_orig=}')

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-darkgrid")
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 20,
            }
    font_title = {'family': 'STIXGeneral',
                  'weight': 'bold',
                  'size': 25,
                  }

    # Iterate over each algorithm and process the data
    for i, adv in enumerate(adv_labels_orig):
        # Extract the columns for the specific algorithm's seeds
        if adv == 'REINFORCE' and is_cliff:
            seed_columns = [col for col in data_reinforce.columns if col.startswith(f'{adv}-seed-')]
            # Compute the mean and standard deviation across seeds
            mean_ep_ret = data_reinforce[seed_columns].mean(axis=1)
            std_ep_ret = data_reinforce[seed_columns].std(axis=1)
        else:
            seed_columns = [col for col in data.columns if col.startswith(f'{adv}-seed-')]
            # Compute the mean and standard deviation across seeds
            mean_ep_ret = data[seed_columns].mean(axis=1)
            std_ep_ret = data[seed_columns].std(axis=1)

        # Plot the mean and add a shaded area for the standard deviation
        if adv == 'REINFORCE' and is_cliff:
            plt.plot(data_reinforce['TotalEnvSteps'], mean_ep_ret, label=adv_labels[i])
            plt.fill_between(data_reinforce['TotalEnvSteps'], mean_ep_ret - std_ep_ret, mean_ep_ret + std_ep_ret,
                             alpha=0.2, label='')
        else:
            plt.plot(data['TotalEnvSteps'], mean_ep_ret, label=adv_labels[i])
            plt.fill_between(data['TotalEnvSteps'], mean_ep_ret - std_ep_ret, mean_ep_ret + std_ep_ret, alpha=0.2,
                             label='')

    plt.xlim([0, max(data['TotalEnvSteps'])])
    max_ep_ret = data.iloc[:, 1:].max().max()
    print(f'{max_ep_ret=}')
    plt.ylim([0, max_ep_ret])

    # Add labels, title, and legend
    plt.xlabel('Environment Steps', fontdict=font)
    plt.ylabel('Episodic Reward', fontdict=font)

    if is_cliff:
        plt.title('CliffCircular', fontdict=font_title)
    else:
        plt.title('SRE', fontdict=font_title)

    plt.legend(loc='best', prop=font)
    plt.tight_layout()

    plt.savefig(f'{env_id}_adv_training.png', dpi=300)

    # Show the plot
    # plt.show()


if __name__ == '__main__':
    plot_ep_ret_training(env_id='cliffcircular')

    # plot_ep_ret_training(env_id='riverine')
