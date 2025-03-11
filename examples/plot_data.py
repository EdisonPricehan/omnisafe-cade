import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Plot the metrics (episodic return) during training
    '''
    # Load the CSV file
    # file_path = 'cliffcircular_adv_3seeds.csv'
    file_path = 'riverine_adv_3seeds.csv'

    data = pd.read_csv(file_path)

    # Get the list of algorithms from the column headers
    adv_labels_orig = list(set(col.split('-seed-')[0] for col in data.columns if '-seed-' in col))

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
            'size': 15,
            }
    font_title = {'family': 'STIXGeneral',
                  'weight': 'bold',
                  'size': 20,
                  }

    # Iterate over each algorithm and process the data
    for i, adv in enumerate(adv_labels_orig):
        # Extract the columns for the specific algorithm's seeds
        seed_columns = [col for col in data.columns if col.startswith(f'{adv}-seed-')]

        # Compute the mean and standard deviation across seeds
        mean_ep_ret = data[seed_columns].mean(axis=1)
        std_ep_ret = data[seed_columns].std(axis=1)

        # Plot the mean and add a shaded area for the standard deviation
        plt.plot(data['TotalEnvSteps'], mean_ep_ret, label=adv_labels[i])
        plt.fill_between(data['TotalEnvSteps'], mean_ep_ret - std_ep_ret, mean_ep_ret + std_ep_ret, alpha=0.2, label='')

    plt.xlim([0, max(data['TotalEnvSteps'])])

    # Add labels, title, and legend
    plt.xlabel('Environment Steps', fontdict=font)
    plt.ylabel('Episodic Reward', fontdict=font)

    # plt.title('SRE', fontdict=font_title)
    plt.title('CliffCircular', fontdict=font_title)

    plt.legend(loc='best', prop=font_title)
    plt.tight_layout()

    plt.savefig(f'{file_path.split("_")[0]}_adv_training.png', dpi=300)

    # Show the plot
    # plt.show()
