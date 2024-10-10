import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(
    sdm_data_path: str,
    sdm_mlp_data_path: str,
    ldm_data_path: str,
    ldm_mlp_data_path: str,
    baseline_data_path: str,
    output_path: str,
    metric: str = 'iou'
):
    """
    Plot function from saved csv files that have metrics
    Args:
        sdm_data_path:
        sdm_mlp_data_path:
        ldm_data_path:
        ldm_mlp_data_path:
        baseline_data_path:
        output_path:
        metric:

    Returns:

    """
    # Load the CSV files
    sdm_data = pd.read_csv(sdm_data_path)
    sdm_mlp_data = pd.read_csv(sdm_mlp_data_path)
    ldm_data = pd.read_csv(ldm_data_path)
    ldm_mlp_data = pd.read_csv(ldm_mlp_data_path)
    baseline_data = pd.read_csv(baseline_data_path)

    # Calculate mean and standard deviation for IoU at each step
    sdm_stats = sdm_data.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()
    sdm_mlp_stats = sdm_mlp_data.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()
    ldm_stats = ldm_data.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()
    ldm_mlp_stats = ldm_mlp_data.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()
    baseline_stats = baseline_data.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()

    # Print statistics for each method
    print(f'Metric: {metric}')
    print("\nSDM Stats:")
    print(sdm_stats)
    print("\nSDM-MLP Stats:")
    print(sdm_mlp_stats)
    print("\nLDM Stats:")
    print(ldm_stats)
    print("\nLDM-MLP Stats:")
    print(ldm_mlp_stats)
    print("\nBaseline Stats:")
    print(baseline_stats)

    # Plot the data with error bars
    plt.figure(figsize=(10, 6))

    # Plot for SDM model with enhanced error bars using high-contrast blue color
    plt.errorbar(
        sdm_stats['step'] + 1,
        sdm_stats['mean'],
        yerr=sdm_stats['std'],
        label='SDM',
        linestyle='-',
        marker='o',
        color='royalblue',  # High-contrast blue color
        markersize=6,
        capsize=4,  # Set a capsize for visibility
        elinewidth=1.5,
        linewidth=1.5,
        capthick=1.5  # Make the caps thicker for clarity
    )

    # Plot for SDM-MLP model with enhanced error bars using orange color
    plt.errorbar(
        sdm_mlp_stats['step'] + 1,
        sdm_mlp_stats['mean'],
        yerr=sdm_mlp_stats['std'],
        label='SDM-MLP',
        linestyle='--',
        marker='s',
        color='darkorange',
        markersize=6,
        capsize=4,  # Set a capsize for visibility
        elinewidth=1.5,
        linewidth=1.5,
        capthick=1.5  # Make the caps thicker for clarity
    )

    # Plot for LDM (RSSM) model with enhanced error bars using forestgreen color
    plt.errorbar(
        ldm_stats['step'] + 1,
        ldm_stats['mean'],
        yerr=ldm_stats['std'],
        label='LDM',
        linestyle='-',
        marker='^',
        color='forestgreen',
        markersize=6,
        capsize=4,  # Set a capsize for visibility
        elinewidth=1.5,
        linewidth=1.5,
        capthick=1.5  # Make the caps thicker for clarity
    )

    # Plot for LDM-MLP model with enhanced error bars using crimson color
    plt.errorbar(
        ldm_mlp_stats['step'] + 1,
        ldm_mlp_stats['mean'],
        yerr=ldm_mlp_stats['std'],
        label='LDM-MLP',
        linestyle=':',
        marker='d',
        color='crimson',
        markersize=6,
        capsize=4,  # Set a capsize for visibility
        elinewidth=1.5,
        linewidth=1.5,
        capthick=1.5  # Make the caps thicker for clarity
    )

    # Plot for baseline with enhanced error bars using black color
    plt.errorbar(
        baseline_stats['step'] + 1,
        baseline_stats['mean'],
        yerr=baseline_stats['std'],
        label='Baseline',
        linestyle='solid',
        marker='d',
        color='black',
        markersize=6,
        capsize=4,  # Set a capsize for visibility
        elinewidth=1.5,
        linewidth=1.5,
        capthick=1.5  # Make the caps thicker for clarity
    )

    # Customize the plot
    if metric == 'iou':
        plt.title('IoU Performance Comparison')
        plt.ylabel('IoU')
    else:
        plt.title('L1 Loss Comparison')
        plt.ylabel('L1')
    plt.xlabel('Prediction Step')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.xticks(range(1, 11))
    plt.tight_layout()

    # Save and display the plot
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    # Compare iou, riverine env
    plot_metric_comparison(
        sdm_data_path='riverine_sdm_iou_test_iou_h10.csv',
        sdm_mlp_data_path='riverine_sdm_mlp_iou_test_iou_h10.csv',
        ldm_data_path='riverine_ldm_test_iou_h10.csv',
        ldm_mlp_data_path='riverine_ldm_mlp_test_iou_h10.csv',
        baseline_data_path='riverine_baseline_test_iou_h10.csv',
        output_path='riverine_iou_comparison.png',
        metric='iou',
    )

    # Compare L1, riverine env
    plot_metric_comparison(
        sdm_data_path='riverine_sdm_iou_test_l1_h10.csv',
        sdm_mlp_data_path='riverine_sdm_mlp_iou_test_l1_h10.csv',
        ldm_data_path='riverine_ldm_test_l1_h10.csv',
        ldm_mlp_data_path='riverine_ldm_mlp_test_l1_h10.csv',
        baseline_data_path='riverine_baseline_test_l1_h10.csv',
        output_path='riverine_l1_comparison.png',
        metric='l1',
    )

    # CliffCircular, 5x5 observation, IoU
    # plot_metric_comparison(
    #     sdm_data_path='sdm_test_l1_cliff_metric_iou.csv',
    #     sdm_mlp_data_path='sdm_mlp_iou_patch8_test_iou_h10_p8.csv',
    #     ldm_data_path='ldm_test_iou_h10_p8.csv',
    #     ldm_mlp_data_path='ldm_mlp_test_iou_h10_p8.csv',
    #     baseline_data_path='baseline_test_iou_h10_p8.csv',
    #     output_path='iou_comparison_patch8.png',
    #     metric='iou',
    # )




