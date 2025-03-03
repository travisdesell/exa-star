import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def perform_mann_whitney_u_test(df):
    """Perform the Mann-Whitney U test for each pair of methods and store p-values."""
    methods = df['Method'].unique()
    metrics = df['Metric'].unique()
    p_value_matrix = {method: {other_method: None for other_method in methods} for method in methods}

    for metric in metrics:
        metric_data = df[df['Metric'] == metric]

        for method1 in methods:
            for method2 in methods:
                method1_values = metric_data[metric_data['Method'] == method1]['Value']
                method2_values = metric_data[metric_data['Method'] == method2]['Value']

                if len(method1_values) > 0 and len(method2_values) > 0:
                    u_stat, p_value = mannwhitneyu(method1_values, method2_values, alternative='two-sided')
                    p_value_matrix[method1][method2] = p_value

    return p_value_matrix


def save_results(p_value_matrix, output_file):
    """Save the Mann-Whitney U test results to a CSV file."""
    results_df = pd.DataFrame(p_value_matrix)
    results_df.to_csv(output_file, index=False)


def plot_visualizations(df):
    """Generate and save violin and box plots for each metric."""
    metrics = df['Metric'].unique()
    for metric in metrics:
        metric_data = df[df['Metric'] == metric]

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=metric_data, x='Method', y='Value', inner="point")
        plt.axhline(metric_data[metric_data['Method'] == "EXA-AE"]['Value'].median(),
                    color='red', linestyle='--', label=f'{"EXA-AE"} Median')
        plt.legend()
        plt.title(f'Violin Plot for {metric}')
        plt.ylabel(metric)
        plt.savefig(f'violin_plot_{metric}_baseline_comparison.png')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=metric_data, x='Method', y='Value')
        plt.axhline(metric_data[metric_data['Method'] == "EXA-AE"]['Value'].median(),
                    color='red', linestyle='--', label=f'{"EXA-AE"} Median')
        plt.legend()
        plt.title(f'Box Plot for {metric}')
        plt.ylabel(metric)
        plt.savefig(f'smap_{metric}.png')
        plt.show()


def main():
    file_path = 'results/datasheet.csv'
    output_file = 'mann_whitney_u_results.csv'

    df = load_data(file_path)
    p_value_matrix = perform_mann_whitney_u_test(df)
    save_results(p_value_matrix, output_file)
    plot_visualizations(df)


if __name__ == "__main__":
    main()

