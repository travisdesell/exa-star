import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Adjust font size on plots
plt.rcParams.update({'font.size': 18})

def plot_trainable_parameters(csv_file):
    df = pd.read_csv(csv_file)

    # Apply logarithmic transformation to the 'Trainable Parameters' for better comparison
    df['Log Trainable Parameters'] = np.log10(df['Trainable Parameters'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Log Trainable Parameters', y='Method', data=df, hue='Method', palette="viridis", legend=False)

    plt.title('Comparison of Trainable Parameters (Log Scale)', fontsize=14)
    plt.xlabel('Log10 of Trainable Parameters (SMAP)', fontsize=12)
    plt.ylabel('Method', fontsize=12)

    plt.savefig("smap_trainable", bbox_inches='tight')
    # Show the plot
    plt.show()


def plot_trainable_param_violin(csv_filenames, dataset_names):
    """
    Plots a violin plot showing the range of trainable parameters for evolved network genome.

    Parameters:
        csv_filenames (list): A list of file paths to the CSV files containing trainable parameters.
        dataset_names (list): A ordered list of dataset names.
    """
    combined_data = []

    for i, file in enumerate(csv_filenames):
        df = pd.read_csv(file)
        genome_df = df[df['Method'].str.startswith("genome_")].copy()

        # Add a column to indicate the file source
        genome_df.loc[:, 'Source'] = dataset_names[i]
        combined_data.append(genome_df)

    combined_df = pd.concat(combined_data, ignore_index=True)

    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=combined_df,
        x="Source",
        y="Trainable Parameters",
        hue="Source",
        inner="quartile",
        palette="muted",
        dodge=False,
    )
    plt.title("Combined Distribution of Trainable Parameters for Evolved Networks")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Trainable Parameters")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('trainable_param_violin.png')
    plt.show()

def plot_trainable_param_box(csv_filenames, dataset_names):
    """
    Plots a box and whisker plot showing the range of trainable parameters for evolved network genome.

    Parameters:
        csv_filenames (list): A list of file paths to the CSV files containing trainable parameters.
        dataset_names (list): An ordered list of dataset names.
    """
    combined_data = []

    for i, file in enumerate(csv_filenames):
        df = pd.read_csv(file)
        genome_df = df[df['Method'].str.startswith("genome_")].copy()

        # Add a column to indicate the file source
        genome_df.loc[:, 'Source'] = dataset_names[i]
        combined_data.append(genome_df)

    combined_df = pd.concat(combined_data, ignore_index=True)

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=combined_df,
        x="Source",
        y="Trainable Parameters",
        hue="Source",
        palette="muted",
        dodge=False,
    )
    plt.title("Combined Distribution of Trainable Parameters for Evolved Networks")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Trainable Parameters")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('trainable_param_box.png')
    plt.show()

def plot_nodes_violin(csv_filename):
    """
    Creates violin plots to compare the number of LSTM nodes vs regular nodes

    Parameters:
        csv_filename (str): Path to the CSV file containing node comparison data
    """
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    melted_df = pd.melt(
        df,
        value_vars=["LSTM", "Regular"],
        var_name="Node Type",
        value_name="Values"
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=melted_df,
        x="Node Type",
        y="Values",
        inner="quartile",
        palette="Set2",
    )
    plt.title("LSTM Nodes vs. Regular Nodes: SMAP Dataset")
    plt.xlabel("Node Type")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()

    plt.savefig('smap_nodes_violin.png')
    plt.show()

def plot_nodes_box(csv_filename):
    """
    Creates box plots to compare the number of LSTM nodes, regular nodes, and total nodes
    for the full network, encoder, and decoder.

    The x-axis will show the network type (Full Network, Encoder, Decoder), and the node type will be color coded.

    Parameters:
        csv_filename (str): Path to the CSV file containing node comparison data
    """
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Melt data for full network
    full_network_df = pd.melt(
        df,
        value_vars=["LSTM", "Regular", "Total"],
        var_name="Node Type",
        value_name="Values"
    )
    full_network_df["Category"] = "Full Network"

    # Melt data for encoder
    encoder_df = pd.melt(
        df,
        value_vars=["Encoder - LSTM", "Encoder - Regular", "Encoder - Total"],
        var_name="Node Type",
        value_name="Values"
    )
    encoder_df["Node Type"] = encoder_df["Node Type"].str.replace("Encoder - ", "")
    encoder_df["Category"] = "Encoder"

    # Melt data for decoder
    decoder_df = pd.melt(
        df,
        value_vars=["Decoder - LSTM", "Decoder - Regular", "Decoder - Total"],
        var_name="Node Type",
        value_name="Values"
    )
    decoder_df["Node Type"] = decoder_df["Node Type"].str.replace("Decoder - ", "")
    decoder_df["Category"] = "Decoder"

    # Combine all data
    combined_df = pd.concat([full_network_df, encoder_df, decoder_df], ignore_index=True)

    # Rearrange x-axis for network type and use node type for color coding
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        data=combined_df,
        x="Category",
        y="Values",
        hue="Node Type",
        palette="Set2"
    )
    plt.title("Comparison of Nodes: Full Network, Encoder, and Decoder - SMAP Dataset")
    plt.xlabel("Network Type")
    plt.ylabel("Number of Nodes")
    plt.legend(title="Node Type")
    plt.tight_layout()

    plt.savefig('smap_nodes.png')
    plt.show()

def plot_edges_violin(csv_filename):
    """
    Creates violin plots to compare the number of recurrent edges vs non-recurrent edges

    Parameters:
        csv_filename (str): Path to the CSV file containing edge comparison data
    """
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    melted_df = pd.melt(
        df,
        value_vars=["Recurrent", "Non-recurrent"],
        var_name="Edge Type",
        value_name="Values"
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=melted_df,
        x="Edge Type",
        y="Values",
        inner="quartile",
        palette="Set2",
    )
    plt.title("Recurrent Edges vs. Non-recurrent Edges: SMAP Dataset")
    plt.xlabel("Edge Type")
    plt.ylabel("Number of Edges")
    plt.tight_layout()

    plt.savefig('smap_edges_violin.png')
    plt.show()

def plot_edges_box(csv_filename):
    """
    Creates box plots to compare the number of recurrent edges, non-recurrent edges, and total edges
    for the full network, encoder, and decoder.

    The x-axis will show the network type (Full Network, Encoder, Decoder), and the edge type will be color coded.

    Parameters:
        csv_filename (str): Path to the CSV file containing edge comparison data
    """
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Melt data for full network
    full_network_df = pd.melt(
        df,
        value_vars=["Recurrent", "Non-recurrent", "Total"],
        var_name="Edge Type",
        value_name="Values"
    )
    full_network_df["Category"] = "Full Network"

    # Melt data for encoder
    encoder_df = pd.melt(
        df,
        value_vars=["Encoder - Recurrent", "Encoder - Non-recurrent", "Encoder - Total"],
        var_name="Edge Type",
        value_name="Values"
    )
    encoder_df["Edge Type"] = encoder_df["Edge Type"].str.replace("Encoder - ", "")
    encoder_df["Category"] = "Encoder"

    # Melt data for decoder
    decoder_df = pd.melt(
        df,
        value_vars=["Decoder - Recurrent", "Decoder - Non-recurrent", "Decoder - Total"],
        var_name="Edge Type",
        value_name="Values"
    )
    decoder_df["Edge Type"] = decoder_df["Edge Type"].str.replace("Decoder - ", "")
    decoder_df["Category"] = "Decoder"

    # Combine all data
    combined_df = pd.concat([full_network_df, encoder_df, decoder_df], ignore_index=True)

    # Rearrange x-axis for network type and use edge type for color coding
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        data=combined_df,
        x="Category",
        y="Values",
        hue="Edge Type",
        palette="Set2"
    )
    plt.title("Comparison of Edges: Full Network, Encoder, and Decoder - MSL Dataset")
    plt.xlabel("Network Type")
    plt.ylabel("Number of Edges")
    plt.legend(title="Edge Type")
    plt.tight_layout()

    plt.savefig('msl_edges.png')
    plt.show()

def plot_recurrent_depth_violin(folder_path):
    """
    This function takes all the CSV files in a folder and creates a single plot
    with side-by-side violin plots for the data inside each CSV file.

    Args:
    - folder_path (str): Path to the folder containing CSV files.

    Returns:
    - None: Displays the plot.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    combined_data = []

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)

        column_name = data.columns[0]
        # Add a column indicating the source of the data
        data["Source"] = os.path.splitext(csv_file)[0]
        data.rename(columns={column_name: "Value"}, inplace=True)

        combined_data.append(data)

    combined_df = pd.concat(combined_data, ignore_index=True)

    genome_order = sorted(combined_df["Source"].unique())
    # Plot violin plots
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=combined_df, x="Source", y="Value", scale="width", palette="muted", order=genome_order)

    plt.title("Ranges of Recurrent Depths on Recurrent Edges: SMAP Dataset", fontsize=16)
    plt.xlabel("Genome", fontsize=14)
    plt.ylabel("Recurrent Depth", fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('smap_recurrent_depth_violin.png')
    plt.show()

def plot_depth_hist(folder_path):
    # Collect files from the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize variables
    data = {}

    # Read each CSV file and store its data
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        data[file] = pd.read_csv(file_path, header=None).squeeze()  # Read as a Series

    # Combine all data into a single DataFrame for grouped plotting
    all_data = []
    for file, depths in data.items():
        all_data.append(pd.DataFrame({'Recurrent Depth': depths, 'Genome': file}))

    combined_data = pd.concat(all_data, ignore_index=True)

    # Group and count occurrences of recurrent depth for each genome
    grouped = combined_data.groupby(['Genome', 'Recurrent Depth']).size().reset_index(name='Count')

    # Normalize the counts by the number of rows (edges) in each CSV file
    # This means dividing the count by the total number of rows in the file (i.e., the number of edges)
    grouped['Total Edges'] = grouped['Genome'].map(lambda genome: len(data[genome]))
    grouped['Normalized Count'] = grouped['Count'] / grouped['Total Edges']

    # Plot grouped bar chart using seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=grouped,
        x='Recurrent Depth',
        y='Normalized Count',
        hue='Genome',
        palette='muted',
        dodge=True,
        edgecolor='black'
    )

    # Configure the plot
    plt.xlabel("Recurrent Depth")
    plt.ylabel("Normalized Count")
    plt.title("Recurrent Depth Distribution Across Genomes - CATS Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Genomes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    plt.savefig("cats_recurrent_depth_histogram.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_depth_boxplot(cats_folder, msl_folder, smap_folder):
    all_data = []

    # Function to process a dataset folder
    def process_dataset(folder_path, dataset_label):
        dataset_data = []

        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                depths = pd.read_csv(file_path, header=None).squeeze()  # Read the depths as a Series
                total_edges = len(depths)  # Calculate the number of rows (edges) in the file

                # Count occurrences of each recurrent depth (group counts)
                group_counts = depths.value_counts().reset_index()
                group_counts.columns = ['Recurrent Depth', 'Count']  # Rename columns

                # Normalize counts by dividing by the total number of rows (edges) in that file
                group_counts['Normalized Count'] = group_counts['Count'] / total_edges

                # Add dataset and genome name
                group_counts['Dataset'] = dataset_label
                group_counts['Genome'] = file

                # Add the processed data to the list
                dataset_data.append(group_counts)

        return dataset_data

    # Process each of the three datasets
    all_data.extend(process_dataset(cats_folder + '/recurrent edge depths', 'CATS Dataset'))
    all_data.extend(process_dataset(msl_folder + '/recurrent edge depths', 'MSL Dataset'))
    all_data.extend(process_dataset(smap_folder + '/recurrent edge depths', 'SMAP Dataset'))

    # Combine all the data into one DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    print(all_data)

    # Plot box-and-whisker plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=combined_data,
        x='Recurrent Depth',
        y='Normalized Count',
        hue='Dataset',
        palette='muted',
        showfliers=True
    )

    # Configure the plot
    plt.xlabel("Recurrent Depth")
    plt.ylabel("Normalized Count")
    plt.title("Recurrent Depth Distribution Across Datasets")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    plt.savefig("recurrent_depth_boxplot.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()


def create_boxplots_trainable(folder_path):
    """
    Creates side-by-side box-and-whisker plots for each CSV file in a folder.

    Parameters:
        folder_path (str): Path to the folder containing the CSV files.
    """
    # Collect files from the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Iterate over each file
    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Filter rows where "Method" starts with "genome_"
        filtered_df = df[df["Method"].str.startswith("genome_", na=False)]

        # Rename "Trainable Parameters" to "Full Network" for plotting
        plot_data = filtered_df.rename(columns={"Trainable Parameters": "Full Network"})

        # Select columns for plotting
        plot_columns = ["Full Network", "Encoder", "Decoder"]

        # Reshape the data for seaborn's boxplot
        melted_df = plot_data.melt(
            id_vars=["Method"],
            value_vars=plot_columns,
            var_name="Component",
            value_name="Value"
        )

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=melted_df,
            x="Component",
            y="Value",
            palette="muted",
            width=0.6
        )

        # Configure the plot
        plt.title(f"Range of Trainable Parameters - Full Network, Encoder, Decoder", fontsize=14)
        plt.xlabel("Network Type", fontsize=12)
        plt.ylabel("Number of Trainable Parameters", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot (optional)
        plt.savefig(os.path.join(folder_path, f"{file}_boxplot.png"), dpi=300, bbox_inches='tight')

        # Show the plot
        plt.tight_layout()
        plt.show()

# create_boxplots_trainable("results/trainable parameters")

# csv_filename = "results/smap/edge_comparison.csv"
# plot_edges_violin(csv_filename)1

# plot_depth_hist("results/cats/recurrent edge depths")
# plot_depth_boxplot("results/cats", "results/msl", "results/smap")

csv_filename = "results/msl/msl_edge_comparison.csv"
plot_edges_box(csv_filename)

# csv_filename = "results/smap/smap_node_comparison.csv"
# plot_nodes_box(csv_filename)

# csv_files = ["results/cats/cats trainable parameters.csv", "results/msl/msl trainable parameters.csv", "results/smap/smap trainable parameters.csv"]
# plot_trainable_param_box(csv_files, ["CATS", "MSL", "SMAP"])

# csv_file = 'results/smap/trainable parameters.csv'
# plot_trainable_parameters(csv_file)
