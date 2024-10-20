import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class GraphingModel(nn.Module):
    """
    A simple neural network used to perform a mapping of N to 2,
    where N is the maximum number of genes.
    The 2D result is used for visualization purposes.
    """

    def __init__(self, genome_size, hidden_size):
        super(GraphingModel, self).__init__()
        self.layer1 = nn.Linear(genome_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 2)
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Scaling factor γ as a learnable parameter

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x) * self.gamma
        return x

    def adjust_output_bias(self, founder_genome_output):
        """
        Adjust bias so that the founder genome maps to [0, 0]
        """
        with torch.no_grad():
            # Reshape founder_genome_output to match the shape of the bias
            self.layer2.bias -= founder_genome_output.view_as(self.layer2.bias)

    def apply_rotation(self, final_genome_output):
        """
        Rotate final genome output to [1, 0] using a rotation matrix
        """
        with torch.no_grad():
            final_genome_output = final_genome_output.squeeze()

            theta = np.atan2(float(final_genome_output[1]), float(final_genome_output[0]))

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            rot = torch.tensor(
                [
                    [cos_theta, sin_theta],
                    [-sin_theta, cos_theta]],
                dtype=torch.float32)

            rot /= torch.norm(final_genome_output)
            self.layer2.weight = nn.Parameter(rot @ self.layer2.weight)
            self.layer2.bias = nn.Parameter(rot @ self.layer2.bias)

    def calc_gamma(self, dataloader):
        """
        Calculate the gamma value needed for this.
        """
        with torch.no_grad():
            total = 0.0
            count = 0
            for x_batch_ in dataloader:
                x_batch = x_batch_[0]
                half_ = len(x_batch) // 2
                x1 = x_batch[:half_]
                x2 = x_batch[half_:]
                if len(x1) != len(x2):
                    break

                y1 = self(x1)
                y2 = self(x2)

                input_distances = torch.norm(x1 - x2, dim=1)
                output_distances = torch.norm(y1 - y2, dim=1)
                ratio = input_distances / output_distances
                for element in ratio:
                    if not element.isnan():
                        total += torch.mean(element)
                        count += 1
            self.gamma = nn.Parameter(self.gamma * total / count)


def fit(
    model, data, founder_genes, final_best_genes,
    batch_size=64, epochs=1000, lr=0.001
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data_tensor = data.clone().detach()

    founder_output = model(founder_genes)
    model.adjust_output_bias(founder_output)

    final_output = model(final_best_genes)
    model.apply_rotation(final_output)

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.calc_gamma(dataloader)

    best_genome_input = np.array([final_best_genes], dtype=np.float32)
    best_genome_output = model(torch.from_numpy(best_genome_input)) / model.gamma
    print(f"best genome output: {best_genome_output}")

    founder_input = np.array([founder_genes], dtype=np.float32)
    founder_output = model(torch.from_numpy(founder_input)) / model.gamma
    print(f"founder output: {founder_output}")

    for epoch in range(epochs):
        model.train()

        total_loss_primary = 0.0
        total_batches = 0

        for x_batch_ in dataloader:
            x_batch = x_batch_[0]
            half_ = len(x_batch) // 2
            x1 = x_batch[:half_]
            x2 = x_batch[half_:]
            if len(x1) != len(x2):
                break

            y1 = model(x1)
            y2 = model(x2)

            input_distances = torch.norm(x1 - x2, dim=1)
            output_distances = torch.norm(y1 - y2, dim=1)

            loss = loss_fn(output_distances, input_distances)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_primary += loss.item()
            total_batches += 1

        avg_loss_primary = total_loss_primary / total_batches

        print(f"Epoch [{epoch + 1}/{epochs}], Primary Loss: {avg_loss_primary:.4f}")

    founder_input = np.array([founder_genes], dtype=np.float32)
    founder_output = model(torch.from_numpy(founder_input)) / model.gamma
    model.adjust_output_bias(founder_output)

    best_genome_input = np.array([final_best_genes], dtype=np.float32)
    best_genome_output = model(torch.from_numpy(best_genome_input)) / model.gamma
    model.apply_rotation(best_genome_output)

    best_genome_input = np.array([final_best_genes], dtype=np.float32)
    best_genome_output = model(torch.from_numpy(best_genome_input)) / model.gamma
    print(f"best genome output: {best_genome_output}")

    founder_input = np.array([founder_genes], dtype=np.float32)
    founder_output = model(torch.from_numpy(founder_input)) / model.gamma
    print(f"founder output: {founder_output}")


def map_genomes_to_2d(
        genome_data: np.ndarray, genome_id_to_index: dict, final_best_genes: np.ndarray, hidden_size: int=128):
    genome_size = genome_data.shape[1]
    model = GraphingModel(genome_size, hidden_size)

    genome_data_tensor = torch.tensor(genome_data, dtype=torch.float32)

    founder_genome = np.zeros(genome_size, dtype=np.float32)
    founder_genome = torch.tensor(founder_genome, dtype=torch.float32).unsqueeze(0)
    final_best_genes = torch.tensor(final_best_genes, dtype=torch.float32).unsqueeze(0)

    fit(model, genome_data_tensor, founder_genome, final_best_genes, batch_size=8, epochs=1000, lr=0.001)

    model.eval()
    with torch.no_grad():
        reduced_genome = (model(genome_data_tensor) / model.gamma).numpy()
        return {genome_id: reduced_genome[i] for genome_id, i in genome_id_to_index.items()}
