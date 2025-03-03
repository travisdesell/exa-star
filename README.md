# EXA-AE: Evolutionary eXploration of Augmenting AutoEncoders

## Overview

EXA-AE is a novel and highly efficient methodology for multivariate time series anomaly detection, leveraging 
neuro-evolutionary techniques. It extends the Evolutionary eXploration of Augmenting Memory Models (EXAMM) algorithm to 
co-evolve asymmetric encoder and decoder components of autoencoder networks in a new algorithm called Evolutionary 
eXploration of Augmenting AutoEncoders (EXA-AE).

The evolutionary process occurs at a fine-grained edge and node level, generating LSTM recurrent autoencoder networks 
with minimal compute requirements and a significantly reduced number of trainable parameters. The trained networks are 
combined with a one-class support vector machine (OCSVM) to define boundary thresholds, resulting in anomaly detection 
that outperforms existing methods in accuracy and cumulative F1 score, while drastically reducing computational overhead.

## Installation

Ensure you have the necessary dependencies installed by running:

```
pip install -r requirements.txt 
```

## Usage

### Running EXA-AE

To execute the EXA-AE algorithm, run:

```
python exa_ae.py
```

This script will use exa_ae_config.ini for configuration. The config file includes the following customizable parameters:

* training_data

* parameters

* bidirectional_ae

* num_generations

* num_iterations

* learning_rate

The best-fit genome will be saved as a .pkl file in the test_genomes/ directory.

### Evaluating the Best Fit Genome

To analyze the best-evolved genome and generate statistics, reconstructions, and a network diagram, run:

```
python test_genomes/evaluate_genome.py <genome_pkl> <testing_data> <output_filename>
```

* <genome_pkl>: Path to the saved genome file.

* <testing_data>: Dataset used for reconstructions.

* <output_filename>: File to save the reconstruction outputs.

### Performing Anomaly Detection

To apply anomaly detection on the generated reconstructions, use:

```
python anomaly_detection_threshold.py <train_predictions> <test_predictions> <anomaly_labels>
```

* <train_predictions>: Reconstructions used to train the anomaly threshold.

* <test_predictions>: Reconstructions used for anomaly detection.

* <anomaly_labels>: Ground truth labels for performance evaluation.