# Bio-inspired Evolving Connectivity for Recurrent Spiking Neural Networks

This repository is an advanced extension of the [Evolving Connectivity (EC)](https://arxiv.org/abs/2305.17650) framework. It integrates **biological constraints** from the mouse visual cortex (VISp) into Recurrent Spiking Neural Networks (RSNNs) to solve computer vision tasks (MNIST).

Key features include:
*   **Biological Initialization**: Network connectivity and neuron parameters ($\tau_m$) are initialized using biological data (Allen Institute / Blue Brain Project data).
*   **ConnSNN_Selected Architecture**: A novel readout mechanism that selects specific Layer 5 Excitatory (L5E) neurons as output nodes, mimicking biological functional columns, instead of using a fully connected readout layer.
*   **Poisson Encoding**: Converts static images into dynamic Poisson spike trains with input energy normalization.
*   **Optimized Training**: Implements "Synchronized Balanced Batch Training" to stabilize evolutionary gradients for stochastic SNNs.

## Getting Started

### 1. Prerequisites

Ensure you have a Python environment (Python 3.8+ recommended).

1.  **Install JAX**: Follow the [official guide](https://github.com/google/jax#installation) to install JAX with GPU support (highly recommended).
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install tensorflow tensorflow-datasets pandas
    ```
    *Note: `tensorflow` is required for efficient data loading and GPU memory management.*

3.  **WandB Setup**:
    Install and login to Weights & Biases to visualize training metrics (Fitness, Firing Rate, etc.).
    ```bash
    pip install wandb
    wandb login
    ```

### 2. Data Preparation

To run the bio-inspired experiments, you must place the pre-processed biological data files in the root directory:

*   `neuron_physics.npz`: Contains neuron physical parameters (Tau, etc.) and metadata.
*   `init_probability.npy`: The biological connection probability matrix.
*   `../dataset/mice_unnamed/neurons.csv.gz`: (Path configurable in code) Metadata for L5E neuron selection.

## Usage

This project focuses on two main scripts for training RSNNs on MNIST.

### 1. 2-Class Classification (Debug & Fast Validation)

`ec_2class.py` is designed for rapid experimentation on a subset of MNIST (digits 0 vs 1). It includes detailed diagnostic probes (Logits, Firing Rates) printed to the console.

**Run command:**
```bash
python ec_2class.py
```

*   **Default Configuration**:
    *   Pop Size: 1048
    *   Generations: 500
    *   Physics: $K_{in}=2.0, K_{h}=0.1, K_{out}=20.0$
    *   Input: 200 Hz Poisson spikes

### 2. 10-Class Classification (Full Training)

`ec.py` is the main entry point for the full MNIST 10-digit classification task. It uses the `ConnSNN_Selected` architecture with 10 specific L5E readout neurons.

**Run command:**
```bash
python ec.py
```

**Customizing Parameters (CLI):**
You can override any configuration parameter from the command line using OmegaConf syntax.

*   **Adjusting Biological Prior:**
    *   `use_bio_probability=True`: Use biological connection matrix.
    *   `bio_prob_mix_factor=1。0`:  Set to `1.0` for pure bio, `0.0` for pure random.

*   **Adjusting Physics Dynamics:**
    *   `network_conf.K_in=2.0`: Input current gain.
    *   `network_conf.K_h=0.1`: Recurrent weight gain (Inhibition/Excitation balance).
    *   `network_conf.K_out=20.0`: Output scaling factor.

**Example: Running a Pure Random Baseline**
```bash
python ec.py use_bio_probability=False run_name="Random_Baseline"
```

**Example: Running Strong Bio-Prior with tuned dynamics**
```bash
python ec.py bio_prob_mix_factor=0.8 network_conf.K_in=3.0 network_conf.K_h=0.1
```

## Key Modules

*   `ec.py`: Main training loop implementing Evolution Strategies (NES) with balanced mini-batches.
*   `networks/conn_snn.py`: Contains `ConnSNN_Selected`. Implements LIF dynamics, Dale's Law, and selected neuron readout.
*   `envs/mnist_env.py`: Custom JAX-compatible environment that converts MNIST images into time-tensorized Poisson spike trains.

## Citation

Original paper:
```
@inproceedings{wang2023evolving,
    title={Evolving Connectivity for Recurrent Spiking Neural Networks},
    author={Wang, Guan and Sun, Yuhao and Cheng, Sijie and Song, Sen},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=30o4ARmfC3}
}
```

## License

This project is licensed under the Apache License 2.0.