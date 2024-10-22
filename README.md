# BenchDaddi GPU - Benchmarking Suite

## Project Overview

This project aims to create a free and open-source benchmark suite for evaluating GPU performance on AI tasks. The current version supports deep learning tests across three major architectures: **Transformers (BERT)**, **RNNs (LSTM)**, and **CNNs (ResNet50)**. It also includes tests for **Ollama model variants** during inference. The benchmark suite is configurable, allowing users to test different model parameters and GPU configurations, with plans to expand coverage to additional architectures and models in future updates.

### Deep Learning Tests
Each GPU is evaluated across the following metrics for both training and inference:
- **Throughput**: Number of samples processed per second.
- **Execution Time**: Time taken for one training or inference iteration.
- **Data Transfer Time**: Time spent transferring data between the host and the GPU.
- **Memory Usage**: GPU memory consumption during training and inference.

### Ollama Model Inference Tests
For Ollama models, GPU performance is measured based on:
- **Total Duration**: Overall time spent generating a response.
- **Load Duration**: Time spent loading the model.
- **Prompt Evaluation Duration**: Time taken to evaluate the input prompt.
- **Response Evaluation Duration**: Time spent generating the response.
- **Tokens per Second**: Speed of response generation, calculated as `eval_count / eval_duration * 10^9` (tokens per second).


This benchmark suite will eventually support all major datasets and AI models, allowing comprehensive performance comparisons across a variety of AI tasks and GPU configurations.

[Link To YouTube Video Explanation](https://youtu.be/aCRgkRWY4gw)

## Repository Structure

- `bert_train_test.py`: Benchmark script for evaluating GPUs on the BERT model with Hugging Face's publicly available [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) for text classification.
- `lstm_train_test.py`: Benchmark script for evaluating GPUs on the LSTM model with UCI's publicly available [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) for time series forecasting.
- `resnet50_train_test.py`: Benchmark script for evaluating GPUs on the ResNet50 model with PyTorch's available [CIFAR10 dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10) for image classification. The images are resized to 256x256 to make the benchmarking scenario more realistic.
- `gpu_max_load.py`: Script to test GPU maximum load, useful for checking thermals or server noise at maximum loading.
- `run.py`: Main script to run the full benchmarking suite.
- `setup.py`: Script for setting up the project environment. This script requires a version of Conda installed as it creates a Conda environment.
- `requirements.txt`: List of required packages and dependencies.
- `bert.yaml`: Configuration file for the BERT model.
- `lstm.yaml`: Configuration file for the LSTM model.
- `resnet50.yaml`: Configuration file for the ResNet50 model.

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher (3.10 recommended)
- CUDA compatible GPU(s)
- NVIDIA CUDA Toolkit and cuDNN

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:thedatadaddi/BenchDaddi.git
   cd BenchDaddi
   ```

2. To use Conda, install Miniconda or Anaconda. [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).
   ```bash
   python setup.py
   conda activate gpu_bm_venv
   ```

3. Alternatively, create a virtual environment with `venv` or `virtualenv` and install the required packages:
   ```bash
   python3 -m venv gpu_bm_venv
   # Windows
   gpu_bm_venv\Scripts\activate
   # Linux or Mac
   source gpu_bm_venv/bin/activate
   pip install -r requirements.txt
   ```

### Configuration

Configuration files for each model are provided in YAML format. These files include settings for logging, GPU usage, hyperparameters, and data loading options. Note that mixed precision training is available with the option `use_mixed_precision=True`. The default setting is `False`, which will train with FP32.

- `bert.yaml`:
  ```yaml
  # Logging related
  batch_logging_output_inc: 100

  # GPU related
  gpu_ids: [0]  # [-1] will set CPU for use
  master_address: 'localhost'
  master_port: '65531'
  use_mixed_precision: False

  # Hyperparameters
  epochs: 3
  learning_rate: 0.00002

  # Dataset & dataloader related
  max_length: 512
  batch_size: 16
  num_labels: 2
  num_workers: 4
  ```

- `lstm.yaml`:
  ```yaml
  # Logging related
  batch_logging_output_inc: 100

  # GPU related
  gpu_ids: [0]  # [-1] will set CPU for use
  master_address: 'localhost'
  master_port: '65531'
  use_mixed_precision: False

  # Hyperparameters
  epochs: 3
  learning_rate: 0.001
  input_size: 7
  hidden_size: 50
  output_size: 1
  num_layers: 1
  seq_length: 10

  # Dataset & dataloader related
  batch_size: 64
  num_workers: 4
  ```

- `resnet50.yaml`:
  ```yaml
  # Logging related
  batch_logging_output_inc: 100

  # GPU related
  gpu_ids: [0]  # [-1] will set CPU for use
  master_address: 'localhost'
  master_port: '65531'
  use_mixed_precision: False

  # Hyperparameters
  epochs: 3
  learning_rate: 0.0001

  # Dataset & dataloader related
  num_workers: 4
  batch_size: 64
  num_classes: 10
  ```

## Usage

### Training and Testing

1. To benchmark with the BERT model:
   ```bash
   python bert_train_test.py
   ```

2. To benchmark with the LSTM model:
   ```bash
   python lstm_train_test.py
   ```

3. To benchmark with the ResNet50 model:
   ```bash
   python resnet50_train_test.py
   ```

### GPU Load Test

To test the maximum load on the GPU:
```bash
python gpu_max_load.py
```

### Main Script

To execute all benchmarks and obtain overall benchmark scores:
```bash
python run.py
```

## Dependencies

The project dependencies are listed in the `requirements.txt` file and can be installed using pip.

## GPU Evaluation Metrics

The default setting evaluates each GPU using FP32, 3 epochs, and reasonable configurations pertaining to each model.

### Memory

- **Memory Allocated**: Reflects the memory currently in use by tensors, models, and other GPU data structures that are actively holding data.
- **Memory Reserved**: Includes both the memory currently allocated and additional memory that has been set aside for future allocations to avoid fragmentation and allocation overhead.

These metrics are measured just after the first model is loaded onto the GPU(s) and after every `batch_logging_output_inc` (defaults to 100) batches are processed.

### Execution Time

Measures the time taken for each operation: batch, epoch, etc. This is averaged and reported for each GPU per epoch and globally averaged for training and testing.

### Data Transfer Time

Measures the time taken to load data onto the GPU(s). This metric is measured for each batch, epoch, etc., and is also expressed as a percentage of total execution time to help identify bottlenecks.

### Throughput

Indicates how many samples a GPU can process per second for a particular model, dataset, and task. A sample could be any piece of information ingested by the model, such as text tokens or images. The training and testing throughput are summed to provide the benchmark score for each model. The global score of GPU performance across all models is the sum of each model's benchmark score. These scores are recorded in the `./logs/results_*.log` file.

## License

This project is licensed under the MIT License. See the LICENSE.txt file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## Citation

If you use this benchmark in your research, please cite the following:

```plaintext
@misc{TDD GPU Benchmark Suite,
  author = {TheDataDaddi},
  title = {TDD GPU Benchmark Suite},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thedatadaddi/gpu_bm_suite.git}},
}
```

## Contact

For any questions or issues, please contact [skingutube22@gmail.com](mailto:skingutube22@gmail.com).
