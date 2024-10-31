# BenchDaddi GPU - Benchmarking Suite

## Project Overview

This project aims to create a free and open-source benchmark suite for evaluating GPU performance on AI tasks. The current version supports **standard deep learning tests** across three major architectures: **Transformers (BERT)**, **RNNs (LSTM)**, and **CNNs (ResNet50)**. It also includes **tests for Ollama model variants during inference**. The benchmark suite is configurable, allowing users to test different model parameters and GPU configurations, with plans to expand coverage to additional architectures and models in future updates.

### Standard Deep Learning (DL) Tests (./standard_dl_test)

[Link To YouTube Video Explanation of Standard DL Test](https://youtu.be/aCRgkRWY4gw)

### Ollama Model Inference Tests (./ollama_test)

[Link To YouTube Video Explanation of Ollama Inference Tests](https://youtu.be/1kJczOUZXcs)


## Repository Structure

- `standard_dl_test`
   - `config`
      - `bert.yaml`: Configuration file for the BERT model.
      - `lstm.yaml`: Configuration file for the LSTM model.
      - `resnet50.yaml`: Configuration file for the ResNet50 model.
   - `bert_train_test.py`: Benchmark script for evaluating GPUs on the BERT model with Hugging Face's publicly available [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) for text classification.
   - `lstm_train_test.py`: Benchmark script for evaluating GPUs on the LSTM model with UCI's publicly available [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) for time series forecasting.
   - `resnet50_train_test.py`: Benchmark script for evaluating GPUs on the ResNet50 model with PyTorch's available [CIFAR10 dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10) for image classification. The images are resized to 256x256 to make the benchmarking scenario more realistic.
   - `run.py`: Main script to run the standard DL test suite.
- `ollama_test`
   - `config.yaml`: Configuration file for for ollama test suite
   - `log_2_csv.py`: Converts the output in ./logs (created when ./test_all.py is run) to more easily ingestible CSV file.
   - `test_all.py`: This is the main script to run to test with Ollama. It automates running Docker containers for different GPU and model combinations, pulling models via Ollama's API, and ensuring they are accessible for prompt testing. It manages container lifecycle events, GPU selection, and volume handling based on a YAML configuration file.
   - `test_model_prompts.py`: This script loads test prompts from a YAML file, runs multiple inference tests on a specified LLM model via a local API, and logs key performance metrics such as total duration, tokens per second, and GPU utilization. The results are averaged across test runs, saved to uniquely named log files, and include example responses from the model for each prompt.
   - `test_prompts.yaml`: A YAML file containing all prompts to be tested. 
- `gpu_max_load.py`: Script to test GPU maximum load, useful for checking thermals or server noise at maximum loading.
- `requirements.txt`: List of required packages and dependencies.
- `setup.py`: Script for setting up the project environment. This script requires a version of Conda installed as it creates a Conda environment.

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher (3.10 recommended)
- CUDA compatible GPU(s)
- NVIDIA CUDA Toolkit and cuDNN
- Docker must be installed. Below are links to outline that process if needed.
   - Docker Desktop Installation Instructions: https://www.docker.com/get-started/
   - Docker Engine (No GUI) Installation Instructions: https://docs.docker.com/engine/install/ubuntu/

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

## Usage

### For Standard DL Test (./standard_dl_test)

#### Training and Testing

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

#### Main Script

- To execute all benchmarks and obtain overall benchmark scores:

   ```bash
   python run.py
   ```

### For Ollama Test (./ollama_test)

1. ```bash 
      cd ./ollama_test
   ``` 

1. Adjust `config.yaml` if needed

1. Add or remove prompts to test_prompts.yaml

1. ```bash 
      python test_all.py
   ``` 

## Dependencies

The project dependencies are listed in the `requirements.txt` file and can be installed using pip.

## GPU Evaluation Metrics

### For Standard DL Test (./standard_dl_test)

The default setting evaluates each GPU using FP32, 3 epochs, and reasonable configurations pertaining to each model.

#### Memory

- **Memory Allocated**: Reflects the memory currently in use by tensors, models, and other GPU data structures that are actively holding data.
- **Memory Reserved**: Includes both the memory currently allocated and additional memory that has been set aside for future allocations to avoid fragmentation and allocation overhead.

These metrics are measured just after the first model is loaded onto the GPU(s) and after every `batch_logging_output_inc` (defaults to 100) batches are processed.

#### Execution Time

Measures the time taken for each operation: batch, epoch, etc. This is averaged and reported for each GPU per epoch and globally averaged for training and testing.

#### Data Transfer Time

Measures the time taken to load data onto the GPU(s). This metric is measured for each batch, epoch, etc., and is also expressed as a percentage of total execution time to help identify bottlenecks.

#### Throughput

Indicates how many samples a GPU can process per second for a particular model, dataset, and task. A sample could be any piece of information ingested by the model, such as text tokens or images. The training and testing throughput are summed to provide the benchmark score for each model. The global score of GPU performance across all models is the sum of each model's benchmark score. These scores are recorded in the `./logs/results_*.log` file.

### For Ollama Test (./ollama_test)

The Ollama test evaluates each model's performance by focusing on the duration and efficiency of processing prompt tokens and generating responses. Each metric is measured in nanoseconds by Ollama as default and is converted to seconds by dividing by \( 10^9 \).

#### Total Duration

The overall time spent generating the full response from the model, including both the time spent evaluating the prompt and generating the response tokens.

#### Load Duration

The time taken to load the model into memory, which includes the initialization of all necessary weights and parameters.

#### Prompt Evaluation

- **Prompt Eval Count**: Represents the number of tokens in the input prompt that need to be evaluated before the model generates the response.

- **Prompt Eval Duration**: The time taken to evaluate the input prompt. This includes tokenizing the prompt and processing it through the model to understand the context.

#### Response Evaluation

- **Response Eval Count**: The number of tokens generated by the model as part of the response.

- **Response Eval Duration**: The time spent generating the response tokens. This measures the speed of the model during the generation phase.

#### Tokens per Second

This metric is calculated by dividing API response parameter `eval_count` by `eval_duration` and multiplying by \( 10^9 \), which provides the rate of token generation in tokens per second.
 
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

For any questions or issues, please contact [tdd@thedatadaddi.com](mailto:tdd@thedatadaddi.com).
