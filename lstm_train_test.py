"""
Description: This code sets up a distributed training and testing pipeline for a Long Short-Term Memory (LSTM) model
using PyTorch's Distributed Data Parallel (DDP). The model is designed to perform time series forecasting
using a dataset fetched from the UCI Machine Learning Repository. The code includes data loading with 
a custom TimeSeriesDataset class, model definition, and training and testing functions. The training 
process utilizes AdamW optimizer, mixed precision for efficiency, and logs various metrics, including 
execution times and memory usage. The main function initializes the distributed setup and spawns processes 
for each GPU specified in the configuration file.

Author: TheDataDaddi
Date: 2024-02-02
Version: 1.0
License: MIT License
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.nn import MSELoss, LSTM, Linear
import pandas as pd
import time
import os
import logging
from datetime import datetime
import numpy as np
import yaml
from ucimlrepo import fetch_ucirepo
from torch.cuda.amp import GradScaler, autocast

# Define a custom dataset class for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length, 0]  # Assuming the first column is the target variable
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Function to load configuration from associated YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Function to set up logging
def setup_logging(log_directory, current_time):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_filename = f"{log_directory}/lstm_tt_{current_time}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Setup devices and log device information
def setup_and_log_devices(gpu_ids, local_rank):
    if not torch.cuda.is_available():
        logging.info(f"[{device.type.upper()} {local_rank}] CUDA is not available. Using CPU.")
        print(f"[{device.type.upper()} {local_rank}] CUDA is not available. Using CPU.")
        return torch.device("cpu")
    
    if -1 in gpu_ids:
        print("Using CPU by choice.")
        return torch.device("cpu")

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    prop = torch.cuda.get_device_properties(device)
    
    logging.info(f"[{device.type.upper()} {local_rank}] CUDA is available: Version {torch.version.cuda}")
    print(f"[{device.type.upper()} {local_rank}] CUDA is available: Version {torch.version.cuda}")    
    device_info = f"[{device.type.upper()} {local_rank}] Selected Device ID {device.index}: {prop.name}, Compute Capability: {prop.major}.{prop.minor}, Total Memory: {prop.total_memory / 1e9:.2f} GB"
    logging.info(device_info)
    print(device_info)

    return device

# Function to log memory usage
def log_memory_usage(device, local_rank):
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logging.info(f"[{device.type.upper()} {local_rank}] Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")
    else:
        logging.info(f"[{device.type.upper()} {local_rank}] CPU mode, no GPU device memory to log.")

# Function to load data and create data loaders
def load_data(data_directory, seq_length, batch_size, num_workers, local_rank):
    df = fetch_ucirepo(id=235).data.original
    df.replace('?', np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df = df.astype(float)

    data = df.values
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_dataset = TimeSeriesDataset(train_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)

    if torch.cuda.is_available():
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    
    return train_loader, test_loader

# Define the LSTM model class
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))
        output = self.fc(output[:, -1, :])
        return output

# Function to initialize the LSTM model with DDP if cuda is avaliable
def initialize_model(input_size, hidden_size, output_size, num_layers, local_rank, device):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    
    if device.type == 'cuda':
        model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model.to(device)
        
    return model

# Function to train the model
def train_model(model, train_loader, epochs, learning_rate, batch_logging_output_inc, device, local_rank, use_mixed_precision):
    if device.index == 0 or device.type == 'cpu':  # Log headers only once from the main device
        logging.info(f'###############################################################################')
        logging.info(f'TRAINING')
        logging.info(f'###############################################################################')

    criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=use_mixed_precision)

    model.train()

    avg_batch_exec_time = 0
    avg_batch_data_transfer_time = 0
    total_batches = 0
    total_samples = 0
    start_time = time.time()

    for epoch in range(epochs):
        batch_time_list = []
        batch_data_transfer_time_list = []
        epoch_start_time = time.time()

        for step, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()

            data_transfer_start_time = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += inputs.size(0)
            data_transfer_time = time.time() - data_transfer_start_time
            batch_data_transfer_time_list.append(data_transfer_time)

            optimizer.zero_grad()
            with autocast(enabled=use_mixed_precision):
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze(-1)  # Squeeze the last dimension
                loss = criterion(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time = time.time() - batch_start_time
            batch_time_list.append(batch_time)

            total_batches += 1

            batch_loss = loss.item()

            if (step + 1) % batch_logging_output_inc == 0:
                logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1}, Batch {step + 1}/{len(train_loader)}, Loss: {batch_loss:.3f}, Data Transfer Time: {data_transfer_time:.4f} seconds, Batch Exec Time: {batch_time:.3f} seconds')
                log_memory_usage(device, local_rank)

        avg_ep_batch_exec_time = np.mean(batch_time_list)
        avg_ep_batch_data_transfer_time = np.mean(batch_data_transfer_time_list)
        epoch_time = time.time() - epoch_start_time

        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> Average Batch Exec Time {avg_ep_batch_exec_time:.3f} seconds')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> Average Data Transfer Time {avg_ep_batch_data_transfer_time:.3f} seconds')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> % Average Data Transfer Time of Batch Execution Time {avg_ep_batch_data_transfer_time/avg_ep_batch_exec_time*100 :.3f} %')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> completed in {epoch_time:.3f} seconds')

        avg_batch_exec_time += avg_ep_batch_exec_time
        avg_batch_data_transfer_time += avg_ep_batch_data_transfer_time

    total_training_time = time.time() - start_time

    # Aggregate global metrics across all devices
    global_avg_batch_exec_time = torch.tensor([avg_batch_exec_time], dtype=torch.float64, device=device)
    global_avg_batch_data_transfer_time = torch.tensor([avg_batch_data_transfer_time], dtype=torch.float64, device=device)
    global_total_training_time = torch.tensor([total_training_time], dtype=torch.float64, device=device)
    global_total_samples = torch.tensor([total_samples], dtype=torch.float64, device=device)

    if device.type == 'cpu':
        logging.info(f'###############################################################################')
        logging.info(f'GLOBAL TRAINING METRICS')
        logging.info(f'Total Time: {global_total_training_time.item():.3f} seconds')
        logging.info(f'Average Batch Execution Time: {global_avg_batch_exec_time.item():.3f} seconds')
        logging.info(f'Average Batch Data Transfer Time: {global_avg_batch_data_transfer_time.item() / (epochs):.3f} seconds')
        logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {(global_avg_batch_data_transfer_time.item() / global_avg_batch_exec_time.item()) * 100:.3f} %')
        logging.info(f'Global Training Throughput: {global_total_samples.item() / global_total_training_time.item():.3f} samples/second')

    else:
        dist.reduce(global_avg_batch_exec_time, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(global_avg_batch_data_transfer_time, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(global_total_training_time, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(global_total_samples, dst=0, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            time.sleep(1)
            world_size = dist.get_world_size()
            logging.info(f'###############################################################################')
            logging.info(f'GLOBAL TRAINING METRICS')
            logging.info(f'Total Time: {global_total_training_time.item() / world_size:.3f} seconds')
            logging.info(f'Average Batch Execution Time: {global_avg_batch_exec_time.item() / (epochs * world_size):.3f} seconds')
            logging.info(f'Average Batch Data Transfer Time: {global_avg_batch_data_transfer_time.item() / (epochs * world_size):.3f} seconds')
            logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {(global_avg_batch_data_transfer_time.item() / global_avg_batch_exec_time.item()) * 100:.3f} %')
            logging.info(f'Global Training Throughput: {global_total_samples.item() / (global_total_training_time.item() / world_size):.3f} samples/second')

# Function to test the model
def test_model(model, test_loader, batch_logging_output_inc, device, local_rank):
    if device.index == 0 or device.type == 'cpu':  # Log headers only once from the main device
        logging.info(f'###############################################################################')
        logging.info(f'TESTING')
        logging.info(f'###############################################################################')
    
    total_test_time = 0
    total_samples = 0
    num_batches = len(test_loader)

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)  # Squeeze the last dimension
            torch.cuda.synchronize()

            batch_time = time.time() - start_time
            total_test_time += batch_time
            total_samples += inputs.size(0)

            if (i + 1) % batch_logging_output_inc == 0:
                logging.info(f'[{device.type.upper()} {local_rank}] Batch {i + 1}/{num_batches}, Batch Exec Time: {batch_time:.3f} seconds')

    logging.info(f'[{device.type.upper()} {local_rank}] Total Time: {total_test_time:.3f} seconds')
    logging.info(f'[{device.type.upper()} {local_rank}] Average Batch Execution Time: {total_test_time/num_batches:.3f} seconds')
    logging.info(f'[{device.type.upper()} {local_rank}] Throughput: {total_samples/total_test_time:.3f} samples/second')

    # Aggregate global metrics across all devices
    global_total_test_time = torch.tensor([total_test_time], dtype=torch.float64, device=device)
    global_total_samples = torch.tensor([total_samples], dtype=torch.float64, device=device)

    if device.type == 'cpu':
            logging.info(f'###############################################################################')
            logging.info(f'GLOBAL TESTING METRICS')
            logging.info(f'Total Time: {global_total_test_time.item():.3f} seconds')
            logging.info(f'Average Batch Execution Time: {global_total_test_time.item() / (num_batches):.3f} seconds')
            logging.info(f'Global Testing Throughput: {global_total_samples.item() / global_total_test_time.item():.3f} samples/second')

    else:
        dist.reduce(global_total_test_time, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(global_total_samples, dst=0, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            time.sleep(1)
            world_size = dist.get_world_size()
            logging.info(f'###############################################################################')
            logging.info(f'GLOBAL TESTING METRICS')
            logging.info(f'Total Time: {global_total_test_time.item() / world_size:.3f} seconds')
            logging.info(f'Average Batch Execution Time: {global_total_test_time.item() / (num_batches * world_size):.3f} seconds')
            logging.info(f'Global Testing Throughput: {global_total_samples.item() / (global_total_test_time.item() / world_size):.3f} samples/second')

# Function to run the main training and testing workflow
def main_worker(local_rank, config):
    # Set the necessary environment variables
    os.environ['MASTER_ADDR'] = config['master_address']
    os.environ['MASTER_PORT'] = config['master_port']

    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(config['gpu_ids']), rank=local_rank)
    setup_logging('./logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    device = setup_and_log_devices(config['gpu_ids'], local_rank)
    print(f'[{device.type.upper()} {local_rank}] Setting up devices and loading data')
    
    train_loader, test_loader = load_data(
        './data', config['seq_length'], config['batch_size'], config['num_workers'], local_rank
    )

    print(f'[{device.type.upper()} {local_rank}] Initializing model')
    model = initialize_model(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers'], local_rank, device)

    log_memory_usage(device, local_rank)
    time.sleep(1)

    batch_logging_output_inc = config['batch_logging_output_inc']
    use_mixed_precision = config.get('use_mixed_precision', False)

    print(f'[{device.type.upper()} {local_rank}] Starting training')
    train_model(model, train_loader, config['epochs'], config['learning_rate'], batch_logging_output_inc, device, local_rank, use_mixed_precision)

    print(f'[{device.type.upper()} {local_rank}] Starting testing')
    test_model(model, test_loader, batch_logging_output_inc, device, local_rank)

    print(f'[{device.type.upper()} {local_rank}] Benchmarking complete')

# Main function to load configuration and start the main worker processes
def main(config_path):
    config = load_config(config_path)
    gpu_ids = config['gpu_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    if torch.cuda.is_available():
        mp.spawn(main_worker, args=(config,), nprocs=len(gpu_ids), join=True)
    else:
        main_worker(0, config)
    
if __name__ == "__main__":
    main("./config/lstm.yaml")
