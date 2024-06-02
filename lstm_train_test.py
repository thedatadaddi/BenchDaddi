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

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_directory, current_time):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_filename = f"{log_directory}/lstm_tt_{current_time}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def setup_and_log_devices(gpu_ids, local_rank):
    if not torch.cuda.is_available():
        logging.info(f"[GPU {local_rank}] CUDA is not available. Using CPU.")
        print(f"[GPU {local_rank}] CUDA is not available. Using CPU.")
        return torch.device("cpu")

    logging.info(f"[GPU {local_rank}] CUDA is available: Version {torch.version.cuda}")
    print(f"[GPU {local_rank}] CUDA is available: Version {torch.version.cuda}")

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    prop = torch.cuda.get_device_properties(device)
    device_info = f"[GPU {local_rank}] Selected Device ID {device.index}: {prop.name}, Compute Capability: {prop.major}.{prop.minor}, Total Memory: {prop.total_memory / 1e9:.2f} GB"
    logging.info(device_info)
    print(device_info)

    return device

def log_memory_usage(device, local_rank):
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logging.info(f"[GPU {local_rank}] Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")
    else:
        logging.info(f"[GPU {local_rank}] CPU mode, no GPU device memory to log.")

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

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=local_rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    
    return train_loader, test_loader

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

def initialize_model(input_size, hidden_size, output_size, num_layers, local_rank):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    return model

def train_model(model, train_loader, epochs, learning_rate, batch_logging_output_inc, device, local_rank, use_mixed_precision):
    if device.index == 0:  # Log headers only once from the main GPU
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
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
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
                logging.info(f'[GPU {local_rank}] Epoch {epoch + 1}, Batch {step + 1}/{len(train_loader)}, Loss: {batch_loss:.3f}, Data Transfer Time: {data_transfer_time:.4f} seconds, Batch Exec Time: {batch_time:.3f} seconds')
                log_memory_usage(device, local_rank)

        avg_ep_batch_exec_time = np.mean(batch_time_list)
        avg_ep_batch_data_transfer_time = np.mean(batch_data_transfer_time_list)
        epoch_time = time.time() - epoch_start_time

        logging.info(f'[GPU {local_rank}] Epoch {epoch + 1} -> Average Batch Exec Time {avg_ep_batch_exec_time:.3f} seconds')
        logging.info(f'[GPU {local_rank}] Epoch {epoch + 1} -> Average Data Transfer Time {avg_ep_batch_data_transfer_time:.3f} seconds')
        logging.info(f'[GPU {local_rank}] Epoch {epoch + 1} -> % Average Data Transfer Time of Batch Execution Time {avg_ep_batch_data_transfer_time/avg_ep_batch_exec_time*100 :.3f} %')
        logging.info(f'[GPU {local_rank}] Epoch {epoch + 1} -> completed in {epoch_time:.3f} seconds')

        avg_batch_exec_time += avg_ep_batch_exec_time
        avg_batch_data_transfer_time += avg_ep_batch_data_transfer_time

    total_training_time = time.time() - start_time

    # Aggregate global metrics across all GPUs
    global_avg_batch_exec_time = torch.tensor([avg_batch_exec_time], dtype=torch.float64, device=device)
    global_avg_batch_data_transfer_time = torch.tensor([avg_batch_data_transfer_time], dtype=torch.float64, device=device)
    global_total_training_time = torch.tensor([total_training_time], dtype=torch.float64, device=device)
    global_total_samples = torch.tensor([total_samples], dtype=torch.float64, device=device)

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
        logging.info(f'Global Training Throughput: {global_total_samples.item() / global_total_training_time.item():.3f} samples/second')

def test_model(model, test_loader, batch_logging_output_inc, device, local_rank):
    if device.index == 0:  # Log headers only once from the main GPU
        logging.info(f'###############################################################################')
        logging.info(f'TESTING')
        logging.info(f'###############################################################################')
    
    total_test_time = 0
    total_samples = 0
    num_batches = len(test_loader)

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            start_time = time.time()
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)  # Squeeze the last dimension
            torch.cuda.synchronize()

            batch_time = time.time() - start_time
            total_test_time += batch_time
            total_samples += inputs.size(0)

            if (i + 1) % batch_logging_output_inc == 0:
                logging.info(f'[GPU {local_rank}] Batch {i + 1}/{num_batches}, Batch Exec Time: {batch_time:.3f} seconds')

    logging.info(f'[GPU {local_rank}] Total Time: {total_test_time:.3f} seconds')
    logging.info(f'[GPU {local_rank}] Average Batch Execution Time: {total_test_time/num_batches:.3f} seconds')
    logging.info(f'[GPU {local_rank}] Throughput: {total_samples/total_test_time:.3f} samples/second')

    # Aggregate global metrics across all GPUs
    global_total_test_time = torch.tensor([total_test_time], dtype=torch.float64, device=device)
    global_total_samples = torch.tensor([total_samples], dtype=torch.float64, device=device)

    dist.reduce(global_total_test_time, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(global_total_samples, dst=0, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        time.sleep(1)
        world_size = dist.get_world_size()
        logging.info(f'###############################################################################')
        logging.info(f'GLOBAL TESTING METRICS')
        logging.info(f'Total Time: {global_total_test_time.item() / world_size:.3f} seconds')
        logging.info(f'Average Batch Execution Time: {global_total_test_time.item() / (num_batches * world_size):.3f} seconds')
        logging.info(f'Global Testing Throughput: {global_total_samples.item() / global_total_test_time.item():.3f} samples/second')

def main_worker(local_rank, config):
    # Set the necessary environment variables
    os.environ['MASTER_ADDR'] = config['master_address']
    os.environ['MASTER_PORT'] = config['master_port']

    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(config['gpu_ids']), rank=local_rank)
    setup_logging('./logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print(f'[GPU {local_rank}] Setting up devices and loading data')
    device = setup_and_log_devices(config['gpu_ids'], local_rank)

    train_loader, test_loader = load_data(
        './data', config['seq_length'], config['batch_size'], config['num_workers'], local_rank
    )

    print(f'[GPU {local_rank}] Initializing model')
    model = initialize_model(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers'], local_rank)

    log_memory_usage(device, local_rank)

    batch_logging_output_inc = config['batch_logging_output_inc']
    use_mixed_precision = config.get('use_mixed_precision', False)

    print(f'[GPU {local_rank}] Starting training')
    train_model(model, train_loader, config['epochs'], config['learning_rate'], batch_logging_output_inc, device, local_rank, use_mixed_precision)

    print(f'[GPU {local_rank}] Starting testing')
    test_model(model, test_loader, batch_logging_output_inc, device, local_rank)

    print(f'[GPU {local_rank}] Benchmarking complete')

def main(config_path):
    config = load_config(config_path)
    gpu_ids = config['gpu_ids']
    
    mp.spawn(main_worker, args=(config,), nprocs=len(gpu_ids), join=True)

if __name__ == "__main__":
    main("./config/lstm.yaml")
