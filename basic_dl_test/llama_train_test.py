import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import LLaMATokenizer, LLaMAForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
import time
import os
import logging
from datetime import datetime
import numpy as np
import yaml
from torch.cuda.amp import GradScaler, autocast

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_directory, current_time):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_filename = f"{log_directory}/llama_tt_{current_time}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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

def log_memory_usage(device, local_rank):
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logging.info(f"[{device.type.upper()} {local_rank}] Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")
    else:
        logging.info(f"[{device.type.upper()} {local_rank}] CPU mode, no GPU device memory to log.")

def load_data(tokenizer, data_directory, max_length, batch_size, num_workers, local_rank):
    dataset = load_dataset('imdb', cache_dir=data_directory)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    if torch.cuda.is_available():
        train_sampler = DistributedSampler(tokenized_datasets['train'], num_replicas=dist.get_world_size(), rank=local_rank)
        test_sampler = DistributedSampler(tokenized_datasets['test'], num_replicas=dist.get_world_size(), rank=local_rank)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    
    return train_loader, test_loader

def initialize_model(model_name, num_labels, local_rank, device):
    model = LLaMAForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    if device.type == 'cuda':
        model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model.to(device)
        
    return model

def train_model(model, train_loader, epochs, learning_rate, batch_logging_output_inc, device, local_rank, use_mixed_precision):
    time.sleep(1)
    if device.index == 0 or device.type == 'cpu': 
        logging.info(f'###############################################################################')
        logging.info(f'TRAINING')
        logging.info(f'###############################################################################')

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    scaler = GradScaler(enabled=use_mixed_precision)

    model.train()

    ep_batch_exec_time_list = []
    ep_batch_data_transfer_time_list = []
    total_batches = 0
    total_samples = 0
    start_time = time.time()

    for epoch in range(epochs):
        batch_time_list = []
        batch_data_transfer_time_list = []
        total_samples_epoch = 0
        epoch_start_time = time.time()

        for step, batch in enumerate(train_loader):
            batch_start_time = time.time()

            data_transfer_start_time = time.time()
            inputs = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            data_transfer_time = time.time() - data_transfer_start_time
            batch_data_transfer_time_list.append(data_transfer_time)
            
            total_samples += inputs.size(0)
            total_samples_epoch += inputs.size(0)

            optimizer.zero_grad()
            
            with autocast(enabled=use_mixed_precision):
                outputs = model(inputs, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_time = time.time() - batch_start_time
            batch_time_list.append(batch_time)

            total_batches += 1

            batch_loss = loss.item()

            if (step + 1) % batch_logging_output_inc == 0:
                logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1}, Batch {step + 1}/{len(train_loader)}, Loss: {batch_loss:.4f}, Data Transfer Time: {data_transfer_time:.4f} seconds, Batch Execution Time: {batch_time:.4f} seconds')
                log_memory_usage(device, local_rank)

        avg_ep_batch_exec_time = np.mean(batch_time_list)
        avg_ep_batch_data_transfer_time = np.mean(batch_data_transfer_time_list)
        epoch_time = time.time() - epoch_start_time

        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> Average Batch Execution Time {avg_ep_batch_exec_time:.4f} seconds')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> Average Data Transfer Time {avg_ep_batch_data_transfer_time:.4f} seconds')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> % Average Data Transfer Time of Batch Execution Time {avg_ep_batch_data_transfer_time/avg_ep_batch_exec_time*100 :.4f} %')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> Total Execution Time {epoch_time:.4f} seconds')
        logging.info(f'[{device.type.upper()} {local_rank}] Epoch {epoch + 1} -> Throughput: {total_samples_epoch/epoch_time:.4f} samples/second')

        ep_batch_exec_time_list.append(avg_ep_batch_exec_time)
        ep_batch_data_transfer_time_list.append(avg_ep_batch_data_transfer_time)
        
    total_training_time = time.time() - start_time
    
    gpu_avg_batch_exec_time = np.mean(ep_batch_exec_time_list)
    gpu_avg_batch_data_transfer_time = np.mean(ep_batch_data_transfer_time_list)

    global_avg_batch_exec_time = torch.tensor([gpu_avg_batch_exec_time], dtype=torch.float64, device=device)
    global_avg_batch_data_transfer_time = torch.tensor([gpu_avg_batch_data_transfer_time], dtype=torch.float64, device=device)
    global_avg_training_time = torch.tensor([total_training_time], dtype=torch.float64, device=device)
    global_max_training_time = torch.tensor([total_training_time], dtype=torch.float64, device=device)
    global_total_samples = torch.tensor([total_samples], dtype=torch.float64, device=device)

    if device.type == 'cpu':
        logging.info(f'###############################################################################')
        logging.info(f'GLOBAL TRAINING METRICS')
        logging.info(f'Average Batch Execution Time: {global_avg_batch_exec_time.item():.4f} seconds')
        logging.info(f'Average Batch Data Transfer Time: {global_avg_batch_data_transfer_time.item():.4f} seconds')
        logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {(global_avg_batch_data_transfer_time.item() / global_avg_batch_exec_time.item()) * 100:.4f} %')
        logging.info(f'Total Execution Time: {global_avg_training_time.item():.4f} seconds')
        logging.info(f'Global Training Throughput: {global_total_samples.item() / global_max_training_time.item():.4f} samples/second')
    
    else:
        dist.reduce(global_avg_batch_exec_time, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(global_avg_batch_data_transfer_time, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(global_avg_training_time, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(global_max_training_time, dst=0, op=dist.ReduceOp.MAX)
        dist.reduce(global_total_samples, dst=0, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            time.sleep(1)
            logging.info(f'###############################################################################')
            logging.info(f'GLOBAL TRAINING METRICS')
            logging.info(f'Average Batch Execution Time: {global_avg_batch_exec_time.item():.4f} seconds')
            logging.info(f'Average Batch Data Transfer Time: {global_avg_batch_data_transfer_time.item():.4f} seconds')
            logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {(global_avg_batch_data_transfer_time.item() / global_avg_batch_exec_time.item()) * 100:.4f} %')
            logging.info(f'Total Execution Time: {global_avg_training_time.item():.4f} seconds')
            logging.info(f'Global Training Throughput: {global_total_samples.item() / global_max_training_time.item():.4f} samples/second')

def test_model(model, test_loader, batch_logging_output_inc, device, local_rank, use_mixed_precision):
    time.sleep(1)
    if device.index == 0 or device.type == 'cpu': 
        logging.info(f'###############################################################################')
        logging.info(f'TESTING')
        logging.info(f'###############################################################################')
    
    batch_test_time_list = []
    batch_data_transfer_time_list = []
    total_samples = 0
    num_batches = len(test_loader)

    model.eval()
    with autocast(enabled=use_mixed_precision), torch.no_grad():
        for i, batch in enumerate(test_loader):
            start_time = time.time()
            
            data_transfer_start_time = time.time()
            inputs = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            data_transfer_time = time.time() - data_transfer_start_time
            batch_data_transfer_time_list.append(data_transfer_time)
        
            outputs = model(inputs, attention_mask=attention_masks, labels=labels)
            torch.cuda.synchronize()

            batch_time = time.time() - start_time
            batch_test_time_list.append(batch_time)
            total_samples += inputs.size(0)

            if (i + 1) % batch_logging_output_inc == 0:
                logging.info(f'[{device.type.upper()} {local_rank}] Batch {i + 1}/{num_batches}, Data Transfer Time: {data_transfer_time:.4f} seconds, Batch Execution Time: {batch_time:.4f} seconds')
                
    avg_batch_data_transfer_time = np.mean(batch_data_transfer_time_list)
    avg_batch_exec_time = np.mean(batch_test_time_list)
    total_test_time = sum(batch_test_time_list)  

    logging.info(f'[{device.type.upper()} {local_rank}] Average Batch Execution Time: {total_test_time/num_batches:.4f} seconds')
    logging.info(f'[{device.type.upper()} {local_rank}] Average Data Transfer Time {avg_batch_data_transfer_time:.4f} seconds')
    logging.info(f'[{device.type.upper()} {local_rank}] % Average Data Transfer Time of Batch Execution Time {avg_batch_data_transfer_time/avg_batch_exec_time*100 :.4f} %')
    logging.info(f'[{device.type.upper()} {local_rank}] Total Execution Time: {total_test_time:.4f} seconds')
    logging.info(f'[{device.type.upper()} {local_rank}] Throughput: {total_samples/total_test_time:.4f} samples/second')

    global_avg_batch_exec_time = torch.tensor([avg_batch_exec_time], dtype=torch.float64, device=device)
    global_avg_batch_data_transfer_time = torch.tensor([avg_batch_data_transfer_time], dtype=torch.float64, device=device)    
    global_avg_test_time = torch.tensor([total_test_time], dtype=torch.float64, device=device)
    global_max_test_time = torch.tensor([total_test_time], dtype=torch.float64, device=device)
    global_total_samples = torch.tensor([total_samples], dtype=torch.float64, device=device)
    
    if device.type == 'cpu':
        logging.info(f'###############################################################################')
        logging.info(f'GLOBAL TESTING METRICS')
        logging.info(f'Average Batch Execution Time: {global_avg_batch_exec_time.item():.4f} seconds')
        logging.info(f'Average Batch Data Transfer Time: {global_avg_batch_data_transfer_time.item():.4f} seconds')
        logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {(global_avg_batch_data_transfer_time.item() / global_avg_batch_exec_time.item()) * 100:.4f} %')
        logging.info(f'Total Execution Time: {global_avg_test_time.item():.4f} seconds')
        logging.info(f'Global Testing Throughput: {global_total_samples.item() / global_max_test_time.item():.4f} samples/second')

    else:
        dist.reduce(global_avg_batch_exec_time, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(global_avg_batch_data_transfer_time, dst=0, op=dist.ReduceOp.AVG)        
        dist.reduce(global_avg_test_time, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(global_max_test_time, dst=0, op=dist.ReduceOp.MAX)
        dist.reduce(global_total_samples, dst=0, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            time.sleep(1)
            logging.info(f'###############################################################################')
            logging.info(f'GLOBAL TESTING METRICS')
            logging.info(f'Average Batch Execution Time: {global_avg_batch_exec_time.item():.4f} seconds')
            logging.info(f'Average Batch Data Transfer Time: {global_avg_batch_data_transfer_time.item():.4f} seconds')
            logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {(global_avg_batch_data_transfer_time.item() / global_avg_batch_exec_time.item()) * 100:.4f} %')
            logging.info(f'Total Execution Time: {global_avg_test_time.item():.4f} seconds')
            logging.info(f'Global Testing Throughput: {global_total_samples.item() / global_max_test_time.item():.4f} samples/second')

def main_worker(local_rank, config):
    os.environ['MASTER_ADDR'] = config['master_address']
    os.environ['MASTER_PORT'] = config['master_port']

    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(config['gpu_ids']), rank=local_rank)
    setup_logging('./logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print(f'Setting up devices and loading data')
    device = setup_and_log_devices(config['gpu_ids'], local_rank)

    tokenizer = LLaMATokenizer.from_pretrained(config['model_name'])
    train_loader, test_loader = load_data(
        tokenizer, './data', config['max_length'], config['batch_size'], config['num_workers'], local_rank
    )

    print(f'[{device.type.upper()} {local_rank}] Initializing model')
    model = initialize_model(config['model_name'], config['num_labels'], local_rank, device)
    
    log_memory_usage(device, local_rank)

    batch_logging_output_inc = config['batch_logging_output_inc']
    use_mixed_precision = config.get('use_mixed_precision', False)

    print(f'[{device.type.upper()} {local_rank}] Starting training')
    train_model(model, train_loader, config['epochs'], config['learning_rate'], batch_logging_output_inc, device, local_rank, use_mixed_precision)

    print(f'[{device.type.upper()} {local_rank}] Starting testing')
    test_model(model, test_loader, batch_logging_output_inc, device, local_rank, use_mixed_precision)

    print(f'[{device.type.upper()} {local_rank}] Benchmarking complete')

def main(config_path):
    config = load_config(config_path)
    gpu_ids = config['gpu_ids']
    
    if gpu_ids[0] == -1: 
        main_worker(0, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        mp.spawn(main_worker, args=(config,), nprocs=len(gpu_ids), join=True)

if __name__ == "__main__":
    main("./config/llama.yaml")
