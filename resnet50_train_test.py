import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import time
import torch.cuda as cuda
import os
import logging
from datetime import datetime
import numpy as np
import math
import yaml 

def load_config(config_path):
    """
    Load configuration from a YAML file.
    Args:
    config_path (str): Path to the configuration YAML file.
    Returns:
    dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_directory, current_time):
    """
    Setup logging configuration.
    Args:
    log_directory (str): Directory to store log files.
    current_time (str): Current timestamp for unique file naming.
    """
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_filename = f"{log_directory}/renet50_tt_{current_time}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def setup_and_log_devices(gpu_ids):
    """
    Setup CUDA devices based on the provided GPU IDs and log details about the selected devices.
    
    Args:
    gpu_ids (list of int): List of GPU IDs to be used. If the list contains -1, use CPU.
    
    Returns:
    list of torch.device or torch.device: List of devices or a single device depending on the setup.
    """
    if -1 in gpu_ids:
        logging.info("Using CPU.")
        print("Using CPU.")
        return torch.device("cpu")
    
    if not torch.cuda.is_available():
        logging.info("CUDA is not available. Using CPU.")
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")
    
    logging.info(f"CUDA is available: Version {torch.version.cuda}")
    print(f"CUDA is available: Version {torch.version.cuda}")
    
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    
    devices = [torch.device(f'cuda:{i}') for i in gpu_ids]
    for device in devices:
        prop = torch.cuda.get_device_properties(device)
        device_info = f"Selected Device ID {device.index}: {prop.name}, Compute Capability: {prop.major}.{prop.minor}, Total Memory: {prop.total_memory / 1e9:.2f} GB"
        logging.info(device_info)
        print(device_info)
    
    return devices

def log_memory_usage(devices):
    """
    Logs the current GPU memory usage for each device provided in the list.
    
    Args:
    devices (list of torch.device or torch.device): List of devices or a single device to log memory usage for.
    """
    if isinstance(devices, torch.device):  # If a single device is provided, wrap it in a list
        devices = [devices]

    if devices[0].type == 'cuda':
        for device in devices:
            allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert bytes to gigabytes
            reserved = torch.cuda.memory_reserved(device) / 1e9    # Convert bytes to gigabytes
            logging.info(f"{device}: Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")
    else:
        logging.info("CPU mode, no GPU device memory to log.")

def load_data(dataset_name, data_directory, img_resize, norm_means, norm_stds, batch_size, num_workers):
    """
    Load a dataset with specified configuration.

    Args:
    config (dict): Configuration dictionary containing dataset parameters.

    Returns:
    DataLoader: DataLoader for the training dataset.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(norm_means, norm_stds)
    ])

    print(f"Loading {dataset_name} dataset")
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root=data_directory, train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root=data_directory, train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root=data_directory, train=True, download=True, transform=transform)
        
    num_classes = len(dataset.classes)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader, num_classes

def initialize_model(num_classes, gpu_ids, devices):
    """
    Initialize the ResNet-50 model, modify it for CIFAR-10 classes, and set it to use multiple GPUs if available.
    
    Args:
    num_classes (int): Number of classes in the dataset.
    gpu_ids (list of int): Optional list of GPU IDs for model parallelization.
    
    Returns:
    torch.nn.Module: ResNet-50 model configured for CIFAR-10 and parallelized across specified GPUs.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info('Model loaded on GPU')
        log_memory_usage(devices)
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    return model

def train_model(model, data_loader, epochs, learning_rate, batch_logging_output_inc, devices):
    """
    Train the model using the provided data loader and specified number of epochs,
    and measure the time taken for each epoch and each batch. Also compute average times
    for epochs and batches at the end.

    Args:
    model (torch.nn.Module): The neural network model to train.
    data_loader (DataLoader): DataLoader for the training data.
    epochs (int): Number of epochs to train the model.

    Returns:
    None: logs training times per epoch, per batch, and averages.
    """
    
    print(f'Training model for {epochs} epochs')
    logging.info(f'###############################################################################')
    logging.info(f'TRAINING')
    logging.info(f'###############################################################################')
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    avg_batch_exec_time = 0
    avg_batch_data_transfer_time = 0
    total_batches = 0
    total_samples = 0
    start_time = time.time()  # Start time for total training

    for i, epoch in enumerate(range(epochs)):
        batch_time_list = []
        batch_data_transfer_time_list = []
        epoch_start_time = time.time()  # Start time for the epoch

        for j, data in enumerate(data_loader, 0):
            
            num_batches = len(data_loader)
            batch_start_time = time.time()  # Start time for the batch

            data_transfer_start_time = time.time()
            inputs, labels = data[0].cuda(), data[1].cuda()
            total_samples += inputs.size(0)
            data_transfer_time = time.time() - data_transfer_start_time
            batch_data_transfer_time_list.append(data_transfer_time)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start_time  # Calculate time taken for this batch
            batch_time_list.append(batch_time)
            
            total_batches += 1

            batch_loss = loss.item()
            
            if (j + 1) % batch_logging_output_inc == 0:
                logging.info(f'Epoch {epoch + 1}, Batch {j + 1}/{num_batches}, Loss: {batch_loss:.3f}, Data Transfer Time: {data_transfer_time:.4f} seconds, Batch Exec Time: {batch_time:.3f} seconds')
                log_memory_usage(devices)
                     
        avg_ep_batch_exec_time = np.mean(batch_time_list)
        avg_ep_batch_data_transfer_time = np.mean(batch_data_transfer_time_list)
        epoch_time = time.time() - epoch_start_time  # Calculate time taken for this epoch        
        
        logging.info(f'Epoch {epoch + 1} -> Average Batch Exec Time {avg_ep_batch_exec_time:.3f} seconds')
        logging.info(f'Epoch {epoch + 1} -> Average Data Transfer Time {avg_ep_batch_data_transfer_time:.3f} seconds')
        logging.info(f'Epoch {epoch + 1} -> % Average Data Transfer Time of Batch Execution Time {avg_ep_batch_data_transfer_time/avg_ep_batch_exec_time*100 :.3f} %')
        logging.info(f'Epoch {epoch + 1} -> completed in {epoch_time:.3f} seconds')
        
        avg_batch_exec_time += avg_ep_batch_exec_time
        avg_batch_data_transfer_time += avg_ep_batch_data_transfer_time

    total_training_time = time.time() - start_time

    logging.info(f'###############################################################################')
    logging.info(f'TRAINING METRICS')
    logging.info(f'Total Time: {total_training_time:.3f} seconds')
    logging.info(f'Average Batch Execution Time: {avg_batch_exec_time/epochs:.3f} seconds')
    logging.info(f'Average Batch Data Transfer Time: {avg_batch_data_transfer_time/epochs:.3f} seconds')
    logging.info(f'% Average Batch Data Transfer Time of Batch Average Execution Time: {avg_batch_data_transfer_time/avg_batch_exec_time*100:.3f} %')
    logging.info(f'Throughput: {total_samples/total_training_time:.3f} images/second')

def evaluate_model(model, data_loader, batch_logging_output_inc):
    """
    Evaluate the model using the provided data loader, measure inference time and throughput.
    
    Args:
    model (torch.nn.Module): The neural network model to evaluate.
    data_loader (DataLoader): DataLoader for the evaluation data.
    
    Returns:
    None: Logs inference related metrics.
    
    """
    
    print(f'Evaluating model')
    logging.info(f'###############################################################################')
    logging.info(f'INFERENCE')
    logging.info(f'###############################################################################')
    
    total_inference_time = 0
    total_samples = 0
    num_batches = len(data_loader)
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            
            inputs, labels = data[0].cuda(), data[1].cuda()
            start_time = time.time()
            outputs = model(inputs)
            cuda.synchronize()  # Ensure CUDA has finished processing all batches
            
            batch_time = time.time() - start_time
            total_inference_time += batch_time
            total_samples += inputs.size(0)
            
            if (i + 1) % batch_logging_output_inc == 0:
                logging.info(f'Batch {i + 1}/{num_batches}, Batch Exec Time: {batch_time:.3f} seconds')

    logging.info(f'###############################################################################')
    logging.info(f'INFERENCE METRICS')
    logging.info(f'Total Time: {total_inference_time:.3f} seconds')
    logging.info(f'Average Batch Execution Time: {total_inference_time/num_batches:.3f} seconds')
    logging.info(f'Throughput: {total_samples/total_inference_time:.3f} images/second')

def main(config_path):
    config = load_config(config_path)
    setup_logging(config['log_directory'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print('Setting up devices and loading data')
    devices = setup_and_log_devices(config['gpu_ids'])
    
    data_loader, num_classes = load_data(
        config['dataset_name'], config['data_directory'], config['image_resize'], config['normalization_means'], 
        config['normalization_stds'], config['batch_size'], config['num_workers']
    )
    
    print('Initializing model')
    model = initialize_model(num_classes, config['gpu_ids'], devices)
    
    batch_logging_output_inc = config['batch_logging_output_inc']
    
    print('Starting training')
    train_model(model, data_loader, config['epochs'], config['learning_rate'], batch_logging_output_inc, devices)
    
    print('Starting evaluation')
    evaluate_model(model, data_loader, batch_logging_output_inc)
    
    print('Benchmarking complete')

if __name__ == "__main__":
    main("./config/resnet50.yaml")
