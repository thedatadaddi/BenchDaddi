"""
Description: This code performs intensive GPU loading by repeatedly executing matrix multiplication
on large random matrices using PyTorch. Each GPU specified in the list is loaded with heavy 
computation in a separate process, utilizing the multiprocessing module. The script aims to stress 
test the GPUs by continuously performing operations that demand significant computational power.

Author: TheDataDaddi
Date: 2024-02-02
Version: 1.0
License: MIT License
"""


import torch
from multiprocessing import Process

# Function to load a specific GPU with heavy computation
def load_gpu(gpu_id):
    # Set the device to the specified GPU
    device = torch.device(f'cuda:{gpu_id}')
    
    # Create large random matrices
    matrix1 = torch.randn(10000, 10000, device=device)
    matrix2 = torch.randn(10000, 10000, device=device)

    # Perform matrix multiplication repeatedly to load the GPU
    while True:
        _ = torch.matmul(matrix1, matrix2)

if __name__ == '__main__':
    # List of GPU IDs to load
    gpu_id_list = [0, 1]

    processes = []

    # Load each GPU in a separate process
    for id_ in gpu_id_list:
        # Create a new process for each GPU
        p = Process(target=load_gpu, args=(id_,))
        # Start the process
        p.start()
        # Add the process to the list of processes
        processes.append(p)

    # Join all processes
    for p in processes:
        p.join()
