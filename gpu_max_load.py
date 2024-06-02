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
    
    gpu_id_list = [0, 1]

    processes = []

    # Load each GPU in a separate process
    for id_ in gpu_id_list:
        p = Process(target=load_gpu, args=(id_,))
        p.start()
        processes.append(p)

    # Join all processes
    for p in processes:
        p.join()

