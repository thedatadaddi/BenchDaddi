"""
Description: This code performs intensive GPU loading by repeatedly executing matrix multiplication
on large random matrices using PyTorch. Each GPU specified in the list is loaded with heavy 
computation in a separate process, utilizing the torch.multiprocessing module. The script aims to stress
test the GPUs by continuously performing operations that demand significant computational power.

Author: TheDataDaddi
Date: 2024-02-02
Version: 1.0
License: MIT License
"""

import time
import torch
import argparse
import threading
import multiprocessing
import queue
import torch.multiprocessing as mp

from typing import Any


# global state for threads (do not modify)
run_ctx = {
    't_start': 0.0,  # time when test started
    'dt_stress': 0.0,  # duration of the test
    'flag_start': False,  # set to true when stress test is started
    'flag_stop': False,  # set to true when stress test must stop
}


def load_gpu(
        gpu_id: int,
        q_in: multiprocessing.Queue,
        q_out: multiprocessing.Queue,
        n: int = 10000,
        t_max: float = 3600 * 8
):
    """Load a specific GPU with heavy computation."""
    assert gpu_id >= 0, 'GPU ID must be a positive integer.'
    t0 = time.time()
    # Set the device to the specified GPU
    device = torch.device(f'cuda:{gpu_id}')
    
    # Create large random matrices
    matrix1 = torch.randn(n, n, device=device)
    matrix2 = torch.randn(n, n, device=device)

    count_matmul = 0
    dt_count = 5.0
    t_cur = time.time()
    t_ops0 = t_cur
    try:
        # Perform matrix multiplication repeatedly to load the GPU
        while t_cur - t0 < t_max:
            try:
                data = q_in.get(False)
            except queue.Empty:
                pass
            else:
                if data == 'STOP':
                    break
            if t_cur - t_ops0 >= dt_count:
                matmul_per_second = count_matmul / (time.time() - t_ops0)
                count_matmul = 0
                t_ops0 = time.time()
                q_out.put(f'load_gpu(gpu_id:{gpu_id}), M1({n}:{n}) x M2({n}:{n}), matmul_per_second={matmul_per_second:.2f}')

            _ = torch.matmul(matrix1, matrix2)
            count_matmul += 1
            t_cur = time.time()
    except KeyboardInterrupt:
        print('KeyboardInterrupt Exception received inside load_gpu process. Exiting GPU load.')


def get_num_gpus() -> int:
    """How many cuda devices are available?"""
    return torch.cuda.device_count()


def list_gpu_properties(output: str = 'str') -> str | list[(int, Any)]:
    """List properties of all available cuda GPUs."""
    l = [(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]
    return '\n'.join(['#device#\t#properties#'] + [f'cuda:{i}\t\t{p}' for i, p in l]) if output == 'str' else l


def list_gpu_running_state(gpu_id_list: list[int]):
    """List the running state of each GPU in the list. This includes: Temperature, Memory Usage, and Power Usage."""
    if not run_ctx['flag_start']:
        print('GPU state before stress test started:')
    else:
        t_remaining = run_ctx['t_start'] + run_ctx['dt_stress'] - time.time()
        print(f'GPU state after stress test started -- stress time remaining {t_remaining:.2f} secs: {time.strftime("%H:%M:%S", time.gmtime(t_remaining))}')
    for gpu_id in gpu_id_list:
        print(f'GPU cuda:{gpu_id} - {torch.cuda.get_device_name(gpu_id)}')
        device = torch.device(f'cuda:{gpu_id}')

        gpu_temperature = torch.cuda.temperature(device)
        gpu_power_draw = torch.cuda.power_draw(device)  # average power draw in mW (MilliWatts)

        print(f"GPU Temperature: {gpu_temperature}°C")
        print(f"GPU Power Draw: {gpu_power_draw / 1000} W")

        free, total = torch.cuda.mem_get_info(device)

        print('Memory Usage:')
        print('Allocated:', round((total - free) / 1024 ** 3, 1), 'GB')
        print('Total:   ', round(total / 1024 ** 3, 1), 'GB')
        print('\n')


def report_loop(gpu_id_list: list[int], q_out: multiprocessing.Queue, t_sleep: float = 10):
    """Report GPU load every t_sleep seconds."""
    while not run_ctx['flag_stop'] and run_ctx['t_start'] + run_ctx['dt_stress'] > time.time():
        list_gpu_running_state(gpu_id_list)
        print_messages_from_queue(q_out)
        time.sleep(t_sleep)


def print_messages_from_queue(q_out: multiprocessing.Queue):
    out = ['Now reporting messages from load_gpu Processes:', ]
    while True:
        try:
            msg = q_out.get(False)
            out.append(str(msg))
        except queue.Empty:
            break
    if len(out) > 1:
        print('\n'.join(out) + '\n')


def send_stop_signal(gpu_id_list: list[int], q: multiprocessing.Queue):
    """Send a stop signal to the queue."""
    run_ctx['flag_stop'] = True
    for i in range(len(gpu_id_list)):
        q.put('STOP')


def cli():
    """Command line interface for the script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', type=int, default=120, help='Time in seconds to run the GPU load. (Default: 120 secs).')
    parser.add_argument('-n', type=int, default=10000, help='Num of rows in the square matrix with which to perform matrix multiplication. Matrix will have dimensions (n x n). (Default: 10000).')
    return parser.parse_args()


def main():
    """Main function to run the script."""
    args = cli()

    print(f'List of GPUs:\n{list_gpu_properties()}\n')

    # List of GPU IDs to load
    gpu_id_list = list(range(0, get_num_gpus()))  # [0, 1]

    list_gpu_running_state(gpu_id_list)

    print(f'>> Initiating run of max load GPU for {args.t} seconds...\n\n')

    mp.set_start_method('spawn')

    t_start = run_ctx['t_start'] = time.time()
    dt_stress = run_ctx['dt_stress'] = args.t
    flag_stop = run_ctx['flag_stop'] = False
    run_ctx['flag_start'] = True

    q_in = mp.Queue()
    q_out = mp.Queue()
    processes = []

    # Load each GPU in a separate process
    for id_ in gpu_id_list:
        # Create a new process for each GPU
        p = mp.Process(target=load_gpu, args=(id_, q_in, q_out, args.n, args.t))
        # Start the process
        p.start()
        # Add the process to the list of processes
        processes.append(p)

    t_report = threading.Thread(target=report_loop, args=(gpu_id_list, q_out, 5))
    t_report.start()

    try:
        while not flag_stop and t_start + dt_stress > time.time():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nStopping stress test by user request. Please wait for the processes to finish. This may take a while.')
    send_stop_signal(gpu_id_list, q_in)

    # join report threads and wait for it to finish
    t_report.join()
    # Join all processes to wait until they finish
    for p in processes:
        p.join()

    print('<< Done stressing GPUs!')


if __name__ == '__main__':
    main()
