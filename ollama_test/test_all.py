import yaml
import subprocess
import os
import time
import requests

# Function to check if the model is accessible
def check_model_loaded(port=11434):
    try:
        response = requests.get(f"http://localhost:{port}/api/ps")
        if response.status_code == 200:
            data = response.json()
            if "models" in data and len(data["models"]) > 0:
                return True
    except requests.exceptions.RequestException:
        return False
    return False

# Function to pull the required model using Ollama API
def pull_model(model_name, port=11434):
    url = f"http://localhost:{port}/api/pull"
    payload = {
        "name": model_name,
    }
    
    try:
        response = requests.post(url, json=payload, stream=False)
        if response.status_code == 200:
            print(f"Model {model_name} pulled successfully!")
        else:
            print(f"Failed to pull model {model_name}. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while pulling model {model_name}: {e}")
        
def remove_ollama_vol():
    try:
        subprocess.run(['docker', 'volume', 'rm', 'ollama'], check=True)
        print("Volume 'ollama' removed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error removing volume: {e.stderr.decode('utf-8')}")

# Function to stop and remove Docker container
def stop_and_remove_container(remove_ollama_vol_flag, cont_name):
    subprocess.run(["docker", "stop", cont_name])  # Use cont_name
    subprocess.run(["docker", "rm", cont_name])
    print(f"Container {cont_name} removed successfully!")
    if remove_ollama_vol_flag:
        remove_ollama_vol()
    
# Function to run Docker container for each model and GPU combination
def run_docker_container(ollama_vol_flag, model_name, cont_name, gpu_ids, port=11434):
    # Prepare the GPU devices flag based on the gpu_ids input
    if gpu_ids[0] == 'cpu':
        gpu_flag = []
    else:
        gpu_devices = ','.join([str(gpu) for gpu in gpu_ids])
        gpu_flag = ["--gpus", f"\"device={gpu_devices}\""]

    # Prepare the base Docker run command
    docker_run_command = ["docker", "run", "-d"] + gpu_flag

    # Add the volume mount if ollama_vol_flag is True
    if ollama_vol_flag:
        docker_run_command += ["-v", "ollama:/root/.ollama"]

    # Add port mapping and container name
    docker_run_command += ["-p", f"{port}:{port}", "--name", cont_name, "ollama/ollama"]

    # Run the docker command
    subprocess.run(docker_run_command)

    # Wait for the container to be fully running
    container_status_command = f"docker inspect -f '{{{{.State.Running}}}}' {cont_name}"
    container_running = False
    while not container_running:
        status = subprocess.check_output(container_status_command, shell=True).strip().decode('utf-8')
        if status == "true":
            container_running = True
        else:
            time.sleep(1)
    
    # Pull the model using the Ollama API
    pull_model(model_name, port)

    docker_exec_command = [
        "docker", "exec", "-dit", cont_name,
        "ollama", "run", model_name  # Adjust based on correct API usage
    ]
    subprocess.run(docker_exec_command)
    
    print(f"Waiting for model {model_name} to be accessible...")
    while not check_model_loaded(port=port):
        time.sleep(5)
    print(f"Model {model_name} is now accessible!")

# Import the test function from test_model_prompts
from test_model_prompts import test_all_prompts

# Main execution block
if __name__ == "__main__":
    
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test_file = config['test_file']
    test_runs = config['test_runs']
    model_list = config['model_list']
    gpu_id_lists = config['gpu_id_lists']
    ollama_vol_flag = config['ollama_vol_flag']
    remove_ollama_vol_flag = config['remove_ollama_vol_flag']
    
    # Loop over GPU and model combinations
    for gpu_id_list in gpu_id_lists:
        for model_name in model_list:
            cont_name = f"ollama_{model_name.split(':')[-1]}"
            
            try:
                # Run Docker container
                run_docker_container(ollama_vol_flag, model_name, cont_name, gpu_id_list)

                # Run tests after the model is loaded
                print("Testing prompts...")
                test_all_prompts(gpu_id_list, test_file, model=model_name, test_runs=test_runs)
                print("Testing finished successfully!")
                
            finally:
                # Clean up container
                stop_and_remove_container(remove_ollama_vol_flag, cont_name)
                print(f"Cleaned up container for model {model_name}")
