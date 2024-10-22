import subprocess
import requests
import json
import yaml
from statistics import mean
import os
import logging

# Load test prompts from YAML file
def load_test_prompts(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to set up a unique logger for each prompt
def setup_logger(gpu_name, model_name, prompt_index):
    model_name = model_name.split(':')[-1]
    
    # Create the logs directory if it does not exist
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the logger for this prompt
    logger = logging.getLogger(f"gpu_{gpu_name}.{model_name}.prompt{prompt_index}")
    logger.setLevel(logging.INFO)

    # Create a file handler for this prompt
    log_file = os.path.join(log_dir, f"gpu_{gpu_name}.{model_name}.prompt{prompt_index}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a logging format and add it to the file handler
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger

# Function to check GPU utilization using nvidia-smi command
def check_gpu_utilization():
    try:
        # Run nvidia-smi to get the status of all GPUs
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            return []

        # Parse the output of nvidia-smi
        gpu_utilization = []
        for line in result.stdout.strip().split('\n'):
            gpu_index, memory_used, memory_total = line.split(',')
            gpu_index = int(gpu_index.strip())
            memory_used = int(memory_used.strip())
            memory_total = int(memory_total.strip())

            # If memory is used, assume the model is loaded
            if memory_used > 0:
                gpu_utilization.append({
                    'gpu_id': gpu_index,
                    'memory_used': memory_used,
                    'memory_total': memory_total
                })

        return gpu_utilization
    except Exception as e:
        return []

# Function to call the LLM API and collect metrics
def call_llm_api(prompt, model="llama3.1"):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            api_response = response.json()

            # Extract relevant metrics from the response
            total_duration = api_response.get("total_duration", 0) / 1e9  # Convert to seconds
            load_duration = api_response.get("load_duration", 0) / 1e9  # Convert to seconds
            prompt_eval_duration = api_response.get("prompt_eval_duration", 0) / 1e9  # Convert to seconds
            eval_duration = api_response.get("eval_duration", 0) / 1e9  # Convert to seconds
            eval_count = api_response.get("eval_count", 0)

            # Calculate tokens per second
            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0

            return {
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_duration": eval_duration,
                "tokens_per_second": tokens_per_second,
                "response": api_response.get("response", "")
            }
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None

# Function to run the test for a single prompt multiple times and return averaged metrics
def run_tests_for_prompt(prompt, model="llama3.1", test_runs=3):
    metrics_list = []
    utilized_gpu_ids = "GPUs with models loaded:\n"
    
    for i in range(test_runs):
        
        # Check GPU utilization and log it
        if i == 1:
            loaded_gpus = check_gpu_utilization()
            if loaded_gpus:
                for gpu in loaded_gpus:
                    utilized_gpu_ids += f"GPU {gpu['gpu_id']}: {gpu['memory_used']}MB / {gpu['memory_total']}MB used\n"
            else:
                utilized_gpu_ids = "No GPUs with models loaded."
            
        metrics = call_llm_api(prompt, model)
        if metrics:
            metrics_list.append(metrics)
    
    if metrics_list:
        # Calculate averages for relevant metrics
        average_metrics = {
            "average_total_duration": mean([m["total_duration"] for m in metrics_list]),
            "average_load_duration": mean([m["load_duration"] for m in metrics_list]),
            "average_prompt_eval_duration": mean([m["prompt_eval_duration"] for m in metrics_list]),
            "average_eval_duration": mean([m["eval_duration"] for m in metrics_list]),
            "average_tokens_per_second": mean([m["tokens_per_second"] for m in metrics_list]),
            "response_example": metrics_list[0]["response"],  # Example response from the first run
            "utilized_gpu_ids": utilized_gpu_ids
        }
        return average_metrics
    return None

# Function to run all tests from the YAML file and log results
def test_all_prompts(gpu_id_list=['default'], test_file="test_prompts.yaml", model="llama3.1", test_runs=3):
    test_data = load_test_prompts(test_file)['tests']
    
    gpu_name = ''
    if len(gpu_id_list) == 1:
        gpu_name = str(gpu_id_list[0])
    else:
        gpu_name = "_".join(map(str, gpu_id_list))

    for index, test in enumerate(test_data, start=1):
        prompt = test["prompt"]
        logger = setup_logger(gpu_name, model, index)  # Setup a unique logger for each prompt

        # Add separators and line breaks between sections
        logger.info(f"Running tests for prompt {index}: '{prompt}'")
        logger.info("-" * 50)  # Vertical separator
        avg_metrics = run_tests_for_prompt(prompt, model, test_runs)
        
        if avg_metrics:
            logger.info(f"--- Average Metrics for Prompt {index} ---")
            logger.info(f"Total Duration: {avg_metrics['average_total_duration']:.2f} seconds")
            logger.info(f"Load Duration: {avg_metrics['average_load_duration']:.2f} seconds")
            logger.info(f"Prompt Eval Duration: {avg_metrics['average_prompt_eval_duration']:.2f} seconds")
            logger.info(f"Response Eval Duration: {avg_metrics['average_eval_duration']:.2f} seconds")
            logger.info(f"Tokens per Second: {avg_metrics['average_tokens_per_second']:.2f} tokens/s")
            logger.info(f"Utilized GPU IDs:\n{avg_metrics['utilized_gpu_ids']}")

            logger.info("-" * 50)  # Vertical separator
            logger.info("Example Response:\n")
            logger.info(f"{avg_metrics['response_example']}\n")
        else:
            logger.warning(f"No data collected for prompt {index}.\n")

if __name__ == "__main__":
    
    # GPU_ID - GPU(s) being tested
    gpu_id_list = ['0']
    
    # File containing test prompts (YAML format)
    test_file = "test_prompts.yaml"
    
    # Model name
    model_name = "llama3.1"
    
    # Number of test runs per prompt
    test_runs = 5
    
    # Run all tests
    test_all_prompts(gpu_id_list, test_file, model=model_name, test_runs=test_runs)
