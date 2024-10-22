import os
import csv
import re

# Define the directory containing log files
log_dir = './logs'
output_csv = 'logs_summary.csv'

# Regex pattern to match the average metrics in the file content
metrics_pattern = r"Total Duration: ([\d.]+) seconds\s+Load Duration: ([\d.]+) seconds\s+Prompt Eval Duration: ([\d.]+) seconds\s+Response Eval Duration: ([\d.]+) seconds\s+Tokens per Second: ([\d.]+) tokens/s"

# Regex pattern to match the "No data collected" message
no_data_pattern = r"No data collected for prompt (\d+)"

# Define column headers for the output CSV file
headers = ['gpu_ids', 'model_name', 'prompt_num', 'total_duration', 'load_duration', 'prompt_eval_duration', 'response_eval_duration', 'tokens_per_second']

# Function to parse a single log file and return data for the CSV
def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        # Check if the "No data collected" message is present
        no_data_match = re.search(no_data_pattern, content)
        if no_data_match:
            # Extract the prompt number and return empty values for the metrics
            prompt_num = f'prompt{str(no_data_match.group(1))}'
            # Extract file name components (gpu_ids, model_name)
            file_name = os.path.basename(file_path)
            file_name_parts = file_name.split('.')
            gpu_ids = file_name_parts[0]
            model_name = file_name_parts[1]
            # Return the row with empty metric values
            return [gpu_ids, model_name, prompt_num, None, None, None, None, None]

        # If no "No data collected" message, check for the metrics
        match = re.search(metrics_pattern, content)
        if match:
            # Extract relevant metrics
            total_duration = match.group(1)
            load_duration = match.group(2)
            prompt_eval_duration = match.group(3)
            response_eval_duration = match.group(4)
            tokens_per_second = match.group(5)
        else:
            # If metrics not found, return None
            return None
        
        # Extract file name components (gpu_ids, model_name, prompt_num)
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('.')
        gpu_ids = file_name_parts[0]
        model_name = file_name_parts[1]
        prompt_num = file_name_parts[2].replace('.log', '')

        # Return parsed data as a list
        return [gpu_ids, model_name, prompt_num, total_duration, load_duration, prompt_eval_duration, response_eval_duration, tokens_per_second]

# Main function to process all log files in the directory
def process_logs(log_dir, output_csv):
    # Open the output CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(headers)

        # Iterate over all log files in the directory
        for file_name in os.listdir(log_dir):
            if file_name.endswith('.log'):
                # Parse the log file
                file_path = os.path.join(log_dir, file_name)
                log_data = parse_log_file(file_path)

                # If log data was successfully parsed, write it to the CSV
                if log_data:
                    writer.writerow(log_data)

# The entry point for the script
if __name__ == "__main__":
    process_logs(log_dir, output_csv)
    print(f"Log files processed and saved to {output_csv}")
