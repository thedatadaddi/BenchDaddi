import subprocess
import os
from datetime import datetime
import shutil

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully -> return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        
def sum_global_throughputs(log_file_path):
    total_throughput = 0.0
    
    # Regular expression to find lines with global throughput and extract the numeric value
    throughput_pattern = re.compile(r'Global (?:Training|Testing) Throughput: (\d+\.\d+) samples/second')

    # Open and read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            match = throughput_pattern.search(line)
            if match:
                # Convert the extracted throughput value to float and add to the total
                throughput_value = float(match.group(1))
                total_throughput += throughput_value
                
    return total_throughput

def main():
    
    log_dir = './logs'
    shutil.rmtree(log_dir)
    
    scripts = ['resnet50_train_test.py', 'bert_train_test.py', 'lstm_train_test.py']
    for script in scripts:
        run_script(script)
    
    results_path = f"{log_dir}/results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    bm_score = 0 # Sum across all models
    for log in os.listdir(log_dir):
        log_path = os.path.join(log_dir, log)
        print(log_path)
        sgt = sum_global_throughputs (log_path)
        bm_score = bm_score + sgt
        with open (results_path, 'a+') as res:
            s1 = f"{log.split('_')[0]} score: {sgt}\n"
            res.write(s1)
    with open (results_path, 'a+') as res:
        s2 = f"benchmark score: {bm_score}"
        res.write(s2)

if __name__ == "__main__":
    main()
