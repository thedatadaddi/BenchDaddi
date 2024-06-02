import subprocess

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully -> return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

def main():
    scripts = ['resnet50_train_test.py', 'bert_train_test.py', 'lstm_train_test.py']
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
