"""
Description: This script sets up a Conda environment with a specified Python version and installs the required packages listed in 'requirements.txt' using pip. It is intended for use in environments where dependencies need to be managed via Conda and pip.

Author: TheDataDaddi
Date: 2024-02-02
Version: 1.0
"""

import subprocess

# Define the name of the conda environment and Python version
env_name = 'gpu_bm_venv'
python_version = '3.10'

# Function to create the conda environment and install packages
def create_conda_env():
    try:
        # Create the conda environment
        subprocess.run(['conda', 'create', '--name', env_name, f'python={python_version}', '-y'], check=True)
        print(f"Conda environment '{env_name}' created successfully with Python {python_version}.")
        print("Installing packages now... Please wait!")

        # Activate the environment and install the packages using pip
        subprocess.run(f'conda run -n {env_name} pip install -r requirements.txt', shell=True, check=True)
        print(f"Required packages installed successfully in the Conda environment '{env_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while creating the conda environment or installing packages: {e}")

if __name__ == '__main__':
    create_conda_env()
