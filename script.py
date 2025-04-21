import pathlib
import os # Used for joining paths robustly

# Define the root directory name for the project
project_name = "fintech_conversion_prediction_simple"

# Define the directory structure
# List of directories to create relative to the project root
directories = [
    "data/raw",
    "data/processed",
    "notebooks",
    "models",
    "reports/figures"
]

# Define placeholder files to create
# List of empty files to create relative to the project root
files = [
    "data/raw/fintech_application_data.csv", # Placeholder for the raw data
    "data/processed/.gitkeep", # Placeholder to keep the processed dir
    "notebooks/01_data_understanding.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_model_training_evaluation.ipynb",
    "models/.gitkeep", # Placeholder to keep the models dir
    "reports/figures/.gitkeep", # Placeholder to keep the figures dir
    "requirements.txt",
    "README.md"
]

# Get the path to the directory where the script is running
base_path = pathlib.Path.cwd()

# Create the main project directory
project_path = base_path / project_name
project_path.mkdir(exist_ok=True)
print(f"Created project root directory: {project_path}")

# Create the subdirectories
print("\nCreating subdirectories...")
for dir_name in directories:
    # Create a Path object for the subdirectory
    dir_path = project_path / pathlib.Path(dir_name)
    # Create the directory, including any necessary parent directories
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {dir_path}")

# Create the placeholder files
print("\nCreating placeholder files...")
for file_name in files:
    # Create a Path object for the file
    file_path = project_path / pathlib.Path(file_name)
    # Ensure the parent directory exists (it should, from the previous step)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Create an empty file (like 'touch' in Unix)
    file_path.touch()
    print(f"  Created: {file_path}")

print("\nProject structure created successfully!")

'''
How to Use:
1. Save this code as a Python file (e.g., create_project.py) in the location where you want the fintech_conversion_prediction_simple folder to be created.
2. Run the script from your terminal: python create_project.py

What it Does:
* Creates the main project folder (fintech_conversion_prediction_simple)
* Creates all the subfolders (data/raw, data/processed, notebooks, models, reports/figures)
* Creates empty placeholder files (.ipynb, .txt, .md, .csv, .gitkeep) within the appropriate folders. 
  The .gitkeep files are common practice to ensure empty directories are tracked by Git if you use version control.
  The .csv file is just an empty placeholder; you'll need to add your actual data file later.
'''