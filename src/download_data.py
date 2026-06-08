"""
NYC TLC Yellow Taxi Data Downloader
-----------------------------------
This script automates the retrieval of the New York City Taxi and Limousine 
Commission (TLC) Yellow Taxi trip record datasets in Parquet format.
It is designed to be modular, idempotent (skips existing files), and 
aligned with standard ML project directory structures.
"""

import os
import requests
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Base directory for raw data (aligned with the MLOps project structure).
DATA_DIR = "data/raw"

# Define the specific years and months to download.
# Keys represent sub-directories (e.g., 'train', 'test').
# Values are mappings of Year (int) to List of months (list of ints).
TARGETS_TO_DOWNLOAD = {
    "train": {
        # 2024: [1, 3], # for small test
        2024: list(range(1, 13)),  # Full year 2024 (Jan - Dec)
        # 2025: list(range(1, 13)),  # Full year 2025 (Jan - Dec)
    },
    "test": {
        # 2025: [1,2],  #small test 
        2026: [1, 2]               # Test set: Jan and Feb 2026 only
    }
}

# Set to True to only print URLs and check file existence without downloading.
DRY_RUN = False
# ==========================================


def setup_data_directories(base_dir: str, splits: list) -> dict:
    """
    Ensures the target data sub-directories exist locally.
    
    Args:
        base_dir (str): Path to the target directory.
        splits (list): List of sub-directory names (e.g., ['train', 'test']).
        
    Returns:
        dict: A mapping of split names to their created directory paths.
    """
    dirs = {}
    for split in splits:
        path = os.path.join(base_dir, split)
        os.makedirs(path, exist_ok=True)
        dirs[split] = path
    return dirs


def generate_download_targets(targets_config: dict) -> list:
    """
    Generates a flattened list of (split, year, month) tuples based on the configuration block.
    
    Args:
        targets_config (dict): Dictionary mapping splits to years and months.
        
    Returns:
        list of tuple: A list containing (split, year, month) combinations.
    """
    targets = []
    for split, years_dict in targets_config.items():
        for year, months in years_dict.items():
            for month in months:
                targets.append((split, year, month))
    return targets


def download_file(url: str, output_path: str, file_name: str, dry_run: bool = False) -> bool:
    """
    Downloads a single file via an HTTP GET request with a visual progress bar.
    Skips the download if the file already exists locally to prevent data replication.
    
    Args:
        url (str): The remote URL of the target file.
        output_path (str): The local destination path.
        file_name (str): The name of the file being processed.
        dry_run (bool): If True, only checks existence and prints the URL without downloading.
        
    Returns:
        bool: True if the file exists or downloaded successfully, False otherwise.
    """
    # Check if file exists to ensure idempotency (prevent duplicate downloads)
    if os.path.exists(output_path):
        print(f"  [EXISTS] {file_name} is already locally available.\n           URL: {url}")
        return True

    if dry_run:
        print(f"  [MISSING] {file_name} needs to be downloaded.\n            URL: {url}")
        return False

    print(f"  [FETCHING] {file_name}\n             URL: {url}")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=f"    Progress"
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            return True
        else:
            print(f"  [ERROR] Failed to download {file_name} (HTTP Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"  [EXCEPTION] An error occurred while transferring {file_name}: {e}")
        return False


def run_download_pipeline(targets_config: dict = TARGETS_TO_DOWNLOAD, base_dir: str = DATA_DIR, dry_run: bool = DRY_RUN):
    """
    Main execution pipeline for sequential data retrieval.
    """
    print("=" * 60)
    print("INITIALIZING NYC TLC DATA SYNC PIPELINE")
    print("=" * 60)
    
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    splits = list(targets_config.keys())
    dirs = setup_data_directories(base_dir, splits)
    targets = generate_download_targets(targets_config)
    
    success_count = 0
    
    for split, year, month in targets:
        # Construct standardized file name format: yellow_tripdata_YYYY-MM.parquet
        file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
        url = f"{base_url}/{file_name}"
        output_path = os.path.join(dirs[split], file_name)
        
        if download_file(url, output_path, file_name, dry_run=dry_run):
            success_count += 1

    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Execution Summary: Successfully verified {success_count}/{len(targets)} partitions.")
    print("Target Directories:")
    for split_name, split_path in dirs.items():
        print(f"  - {split_name.upper()}: '{split_path}/'")
    print("=" * 60)


if __name__ == "__main__":
    run_download_pipeline()