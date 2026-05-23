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
# Target directory for raw data (aligned with the MLOps project structure).
DATA_DIR = "data/raw"

# Define the specific years and months to download.
# Key: Year (int), Value: List of months (list of ints).
TARGETS_TO_DOWNLOAD = {
    2024: [1, 2] #for small test
    # 2024: list(range(1, 13)),  # Full year 2024 (Jan - Dec)
    # 2025: list(range(1, 13)),  # Full year 2025 (Jan - Dec)
    # 2026: [1, 2]               # Test set: Jan and Feb 2026 only
}
# ==========================================


def setup_data_directory(base_dir: str = DATA_DIR) -> str:
    """
    Ensures the target data directory exists locally.
    
    Args:
        base_dir (str): Path to the target directory.
        
    Returns:
        str: The validated directory path.
    """
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def generate_download_targets(targets_dict: dict) -> list:
    """
    Generates a flattened list of (year, month) tuples based on the configuration block.
    
    Args:
        targets_dict (dict): Dictionary mapping years to lists of months.
        
    Returns:
        list of tuple: A list containing (year, month) combinations.
    """
    targets = []
    for year, months in targets_dict.items():
        for month in months:
            targets.append((year, month))
    return targets


def download_file(url: str, output_path: str, file_name: str) -> bool:
    """
    Downloads a single file via an HTTP GET request with a visual progress bar.
    Skips the download if the file already exists locally to prevent data replication.
    
    Args:
        url (str): The remote URL of the target file.
        output_path (str): The local destination path.
        file_name (str): The name of the file being processed.
        
    Returns:
        bool: True if the file exists or downloaded successfully, False otherwise.
    """
    # Check if file exists to ensure idempotency (prevent duplicate downloads)
    if os.path.exists(output_path):
        print(f"  [SKIPPED] {file_name} already exists locally.")
        return True

    print(f"  [FETCHING] {file_name}...")
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


def main():
    """
    Main execution pipeline for sequential data retrieval.
    """
    print("=" * 60)
    print("INITIALIZING NYC TLC DATA SYNC PIPELINE")
    print("=" * 60)
    
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    data_dir = setup_data_directory()
    targets = generate_download_targets(TARGETS_TO_DOWNLOAD)
    
    success_count = 0
    
    for year, month in targets:
        # Construct standardized file name format: yellow_tripdata_YYYY-MM.parquet
        file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
        url = f"{base_url}/{file_name}"
        output_path = os.path.join(data_dir, file_name)
        
        if download_file(url, output_path, file_name):
            success_count += 1

    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Execution Summary: Successfully verified {success_count}/{len(targets)} partitions.")
    print(f"Target Directory: '{data_dir}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()