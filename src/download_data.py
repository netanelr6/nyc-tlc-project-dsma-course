"""
NYC TLC Yellow Taxi Data Downloader
-----------------------------------
This script automates the retrieval of the New York City Taxi and Limousine 
Commission (TLC) Yellow Taxi trip record datasets in Parquet format.
It targets all months for 2024 and 2025, and the first two months of 2026.

The data is saved locally to a designated directory which should be excluded
from version control (Git) via .gitignore.
"""

import os
import requests
from tqdm import tqdm


def setup_data_directory(base_dir: str = "data") -> str:
    """
    Ensures the target data directory exists locally.
    
    Args:
        base_dir (str): Path to the target directory.
        
    Returns:
        str: The validated directory path.
    """
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def generate_download_targets() -> list:
    """
    Generates a list of tuples containing the specific years and months 
    required for the assignment pipeline (2024-2025 for training, 
    Jan-Feb 2026 for testing).
    
    Returns:
        list of tuple: A list containing (year, month) combinations.
    """
    targets = []
    
    # Full dataset for training: 2024 and 2025
    for year in [2024, 2025]:
        for month in range(1, 13):
            targets.append((year, month))
            
    # Restricted test set: January and February 2026
    for month in range(1, 3):
        targets.append((2026, month))
        
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
    targets = generate_download_targets()
    
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