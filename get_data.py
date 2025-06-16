import os
import requests
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, dest_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Base URL for the dataset
    base_url = "https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"
    
    # Create directories
    for dir_name in ['train', 'test']:
        os.makedirs(f'MNE-schirrmeister2017-data/{dir_name}', exist_ok=True)
    
    # Download training and test files
    for split in ['train', 'test']:
        logger.info(f"Downloading {split} files...")
        for subj_id in range(1, 15):
            # Download EDF file
            edf_url = f"{base_url}/{split}/{subj_id}.edf"
            edf_path = f"MNE-schirrmeister2017-data/{split}/{subj_id}.edf"
            logger.info(f"Downloading {edf_url}")
            download_file(edf_url, edf_path)

if __name__ == "__main__":
    main()