import os

# Set HF_HUB_ENABLE_HF_TRANSFER environment variable to speed up downloads
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Increase timeout to avoid ReadTimeout errors
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
from huggingface_hub import snapshot_download

def download_aida_dataset(target_dir=None):
    """
    Downloads the MichaelYang-lyx/AIDA dataset from Hugging Face to the specified local directory.
    
    Args:
        target_dir (str, optional): The directory to download the dataset to. 
                                    If None, defaults to 'data' in the project root.
    """
    if target_dir is None:
        # Assuming this script is located at AIDABench/download_data.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_dir, "data")
        
    print(f"Downloading dataset 'MichaelYang-lyx/AIDA' to {target_dir}...")
    
    os.makedirs(target_dir, exist_ok=True)
    
    snapshot_download(
        repo_id="MichaelYang-lyx/AIDA",
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,  # Download actual files
        resume_download=True
    )
    
    print("Download complete.")

if __name__ == "__main__":
    download_aida_dataset()
