from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

def upload_dataset(dataset_dir: str, repo_name: str, repo_type: str = "dataset"):
    """Upload the dataset to Hugging Face Hub."""
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        create_repo(repo_name, repo_type=repo_type)
        print(f"Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository already exists or error: {e}")
    
    # Upload the dataset
    print(f"Uploading dataset to {repo_name}...")
    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_name,
        repo_type=repo_type
    )
    print("Upload complete!")

if __name__ == "__main__":
    # Get Hugging Face token from environment variable
    if "HUGGINGFACE_TOKEN" not in os.environ:
        print("Please set your Hugging Face token as an environment variable:")
        print("export HUGGINGFACE_TOKEN='your_token_here'")
        exit(1)
    
    # Set your dataset name here
    dataset_name = "PokeShadowBench"  # Change this to your desired dataset name
    username = os.environ.get("HUGGINGFACE_USERNAME")  # Get username from environment variable
    
    if not username:
        print("Please set your Hugging Face username as an environment variable:")
        print("export HUGGINGFACE_USERNAME='your_username_here'")
        exit(1)
    
    repo_name = f"{username}/{dataset_name}"
    
    # Upload the dataset
    upload_dataset("pokeshadowbench", repo_name) 