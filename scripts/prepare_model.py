"""
Download PersonaPlex model from HuggingFace and create model.tar.gz
"""

import os
import tarfile
from pathlib import Path
import boto3
from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this as environment variable
LOCAL_MODEL_DIR = "./model_artifacts"
S3_BUCKET = "your-sagemaker-bucket"  # Change this
S3_PREFIX = "personaplex/models"

def download_model():
    """Download model from HuggingFace Hub"""
    print(f"Downloading {MODEL_ID} from HuggingFace Hub...")
    
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=LOCAL_MODEL_DIR,
        token=HF_TOKEN,
        ignore_patterns=["*.md", "*.txt", ".git*"]
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path

def create_tarball(model_path: str):
    """Create model.tar.gz from downloaded model"""
    print("Creating model.tar.gz...")
    
    tarball_path = "model.tar.gz"
    
    # Create tarball from model directory
    with tarfile.open(tarball_path, "w:gz") as tar:
        # Add all files from model directory
        for item in Path(model_path).rglob("*"):
            if item.is_file():
                # Add file with relative path (no leading directories)
                arcname = item.relative_to(model_path)
                print(f"Adding: {arcname}")
                tar.add(item, arcname=arcname)
    
    tarball_size = os.path.getsize(tarball_path) / (1024**3)  # GB
    print(f"Created {tarball_path} ({tarball_size:.2f} GB)")
    
    return tarball_path

def upload_to_s3(tarball_path: str):
    """Upload model.tar.gz to S3"""
    s3_client = boto3.client('s3')
    s3_key = f"{S3_PREFIX}/model.tar.gz"
    
    print(f"Uploading to s3://{S3_BUCKET}/{s3_key}...")
    
    s3_client.upload_file(
        tarball_path,
        S3_BUCKET,
        s3_key,
        Callback=lambda bytes_transferred: print(
            f"Uploaded {bytes_transferred / (1024**2):.2f} MB",
            end='\r'
        )
    )
    
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"\nModel uploaded to: {s3_uri}")
    
    return s3_uri

def main():
    """Main execution"""
    # Download model
    model_path = download_model()
    
    # Create tarball
    tarball_path = create_tarball(model_path)
    
    # Upload to S3
    s3_uri = upload_to_s3(tarball_path)
    
    print("\n" + "="*50)
    print("Model preparation complete!")
    print(f"S3 URI: {s3_uri}")
    print("="*50)
    
    return s3_uri

if __name__ == "__main__":
    main()
