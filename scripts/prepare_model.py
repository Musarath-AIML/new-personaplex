import os
import subprocess
from pathlib import Path

import boto3
from huggingface_hub import snapshot_download

# ----------------------
# CONFIG
# ----------------------
MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.environ.get("HF_TOKEN")  # export HF_TOKEN=...
LOCAL_MODEL_DIR = "./model_artifacts/personaplex-7b-v1"  # clean, flat dir

S3_BUCKET = "676164205626-sagemaker-us-east-1"           # <-- your bucket
S3_PREFIX = "personaplex/models"                         # key prefix


def download_model() -> str:
    """
    Download PersonaPlex into LOCAL_MODEL_DIR as real files (no symlink cache).
    """
    print(f"Downloading {MODEL_ID} from Hugging Face Hub...")
    print(f"Target local_dir: {LOCAL_MODEL_DIR}")

    Path(LOCAL_MODEL_DIR).parent.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,      # ensure real files
        token=HF_TOKEN,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    print(f"Model downloaded to: {model_path}")

    print("\nFiles and sizes in model directory:")
    for p in sorted(Path(model_path).rglob("*")):
        if p.is_file():
            size = os.path.getsize(p)
            print(f"  {p.relative_to(model_path)}  ({size} bytes)")

    return model_path


def create_tarball(model_path: str) -> str:
    """
    Create model.tar.gz from model_path using a whitelist of files.
    This avoids including blobs/, refs/, .cache/, etc.
    """
    model_path = Path(model_path).resolve()
    tarball_path = Path("model.tar.gz").resolve()

    print(f"\nCreating model.tar.gz from: {model_path}")
    print(f"Output: {tarball_path}")

    # Show disk space
    subprocess.run(["df", "-h", "."], check=False)

    # Files we actually need for inference
    include_files = {
        "config.json",
        "model.safetensors",
        "dist.tgz",
        "voices.tgz",
        "tokenizer_spm_32k_3.model",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
    }

    # Optional: whole figures/ directory
    include_dirs = {"figures"}

    filelist_path = model_path.parent / "filelist.txt"
    with open(filelist_path, "w") as f:
        for p in sorted(model_path.rglob("*")):
            rel = p.relative_to(model_path)
            if p.is_file():
                if rel.name in include_files or rel.parts[0] in include_dirs:
                    f.write(str(rel) + "\n")

    print("\nWill include the following files (from filelist.txt):")
    with open(filelist_path) as f:
        for line in f:
            print("  ", line.strip())

    # Use system tar: tar -czf model.tar.gz -C <model_path> -T filelist.txt
    cmd = ["tar", "-czf", str(tarball_path), "-C", str(model_path), "-T", str(filelist_path)]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    size_bytes = os.path.getsize(tarball_path)
    print(f"\nCreated model.tar.gz ({size_bytes} bytes, {size_bytes / 1024**3:.3f} GB)")
    return str(tarball_path)


def upload_to_s3(tarball_path: str) -> str:
    """
    Upload model.tar.gz to S3 and report its size from S3.
    """
    s3 = boto3.client("s3")
    s3_key = f"{S3_PREFIX}/model.tar.gz"

    print(f"\nUploading to s3://{S3_BUCKET}/{s3_key}...")
    s3.upload_file(tarball_path, S3_BUCKET, s3_key)

    head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    size_bytes = head["ContentLength"]
    print(f"Uploaded {size_bytes} bytes ({size_bytes / 1024**3:.3f} GB) to S3")

    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"S3 URI: {s3_uri}")
    return s3_uri


def main():
    model_path = download_model()          # ./model_artifacts/personaplex-7b-v1
    tarball_path = create_tarball(model_path)
    s3_uri = upload_to_s3(tarball_path)

    print("\n" + "=" * 60)
    print("Model preparation complete!")
    print(f"S3 URI: {s3_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
