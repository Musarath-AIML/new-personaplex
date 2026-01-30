# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export AWS_REGION=us-east-1
export HF_TOKEN=hf_xxxxxxxxxxxxx  # Your HuggingFace token

# 3. Prepare model (download and upload to S3)
python scripts/prepare_model.py

# 4. Build and push Docker image
./scripts/build_and_push.sh

# 5. Deploy to SageMaker
python scripts/deploy.py