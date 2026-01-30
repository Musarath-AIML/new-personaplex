# Use SageMaker PyTorch 2.6 DLC with CUDA 12.4
ARG REGION=us-east-1
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus-dev \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for PersonaPlex
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    accelerate==0.27.0 \
    sentencepiece==0.1.99 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    numpy==1.26.4 \
    scipy==1.12.0

# Copy inference code
COPY inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Tell SageMaker which script to use
ENV SAGEMAKER_PROGRAM=inference.py

# Healthcheck (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()"