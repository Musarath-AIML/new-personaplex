#!/bin/bash

# Configuration
REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_NAME="personaplex-sagemaker-inference"
IMAGE_TAG="latest"

# Full image URI
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Building and pushing Docker image..."
echo "Account: $ACCOUNT_ID"
echo "Region: $REGION"
echo "Image URI: $IMAGE_URI"
echo ""

# Create ECR repository if it doesn't exist
echo "Creating ECR repository..."
aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --region ${REGION} || \
    aws ecr create-repository --repository-name ${REPOSITORY_NAME} --region ${REGION}

# Authenticate Docker to ECR
echo "Authenticating Docker to ECR..."
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build the Docker image
echo "Building Docker image..."
cd docker
docker build \
    --build-arg REGION=${REGION} \
    --platform linux/amd64 \
    -t ${REPOSITORY_NAME}:${IMAGE_TAG} \
    -f Dockerfile .

# Tag the image
echo "Tagging image..."
docker tag ${REPOSITORY_NAME}:${IMAGE_TAG} ${IMAGE_URI}

# Push to ECR
echo "Pushing image to ECR..."
docker push ${IMAGE_URI}

echo ""
echo "============================================"
echo "Image successfully pushed!"
echo "Image URI: ${IMAGE_URI}"
echo "============================================"
echo ""

# Save image URI to file for deployment script
echo ${IMAGE_URI} > ../image_uri.txt
