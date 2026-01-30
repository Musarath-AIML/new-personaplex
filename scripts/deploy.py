"""
Deploy PersonaPlex model to SageMaker endpoint
"""

import os
import time
import boto3
from sagemaker import get_execution_role, Session
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configuration
REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT_NAME = "personaplex-7b-v1-endpoint"
INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G GPUs, 96GB VRAM
INSTANCE_COUNT = 1

# Read image URI from file (created by build script)
with open("image_uri.txt", "r") as f:
    IMAGE_URI = f.read().strip()

# Model data location (from prepare_model.py)
MODEL_DATA_S3_URI = "s3://your-sagemaker-bucket/personaplex/models/model.tar.gz"

# Initialize SageMaker session
sagemaker_session = Session()
role = get_execution_role()

print("Deployment Configuration:")
print(f"  Region: {REGION}")
print(f"  Role: {role}")
print(f"  Image: {IMAGE_URI}")
print(f"  Model Data: {MODEL_DATA_S3_URI}")
print(f"  Instance Type: {INSTANCE_TYPE}")
print(f"  Endpoint Name: {ENDPOINT_NAME}")
print("")

def create_model():
    """Create SageMaker Model"""
    print("Creating SageMaker Model...")
    
    model = Model(
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA_S3_URI,
        role=role,
        sagemaker_session=sagemaker_session,
        env={
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
            'SAGEMAKER_REGION': REGION,
            # GPU optimizations
            'OMP_NUM_THREADS': '1',
            'NCCL_ASYNC_ERROR_HANDLING': '1'
        }
    )
    
    print(f"Model created: {model.name}")
    return model

def deploy_model(model):
    """Deploy model to endpoint"""
    print(f"\nDeploying to endpoint: {ENDPOINT_NAME}")
    print("This will take 5-10 minutes...")
    
    start_time = time.time()
    
    predictor = model.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        wait=True,
        # Container startup health check configuration
        container_startup_health_check_timeout=600  # 10 minutes
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Deployment successful! ({elapsed/60:.2f} minutes)")
    print(f"✓ Endpoint name: {ENDPOINT_NAME}")
    print(f"✓ Endpoint ARN: {predictor.endpoint_arn}")
    
    return predictor

def test_endpoint(predictor):
    """Test the deployed endpoint"""
    print("\nTesting endpoint...")
    
    test_payload = {
        "inputs": "Hello, how are you doing today?",
        "parameters": {
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    print(f"Input: {test_payload['inputs']}")
    
    response = predictor.predict(test_payload)
    
    print(f"\nResponse:")
    print(f"  Generated text: {response['generated_text']}")
    print(f"  Tokens: {response.get('num_tokens', 'N/A')}")
    
    return response

def get_endpoint_info():
    """Get endpoint details"""
    sm_client = boto3.client('sagemaker', region_name=REGION)
    
    try:
        response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        
        print("\n" + "="*60)
        print("Endpoint Information:")
        print("="*60)
        print(f"  Name: {response['EndpointName']}")
        print(f"  Status: {response['EndpointStatus']}")
        print(f"  ARN: {response['EndpointArn']}")
        print(f"  Instance Type: {INSTANCE_TYPE}")
        print(f"  Instance Count: {INSTANCE_COUNT}")
        print(f"  Created: {response['CreationTime']}")
        print(f"  Last Modified: {response['LastModifiedTime']}")
        print("="*60)
        
    except sm_client.exceptions.ClientError as e:
        print(f"Endpoint not found: {e}")

def main():
    """Main deployment workflow"""
    
    # Check if endpoint already exists
    sm_client = boto3.client('sagemaker', region_name=REGION)
    
    try:
        existing = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"⚠️  Endpoint '{ENDPOINT_NAME}' already exists!")
        print(f"   Status: {existing['EndpointStatus']}")
        
        response = input("Do you want to update it? (yes/no): ")
        if response.lower() != 'yes':
            print("Deployment cancelled.")
            return
        
        print("\nUpdating existing endpoint...")
        
    except sm_client.exceptions.ClientError:
        print("Creating new endpoint...")
    
    # Create and deploy model
    model = create_model()
    predictor = deploy_model(model)
    
    # Test endpoint
    test_endpoint(predictor)
    
    # Show endpoint info
    get_endpoint_info()
    
    print("\n✓ Deployment complete!")
    print(f"\nTo use this endpoint in Python:")
    print(f"""
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = Predictor(
    endpoint_name='{ENDPOINT_NAME}',
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

response = predictor.predict({{
    "inputs": "Your text here",
    "parameters": {{"max_length": 100, "temperature": 0.7}}
}})

print(response['generated_text'])
    """)

if __name__ == "__main__":
    main()
