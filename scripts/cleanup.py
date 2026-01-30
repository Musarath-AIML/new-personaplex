"""
Delete SageMaker endpoint and resources
"""

import boto3
from sagemaker import Session

ENDPOINT_NAME = "personaplex-7b-v1-endpoint"
REGION = "us-east-1"

sm_client = boto3.client('sagemaker', region_name=REGION)
sagemaker_session = Session()

def delete_endpoint():
    """Delete endpoint"""
    try:
        print(f"Deleting endpoint: {ENDPOINT_NAME}")
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("✓ Endpoint deleted")
    except Exception as e:
        print(f"Error deleting endpoint: {e}")

def delete_endpoint_config():
    """Delete endpoint configuration"""
    try:
        response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        config_name = response['EndpointConfigName']
        
        print(f"Deleting endpoint config: {config_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
        print("✓ Endpoint config deleted")
    except Exception as e:
        print(f"No endpoint config to delete: {e}")

def list_models():
    """List PersonaPlex models"""
    models = sm_client.list_models(NameContains="personaplex")
    
    if models['Models']:
        print("\nRemaining PersonaPlex models:")
        for model in models['Models']:
            print(f"  - {model['ModelName']}")
        
        response = input("\nDelete these models? (yes/no): ")
        if response.lower() == 'yes':
            for model in models['Models']:
                sm_client.delete_model(ModelName=model['ModelName'])
                print(f"✓ Deleted model: {model['ModelName']}")

if __name__ == "__main__":
    delete_endpoint()
    delete_endpoint_config()
    list_models()
    
    print("\n✓ Cleanup complete!")
