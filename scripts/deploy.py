import os
import time
import json
import boto3

# ----------------------
# CONFIG (with env overrides)
# ----------------------

REGION = os.getenv("AWS_REGION", "us-east-1")

ACCOUNT_ID = os.getenv("ACCOUNT_ID", "676164205626")

# ECR image URI written by build script, but can be overridden by env
IMAGE_URI = os.getenv("IMAGE_URI")
if IMAGE_URI is None:
    with open("image_uri.txt", "r") as f:
        IMAGE_URI = f.read().strip()

# Model data in S3
MODEL_DATA_S3_URI = os.getenv(
    "MODEL_DATA_S3_URI",
    "s3://676164205626-sagemaker-us-east-1/personaplex/models/model.tar.gz",
)

# SageMaker execution role
ROLE_ARN = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")  # must be set
if not ROLE_ARN:
    raise RuntimeError(
        "SAGEMAKER_EXECUTION_ROLE_ARN is not set. "
        "Export it, e.g.:\n"
        "  export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::676164205626:role/YourSageMakerExecutionRole"
    )

# Names (can override via env)
MODEL_NAME = os.getenv("SAGEMAKER_MODEL_NAME", "personaplex-7b-v1-model")
ENDPOINT_CONFIG_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_CONFIG_NAME", "personaplex-7b-v1-endpoint-config"
)
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "personaplex-7b-v1-endpoint")

# Instance config
INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.g5.8xlarge")
INSTANCE_COUNT = int(os.getenv("SAGEMAKER_INSTANCE_COUNT", "1"))


def create_model(sm_client):
    print(f"Creating model: {MODEL_NAME}")

    container_def = {
        "Image": IMAGE_URI,
        "ModelDataUrl": MODEL_DATA_S3_URI,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
            "SAGEMAKER_REGION": REGION,
            "OMP_NUM_THREADS": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
        },
    }

    try:
        sm_client.describe_model(ModelName=MODEL_NAME)
        print(f"Model {MODEL_NAME} already exists, skipping create.")
    except sm_client.exceptions.ClientError:
        sm_client.create_model(
            ModelName=MODEL_NAME,
            ExecutionRoleArn=ROLE_ARN,
            PrimaryContainer=container_def,
        )
        print(f"Model {MODEL_NAME} created.")


def create_endpoint_config(sm_client):
    print(f"Creating endpoint config: {ENDPOINT_CONFIG_NAME}")

    production_variant = {
        "VariantName": "AllTraffic",
        "ModelName": MODEL_NAME,
        "InitialInstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "InitialVariantWeight": 1.0,
    }

    try:
        sm_client.describe_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        print(f"Endpoint config {ENDPOINT_CONFIG_NAME} already exists, skipping create.")
    except sm_client.exceptions.ClientError:
        sm_client.create_endpoint_config(
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
            ProductionVariants=[production_variant],
        )
        print(f"Endpoint config {ENDPOINT_CONFIG_NAME} created.")


def create_or_update_endpoint(sm_client):
    print(f"Creating/updating endpoint: {ENDPOINT_NAME}")

    try:
        resp = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        print(f"Endpoint {ENDPOINT_NAME} exists with status {status}, updating config...")

        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
        )
    except sm_client.exceptions.ClientError:
        print(f"Endpoint {ENDPOINT_NAME} does not exist, creating...")
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
        )

    wait_for_endpoint(sm_client)


def wait_for_endpoint(sm_client):
    print("Waiting for endpoint to be InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    start = time.time()
    waiter.wait(EndpointName=ENDPOINT_NAME)
    elapsed = (time.time() - start) / 60.0

    resp = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"\nEndpoint status: {resp['EndpointStatus']}")
    print(f"Deployment took {elapsed:.2f} minutes")


def test_endpoint(rt_client):
    print("\nTesting endpoint with a simple text prompt...")

    payload = {
        "inputs": "Hello, how are you today?",
        "parameters": {
            "max_length": 80,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    }

    response = rt_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    body = response["Body"].read().decode("utf-8")
    print("\nRaw response body:")
    print(body)


def main():
    sm_client = boto3.client("sagemaker", region_name=REGION)
    rt_client = boto3.client("sagemaker-runtime", region_name=REGION)

    print("Configuration:")
    print(f"  Region:        {REGION}")
    print(f"  Role:          {ROLE_ARN}")
    print(f"  Image URI:     {IMAGE_URI}")
    print(f"  Model data:    {MODEL_DATA_S3_URI}")
    print(f"  Model name:    {MODEL_NAME}")
    print(f"  Endpoint name: {ENDPOINT_NAME}")
    print(f"  Instance type: {INSTANCE_TYPE}")
    print(f"  Instance count:{INSTANCE_COUNT}")
    print("")

    create_model(sm_client)
    create_endpoint_config(sm_client)
    create_or_update_endpoint(sm_client)
    test_endpoint(rt_client)


if __name__ == "__main__":
    main()
