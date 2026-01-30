"""
SageMaker inference script for PersonaPlex-7B-v1
Implements the four-function contract for PyTorch DLC
"""

import os
import json
import base64
import logging
from io import BytesIO
from typing import Dict, Any, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load PersonaPlex model, tokenizer, and processor from /opt/ml/model/
    
    This function is called ONCE when the container starts.
    SageMaker automatically extracts model.tar.gz to model_dir.
    
    Args:
        model_dir: Path to /opt/ml/model/ containing extracted model files
    
    Returns:
        Dictionary containing model, tokenizer, and processor
    """
    logger.info(f"Loading PersonaPlex model from {model_dir}")
    logger.info(f"Available files: {os.listdir(model_dir)}")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    logger.info(f"Using device: {device}")
    logger.info(f"Available GPUs: {gpu_count}")
    
    if torch.cuda.is_available():
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    
    try:
        # Load model with automatic device mapping
        logger.info("Loading model with device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load processor if available (for audio processing)
        processor = None
        try:
            logger.info("Attempting to load audio processor")
            processor = AutoProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            logger.info("Audio processor loaded successfully")
        except Exception as e:
            logger.warning(f"No audio processor found: {e}")
        
        model.eval()
        logger.info("Model loaded successfully and set to eval mode")
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'processor': processor,
            'device': device
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise


def input_fn(request_body: bytes, content_type: str = 'application/json') -> Dict[str, Any]:
    """
    Deserialize and prepare the input for inference.
    
    This function is called for EVERY request.
    
    Supported input formats:
    {
        "inputs": "text to process",
        "parameters": {
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": true
        }
    }
    
    Or with audio (base64 encoded):
    {
        "audio": "base64_encoded_audio_bytes",
        "text_prompt": "You are a helpful assistant",
        "parameters": {...}
    }
    
    Args:
        request_body: Raw request bytes
        content_type: Content type header
    
    Returns:
        Parsed input dictionary
    """
    logger.info(f"Parsing input with content_type: {content_type}")
    
    if content_type == 'application/json':
        try:
            input_data = json.loads(request_body)
            logger.info(f"Parsed input keys: {list(input_data.keys())}")
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    elif content_type == 'text/plain':
        # Simple text input
        text = request_body.decode('utf-8')
        return {
            'inputs': text,
            'parameters': {}
        }
    
    else:
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            f"Supported types: application/json, text/plain"
        )


def predict_fn(input_data: Dict[str, Any], model_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run PersonaPlex inference.
    
    This function is called for EVERY request after input_fn.
    
    Args:
        input_data: Parsed input from input_fn
        model_artifacts: Model artifacts from model_fn (cached between requests)
    
    Returns:
        Dictionary with predictions
    """
    logger.info("Starting inference")
    
    # Extract model components
    model = model_artifacts['model']
    tokenizer = model_artifacts['tokenizer']
    processor = model_artifacts.get('processor')
    device = model_artifacts['device']
    
    # Extract input parameters
    text_input = input_data.get('inputs', '')
    audio_input = input_data.get('audio')  # Base64 encoded audio
    text_prompt = input_data.get('text_prompt', '')
    parameters = input_data.get('parameters', {})
    
    # Generation parameters with defaults
    max_length = parameters.get('max_length', 512)
    temperature = parameters.get('temperature', 0.7)
    top_p = parameters.get('top_p', 0.9)
    top_k = parameters.get('top_k', 50)
    do_sample = parameters.get('do_sample', True)
    num_return_sequences = parameters.get('num_return_sequences', 1)
    
    try:
        # Handle text-only inference
        if text_input and not audio_input:
            logger.info(f"Text-only inference: {text_input[:100]}...")
            
            # Tokenize input
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode outputs
            generated_texts = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            result = {
                'generated_text': generated_texts[0] if num_return_sequences == 1 else generated_texts,
                'num_tokens': outputs.shape[1]
            }
        
        # Handle audio + text inference (PersonaPlex speech-to-speech)
        elif audio_input:
            logger.info("Audio inference requested")
            
            if processor is None:
                raise ValueError("Audio processor not available for this model")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_input)
            
            # Process audio (implementation depends on PersonaPlex's audio API)
            # This is a placeholder - adjust based on actual PersonaPlex API
            result = {
                'message': 'Audio processing not fully implemented',
                'received_audio_size': len(audio_bytes),
                'text_prompt': text_prompt
            }
            logger.warning("Audio processing requires PersonaPlex-specific implementation")
        
        else:
            raise ValueError("Must provide either 'inputs' (text) or 'audio'")
        
        logger.info("Inference completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise


def output_fn(predictions: Dict[str, Any], accept: str = 'application/json') -> Tuple[str, str]:
    """
    Serialize the prediction output.
    
    This function is called for EVERY request after predict_fn.
    
    Args:
        predictions: Output from predict_fn
        accept: Accept header from request
    
    Returns:
        Tuple of (serialized_output, content_type)
    """
    logger.info(f"Serializing output with accept: {accept}")
    
    if accept == 'application/json' or accept == '*/*':
        try:
            output = json.dumps(predictions, ensure_ascii=False, indent=2)
            return output, 'application/json'
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise ValueError(f"Failed to serialize predictions: {str(e)}")
    
    else:
        raise ValueError(
            f"Unsupported accept type: {accept}. "
            f"Supported types: application/json"
        )
