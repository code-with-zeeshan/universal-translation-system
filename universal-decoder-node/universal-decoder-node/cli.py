# universal-decoder-node/universal_decoder_node/cli.py
import click
import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Optional
import docker
import yaml
from .decoder import create_decoder_service
from .config import DecoderConfig, load_config, save_config

# Load environment variables with defaults
DEFAULT_DECODER_ENDPOINT = os.environ.get("DECODER_ENDPOINT", "http://localhost:8000")
DEFAULT_COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:5100")
DEFAULT_HOST = os.environ.get("DECODER_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("DECODER_PORT", "8000"))
DEFAULT_WORKERS = int(os.environ.get("DECODER_WORKERS", "1"))
DEFAULT_VOCAB_DIR = os.environ.get("VOCAB_DIR", "vocabs")


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    Universal Decoder Node - High-performance translation decoder service

    Usage Modes:
    - Personal/Private: Run the decoder on your own device or cloud for private translation/testing. Registration is not required.
    - Contributing: If you want to support the project by adding your node to the public decoder pool, follow the registration steps.
    """
    pass


@cli.command()
@click.option('--host', default=DEFAULT_HOST, help='Host to bind to')
@click.option('--port', default=DEFAULT_PORT, type=int, help='Port to bind to')
@click.option('--workers', default=DEFAULT_WORKERS, type=int, help='Number of worker processes')
@click.option('--model-path', type=click.Path(exists=True), help='Path to model file')
@click.option('--vocab-dir', default=DEFAULT_VOCAB_DIR, type=click.Path(), help='Directory containing vocabulary packs')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--docker', is_flag=True, help='Run in Docker container')
def start(host: str, port: int, workers: int, model_path: Optional[str], 
          vocab_dir: str, config: Optional[str], docker: bool):
    """Start the decoder service"""
    
    if config:
        cfg = load_config(config)
        host = cfg.host or host
        port = cfg.port or port
        workers = cfg.workers or workers
        model_path = cfg.model_path or model_path
        vocab_dir = cfg.vocab_dir or vocab_dir
    
    if docker:
        click.echo("🐋 Starting decoder in Docker container...")
        _run_docker(host, port, model_path, vocab_dir)
    else:
        click.echo(f"🚀 Starting decoder service on {host}:{port}...")
        click.echo(f"📊 Workers: {workers}")
        click.echo(f"🧠 Model: {model_path or 'default'}")
        click.echo(f"📚 Vocab directory: {vocab_dir}")
        
        # Set environment variables
        if model_path:
            os.environ['MODEL_PATH'] = model_path
        os.environ['VOCAB_DIR'] = vocab_dir
        
        # Create and run service
        service = create_decoder_service(model_path, vocab_dir)
        service.run(host=host, port=port, workers=workers)


@cli.command()
@click.option('--endpoint', default=DEFAULT_DECODER_ENDPOINT, help='Decoder endpoint')
@click.option('--detailed', is_flag=True, help='Show detailed status')
def status(endpoint: str, detailed: bool):
    """Check decoder status"""
    try:
        # Health check
        health_resp = requests.get(f"{endpoint}/health", timeout=5)
        health_data = health_resp.json()
        
        if health_resp.status_code == 200:
            click.echo("✅ Decoder is healthy")
        else:
            click.echo("❌ Decoder is unhealthy")
        
        if detailed:
            # Get detailed status
            status_resp = requests.get(f"{endpoint}/status", timeout=5)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                click.echo(f"\n📊 Detailed Status:")
                click.echo(f"  Model Version: {status_data.get('model_version', 'unknown')}")
                click.echo(f"  Device: {status_data.get('device', 'unknown')}")
                click.echo(f"  GPU Available: {status_data.get('gpu_available', False)}")
                if status_data.get('gpu_name'):
                    click.echo(f"  GPU Name: {status_data['gpu_name']}")
    
    except requests.exceptions.ConnectionError:
        click.echo("❌ Cannot connect to decoder service")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--name', prompt='Node name', help='Unique name for this decoder node')
@click.option('--endpoint', prompt='Decoder endpoint', help='HTTPS endpoint for your decoder')
@click.option('--region', prompt='Region', default='us-east-1', help='AWS region or location')
@click.option('--gpu-type', prompt='GPU type', default='T4', help='GPU type (T4, V100, A100, etc)')
@click.option('--capacity', prompt='Capacity', default=100, type=int, help='Requests per second capacity')
@click.option('--coordinator-url', default=DEFAULT_COORDINATOR_URL, help='Coordinator service URL')
@click.option('--output', type=click.Path(), help='Save registration to file instead of registering')
def register(name: str, endpoint: str, region: str, gpu_type: str, 
             capacity: int, coordinator_url: str, output: Optional[str]):
    """Register decoder node with coordinator"""
    
    # Validate endpoint
    if not endpoint.startswith('https://'):
        click.echo("⚠️  Warning: Endpoint should use HTTPS for production")
    
    # Check health
    click.echo(f"🔍 Checking decoder health at {endpoint}...")
    try:
        resp = requests.get(f"{endpoint}/health", timeout=5)
        if resp.status_code != 200:
            click.echo(f"❌ Decoder health check failed: {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Cannot reach decoder: {e}")
        sys.exit(1)
    
    click.echo("✅ Decoder is healthy")
    
    # Create registration
    import uuid
    registration = {
        "node_id": f"{name}-{uuid.uuid4().hex[:8]}",
        "endpoint": endpoint,
        "region": region,
        "gpu_type": gpu_type,
        "capacity": capacity
    }
    
    if output:
        # Save to file
        with open(output, 'w') as f:
            json.dump(registration, f, indent=2)
        click.echo(f"💾 Registration saved to {output}")
        click.echo("📋 Add this to configs/decoder_pool.json in your PR")
    else:
        # Register with coordinator
        try:
            resp = requests.post(
                f"{coordinator_url}/api/add_decoder",
                json=registration,
                headers={"Authorization": f"Bearer {os.environ.get('COORDINATOR_TOKEN', '')}"}
            )
            if resp.status_code == 200:
                click.echo("✅ Successfully registered with coordinator")
                click.echo(f"🆔 Node ID: {registration['node_id']}")
            else:
                click.echo(f"❌ Registration failed: {resp.text}")
        except Exception as e:
            click.echo(f"❌ Failed to register: {e}")
            click.echo("\n💡 You can save registration locally and submit via PR:")
            click.echo(f"   universal-decoder-node register ... --output registration.json")


@cli.command()
@click.option('--text', prompt='Text to translate', help='Text to translate')
@click.option('--source-lang', prompt='Source language', default='en', help='Source language code')
@click.option('--target-lang', prompt='Target language', default='es', help='Target language code')
@click.option('--endpoint', default='http://localhost:8000', help='Decoder endpoint')
@click.option('--encoder-endpoint', help='Encoder endpoint (if using remote encoder)')
def test(text: str, source_lang: str, target_lang: str, endpoint: str, encoder_endpoint: Optional[str]):
    """Test translation with the decoder"""
    
    click.echo(f"🔤 Text: {text}")
    click.echo(f"🌐 Translation: {source_lang} → {target_lang}")
    
    try:
        if encoder_endpoint:
            # Use remote encoder
            click.echo(f"📡 Using remote encoder: {encoder_endpoint}")
            
            # Implement remote encoder call
            try:
                # Prepare request data
                encoder_data = {
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang
                }
                
                # Call remote encoder
                encoder_resp = requests.post(
                    f"{encoder_endpoint}/encode",
                    json=encoder_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if encoder_resp.status_code == 200:
                    encoder_result = encoder_resp.json()
                    compressed_data = encoder_result.get('encoded_data')
                    
                    if compressed_data:
                        # Send compressed data to decoder
                        resp = requests.post(
                            f"{endpoint}/decode",
                            data=compressed_data,
                            headers={'X-Target-Language': target_lang}
                        )
                        
                        if resp.status_code == 200:
                            result = resp.json()
                            click.echo(f"✅ Translation: {result.get('translation', 'N/A')}")
                        else:
                            click.echo(f"❌ Decoder failed: {resp.text}")
                    else:
                        click.echo("❌ Encoder returned no data")
                else:
                    click.echo(f"❌ Encoder failed: {encoder_resp.text}")
                    
            except requests.exceptions.RequestException as e:
                click.echo(f"❌ Network error: {e}")
            except Exception as e:
                click.echo(f"❌ Encoder error: {e}")
        else:
            # For testing, create dummy compressed data
            import numpy as np
            import lz4.frame
            
            # Create fake encoder output
            hidden_states = np.random.randn(1, 10, 1024).astype(np.float32)
            scale = 127.0
            quantized = (hidden_states * scale).astype(np.int8)
            
            # Create compressed data
            metadata = bytearray()
            metadata.extend(hidden_states.shape[1].to_bytes(4, 'little'))
            metadata.extend(hidden_states.shape[2].to_bytes(4, 'little'))
            metadata.extend(np.float32(scale).tobytes())
            
            compressed = lz4.frame.compress(quantized.tobytes())
            compressed_data = bytes(metadata) + compressed
            
            # Send to decoder
            resp = requests.post(
                f"{endpoint}/decode",
                data=compressed_data,
                headers={'X-Target-Language': target_lang}
            )
            
            if resp.status_code == 200:
                result = resp.json()
                click.echo(f"✅ Translation: {result.get('translation', 'N/A')}")
            else:
                click.echo(f"❌ Translation failed: {resp.text}")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--output', default='config.yaml', type=click.Path(), help='Output configuration file')
def init(output: str):
    """Initialize configuration file"""
    
    config = {
        'decoder': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1,
            'model_path': None,
            'vocab_dir': 'vocabs',
            'device': 'cuda',
            'max_batch_size': 64,
            'batch_timeout_ms': 10
        },
        'security': {
            'jwt_secret': os.environ.get('JWT_SECRET') or '',  # require override
            'enable_auth': True
        },
        'monitoring': {
            'prometheus_port': 9200,
            'enable_tracing': False
        }
    }
    
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    click.echo(f"✅ Configuration file created: {output}")
    click.echo("📝 Edit this file to customize your decoder settings")


@cli.command()
@click.option('--build-only', is_flag=True, help='Build image without running')
@click.option('--tag', default='universal-decoder:latest', help='Docker image tag')
def docker_build(build_only: bool, tag: str):
    """Build Docker image for decoder"""
    
    dockerfile_content = '''FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install package
COPY . /app/
RUN pip install --no-cache-dir -e .

# Download models (optional)
# RUN universal-decoder-node download-models

EXPOSE 8000

CMD ["universal-decoder-node", "start"]
'''
    
    # Save Dockerfile
    with open('Dockerfile.decoder', 'w') as f:
        f.write(dockerfile_content)
    
    click.echo(f"🐋 Building Docker image: {tag}")
    
    try:
        result = subprocess.run(
            ['docker', 'build', '-f', 'Dockerfile.decoder', '-t', tag, '.'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            click.echo(f"✅ Successfully built {tag}")
            
            if not build_only:
                click.echo("🚀 Starting container...")
                subprocess.run([
                    'docker', 'run', '-d',
                    '--name', 'universal-decoder',
                    '--gpus', 'all',
                    '-p', '8000:8000',
                    '-v', './models:/app/models',
                    '-v', './vocabs:/app/vocabs',
                    tag
                ])
        else:
            click.echo(f"❌ Build failed: {result.stderr}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
    finally:
        # Cleanup
        if os.path.exists('Dockerfile.decoder'):
            os.remove('Dockerfile.decoder')


def _run_docker(host: str, port: int, model_path: Optional[str], vocab_dir: str):
    """Run decoder in Docker container"""
    client = docker.from_env()
    
    # Check if image exists
    try:
        client.images.get('universal-decoder:latest')
    except docker.errors.ImageNotFound:
        click.echo("🐋 Image not found, building...")
        subprocess.run(['docker', 'build', '-t', 'universal-decoder:latest', '.'])
    
    # Run container
    volumes = {}
    if model_path:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        volumes[model_dir] = {'bind': '/app/models', 'mode': 'ro'}
    
    vocab_dir_abs = os.path.abspath(vocab_dir)
    volumes[vocab_dir_abs] = {'bind': '/app/vocabs', 'mode': 'ro'}
    
    container = client.containers.run(
        'universal-decoder:latest',
        detach=True,
        ports={f'{port}/tcp': (host, port)},
        volumes=volumes,
        environment={
            'MODEL_PATH': f"/app/models/{os.path.basename(model_path)}" if model_path else '',
            'VOCAB_DIR': '/app/vocabs'
        },
        runtime='nvidia'  # For GPU support
    )
    
    click.echo(f"✅ Container started: {container.short_id}")
    click.echo(f"📊 Logs: docker logs -f {container.short_id}")


if __name__ == '__main__':
    cli()