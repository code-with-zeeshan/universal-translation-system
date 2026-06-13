# universal-decoder-node/universal_decoder_node/cli.py
import click
import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Optional
import yaml
import torch
from .decoder import create_decoder_service
from .config import DecoderConfig, load_config, save_config

# Optional docker SDK (not always available)
try:
    import docker
except ImportError:
    docker = None

# Load environment variables with defaults
DEFAULT_DECODER_ENDPOINT = os.environ.get("DECODER_ENDPOINT", "http://localhost:8000")
DEFAULT_COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:5100")
DEFAULT_HOST = os.environ.get("DECODER_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("DECODER_PORT", "8000"))
DEFAULT_WORKERS = int(os.environ.get("DECODER_WORKERS", "1"))
DEFAULT_VOCAB_DIR = os.environ.get("VOCAB_DIR", "vocabulary/vocab")


def _detect_gpu() -> Optional[str]:
    """Detect GPU availability and ask user for permission.

    Returns 'cuda', 'mps', or None if user declines or no GPU found.
    The result is cached in DECODER_CONFIG to avoid re-prompting.
    """
    config_path = os.environ.get('DECODER_CONFIG', 'decoder_config.yaml')
    if os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            if cfg.device in ('cuda', 'mps', 'cpu'):
                return cfg.device if cfg.device != 'cpu' else None
        except Exception:
            pass

    try:
        import torch
    except ImportError:
        return None

    gpu_type = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_type = 'cuda'
        click.echo(f"\n🎮 GPU detected: {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_type = 'mps'
        click.echo("\n🎮 Apple Metal GPU detected")

    if gpu_type is None:
        click.echo("ℹ️  No GPU detected — using CPU")
        return None

    click.echo(f"🚀 GPU inference is ~5-10x faster than CPU for translation decoding.")
    use_gpu = click.confirm(f"Use {gpu_type.upper()} for faster inference?", default=True)
    if not use_gpu:
        click.echo("ℹ️  Using CPU (you can change this later in config)")
        return None

    # Save preference to config
    try:
        if os.path.exists(config_path):
            cfg = load_config(config_path)
        else:
            cfg = DecoderConfig()
        cfg.device = gpu_type
        save_config(cfg, config_path)
        click.echo(f"✅ GPU preference saved to {config_path}")
    except Exception:
        pass

    return gpu_type


def _advertise_mdns(host: str, port: int):
    """Advertise decoder via mDNS/Zeroconf so SDKs auto-discover it."""
    try:
        from zeroconf import Zeroconf, ServiceInfo
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        info = ServiceInfo(
            "_universal-translate._tcp.local.",
            f"udn-{port}._universal-translate._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=port,
            properties={"path": "/decode", "version": "1.0.0"},
        )
        zeroconf = Zeroconf()
        zeroconf.register_service(info)
        click.echo(f"📡 Advertising decoder via mDNS ({local_ip}:{port})")
    except ImportError:
        click.echo("ℹ️  Install zeroconf for mDNS discovery: pip install zeroconf")
    except Exception as e:
        click.echo(f"ℹ️  mDNS advertisement skipped: {e}")


@cli.command()
@click.option('--timeout', default=3, type=int, help='Discovery timeout in seconds')
def discover(timeout: int):
    """Discover local decoders on the network via mDNS"""
    try:
        from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange
        from threading import Event

        found = []
        def on_change(zeroconf, service_type, name, state_change):
            if state_change is ServiceStateChange.Added:
                info = zeroconf.get_service_info(service_type, name)
                if info and info.parsed_addresses():
                    addr = info.parsed_addresses()[0]
                    found.append(f"http://{addr}:{info.port}")
                    click.echo(f"  ✅ Decoder at http://{addr}:{info.port}")

        click.echo(f"🔍 Scanning for local decoders ({timeout}s)...")
        z = Zeroconf()
        browser = ServiceBrowser(z, "_universal-translate._tcp.local.", [on_change])
        Event().wait(timeout)
        z.close()

        if not found:
            click.echo("❌ No local decoders found")
        else:
            click.echo(f"\n📡 Use: --local-decoder-url {found[0]} in your SDK")
    except ImportError:
        click.echo("Install zeroconf: pip install zeroconf")


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    udn - Universal Decoder Node

    Deploy a high-performance translation decoder on your own machine or cloud.
    Use 'udn discover' to find running decoders on your network.

    Examples:
      udn start            Run decoder directly (CPU)
      udn serve            Run decoder with LitServe (faster, auto-batching)
      udn docker           Build & run in Docker
      udn discover         Find local decoders via mDNS
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
@click.option('--gpu/--no-gpu', default=None, help='Force GPU or CPU (auto-detect if not set)')
def start(host: str, port: int, workers: int, model_path: Optional[str], 
          vocab_dir: str, config: Optional[str], docker: bool, gpu: Optional[bool]):
    """Start the decoder service"""
    
    if config:
        cfg = load_config(config)
        host = cfg.host or host
        port = cfg.port or port
        workers = cfg.workers or workers
        model_path = cfg.model_path or model_path
        vocab_dir = cfg.vocab_dir or vocab_dir
    
    if gpu is None:
        detected = _detect_gpu()
        if detected:
            os.environ['DECODER_DEVICE'] = detected
    elif gpu:
        os.environ['DECODER_DEVICE'] = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    else:
        os.environ['DECODER_DEVICE'] = 'cpu'
    
    if docker:
        click.echo("🐋 Starting decoder in Docker container...")
        _run_docker(host, port, model_path, vocab_dir)
    else:
        click.echo(f"🚀 Starting decoder service on {host}:{port}...")
        click.echo(f"📊 Workers: {workers}")
        click.echo(f"🧠 Model: {model_path or 'default'}")
        click.echo(f"📚 Vocab directory: {vocab_dir}")
        
        if model_path:
            os.environ['MODEL_PATH'] = model_path
        os.environ['VOCAB_DIR'] = vocab_dir
        
        # Advertise via mDNS so SDKs auto-discover
        _advertise_mdns(host, port)
        
        service = create_decoder_service(model_path, vocab_dir)
        service.run(host=host, port=port, workers=workers)


@cli.command()
@click.option('--host', default=DEFAULT_HOST, help='Host to bind to')
@click.option('--port', default=DEFAULT_PORT, type=int, help='Port to bind to')
@click.option('--model-path', type=click.Path(exists=True), help='Path to model file')
@click.option('--vocab-dir', default=DEFAULT_VOCAB_DIR, type=click.Path(), help='Directory containing vocabulary packs')
@click.option('--max-batch-size', default=8, type=int, help='Max batch size for LitServe auto-batching')
@click.option('--gpu/--no-gpu', default=None, help='Force GPU or CPU (auto-detect if not set)')
def serve(host: str, port: int, model_path: Optional[str], vocab_dir: str, max_batch_size: int, gpu: Optional[bool]):
    """Start decoder with LitServe (2x faster than FastAPI)"""
    import litserve as ls
    from .litserve_decoder import DecoderLitAPI

    click.echo(f"🚀 Starting LitServe decoder on {host}:{port}...")
    click.echo(f"📦 Max batch size: {max_batch_size}")
    click.echo(f"🧠 Model: {model_path or 'default'}")
    click.echo(f"📚 Vocab directory: {vocab_dir}")

    if gpu is None:
        detected = _detect_gpu()
        accelerator = detected if detected else "cpu"
    elif gpu:
        accelerator = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
    else:
        accelerator = "cpu"

    click.echo(f"⚡ Accelerator: {accelerator}")

    if model_path:
        os.environ['MODEL_PATH'] = model_path
    os.environ['VOCAB_DIR'] = vocab_dir

    api = DecoderLitAPI(
        model_path=model_path,
        vocab_dir=vocab_dir,
        max_batch_size=max_batch_size,
        batch_timeout=0.01,
    )
    server = ls.LitServer(api, accelerator=accelerator)
    server.run(port=port)


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
            # Readiness probe
            try:
                ready_resp = requests.get(f"{endpoint}/ready", timeout=5)
                if ready_resp.status_code in (200, 503):
                    ready_data = ready_resp.json()
                    ready_flag = ready_data.get('ready', False)
                    checks = ready_data.get('checks', {})
                    click.echo(f"\n🟢 Ready: {ready_flag}")
                    if not ready_flag:
                        click.echo(f"  Checks: {json.dumps(checks, indent=2)}")
            except Exception:
                click.echo("\n⚠️  Readiness endpoint not available")
            
            # Detailed status
            status_resp = requests.get(f"{endpoint}/status", timeout=5)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                api_ver = status_data.get('api_version') or status_data.get('apiVersion') or 'unknown'
                click.echo(f"\n📊 Detailed Status:")
                click.echo(f"  API Version: {api_ver}")
                click.echo(f"  Model Version: {status_data.get('model_version', 'unknown')}")
                click.echo(f"  Device: {status_data.get('device', 'unknown')}")
                click.echo(f"  GPU Available: {status_data.get('gpu_available', False)}")
                if status_data.get('gpu_name'):
                    click.echo(f"  GPU Name: {status_data['gpu_name']}")
                # Vocabulary info if present
                vocab = status_data.get('vocabulary', {})
                packs = vocab.get('loaded_packs')
                if isinstance(packs, list):
                    click.echo(f"  Vocab Packs Loaded: {len(packs)}")
            else:
                click.echo(f"\n❌ Status endpoint returned {status_resp.status_code}")
    
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
        click.echo("📋 Add this to config/decoder_pool.json in your PR")
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
@click.option('--port', default=8000, type=int, help='Host port to bind')
@click.option('--gpus', is_flag=True, help='Enable GPU support')
def docker(build_only: bool, tag: str, port: int, gpus: bool):
    """Build and run decoder in Docker (simple way)"""
    dockerfile = str(Path(__file__).resolve().parent.parent / 'Dockerfile')
    if not os.path.exists(dockerfile):
        click.echo("❌ Dockerfile not found — run from universal-decoder-node/ directory")
        return

    click.echo(f"🐋 Building Docker image {tag}...")
    build_cmd = ['docker', 'build', '-f', dockerfile, '-t', tag, os.path.dirname(dockerfile)]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"❌ Build failed: {result.stderr}")
        return

    click.echo(f"✅ Built {tag}")
    if build_only:
        return

    click.echo("🚀 Starting container...")
    run_cmd = [
        'docker', 'run', '-d',
        '--name', 'universal-decoder',
        '-p', f'{port}:8000',
    ]
    if gpus:
        run_cmd += ['--gpus', 'all']
    run_cmd.append(tag)

    subprocess.run(run_cmd)
    click.echo(f"✅ Decoder running on http://localhost:{port}")
    click.echo("📊 Logs: docker logs -f universal-decoder")

@cli.command()
@click.option('--build-only', is_flag=True, help='Build image without running')
@click.option('--tag', default='universal-decoder:latest', help='Docker image tag')
def docker_build(build_only: bool, tag: str):
    """(legacy) Build Docker image for decoder"""
    
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
# RUN udn download-models

EXPOSE 8000

CMD ["udn", "start"]
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
                    '-v', './vocabulary/vocab:/app/vocabs',
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