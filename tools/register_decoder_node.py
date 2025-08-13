# tools/register_decoder_node.py
import json
import requests
import uuid
import os
from urllib.parse import urlparse
import argparse
import logging

logger = logging.getLogger(__name__)

def prompt(msg, default=None):
    val = input(f"{msg} [{default}]: ")
    return val.strip() or default

def validate_https(url):
    parsed = urlparse(url)
    return parsed.scheme == 'https'

def check_health(endpoint):
    try:
        resp = requests.get(f"{endpoint}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

def main():
    print("\n=== Universal Translation System: Decoder Node Registration ===\n")
    
    parser = argparse.ArgumentParser(description="Register a decoder node.")
    parser.add_argument("--endpoint", type=str, help="Decoder endpoint (e.g., https://your-decoder.com)")
    parser.add_argument("--region", type=str, default="us-east-1", help="Region (e.g., us-east-1)")
    parser.add_argument("--gpu_type", type=str, default="T4", help="GPU type (e.g., T4, A100)")
    parser.add_argument("--capacity", type=int, default=100, help="Capacity (requests/sec)")
    args = parser.parse_args()

    node_id = str(uuid.uuid4())
    
    endpoint = args.endpoint or prompt("Decoder endpoint (e.g., https://your-decoder.com)")
    region = args.region or prompt("Region (e.g., us-east-1)", "us-east-1")
    gpu_type = args.gpu_type or prompt("GPU type (e.g., T4, A100)", "T4")
    capacity = args.capacity or int(prompt("Capacity (requests/sec)", "100"))

    if not validate_https(endpoint):
        logger.error("❌ Endpoint must use HTTPS.")
        return
    if not check_health(endpoint):
        logger.error(f"❌ Health check failed for {endpoint}/health. Make sure your decoder is running and accessible.")
        return

    node_entry = {
        "node_id": node_id,
        "endpoint": endpoint,
        "region": region,
        "gpu_type": gpu_type,
        "capacity": int(capacity)
    }

    pool_path = os.path.join("configs", "decoder_pool.json")
    if os.path.exists(pool_path):
        with open(pool_path, "r") as f:
            pool = json.load(f)
    else:
        pool = []
    pool.append(node_entry)
    with open(pool_path, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"\n✅ Node registered! Add this entry to your PR for review:")
    print(json.dumps(node_entry, indent=2))

if __name__ == "__main__":
    main()