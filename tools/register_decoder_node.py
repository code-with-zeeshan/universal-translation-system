import json
import requests
import uuid
import os
from urllib.parse import urlparse

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
    node_id = str(uuid.uuid4())
    endpoint = prompt("Decoder endpoint (e.g., https://your-decoder.com)")
    region = prompt("Region (e.g., us-east-1)", "us-east-1")
    gpu_type = prompt("GPU type (e.g., T4, A100)", "T4")
    capacity = prompt("Capacity (requests/sec)", "100")

    if not validate_https(endpoint):
        print("❌ Endpoint must use HTTPS.")
        return
    if not check_health(endpoint):
        print(f"❌ Health check failed for {endpoint}/health. Make sure your decoder is running and accessible.")
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