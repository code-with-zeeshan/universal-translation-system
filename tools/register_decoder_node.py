# tools/register_decoder_node.py
import json
import requests
import uuid
import os
import sys
from urllib.parse import urlparse
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def register_with_redis(node_entry, redis_url):
    """Register the decoder node with Redis using RedisManager"""
    try:
        # Import Redis manager
        from utils.redis_manager import RedisManager
        
        # Get Redis manager instance
        logger.info(f"Using RedisManager to connect to Redis")
        redis_manager = RedisManager.get_instance()
        
        # Override default URL if provided
        if redis_url:
            redis_manager.default_url = redis_url
        
        # Get existing nodes
        nodes = redis_manager.get("decoder_pool:nodes", default=[])
        if not isinstance(nodes, list):
            logger.warning("Invalid data format in Redis, creating new nodes list")
            nodes = []
            
        # Add new node
        nodes.append(node_entry)
        
        # Save back to Redis with retry logic
        success = redis_manager.execute_with_retry(
            redis_manager.set,
            "decoder_pool:nodes",
            nodes
        )
        
        if success:
            logger.info(f"Node registered in Redis with ID: {node_entry['node_id']}")
            return True
        else:
            logger.error("Failed to save node data to Redis")
            return False
    except ImportError:
        logger.error("Redis manager not available. Install with 'pip install redis'")
        return False
    except Exception as e:
        logger.error(f"Failed to register with Redis: {e}")
        return False

def register_with_coordinator(node_entry, coordinator_url, api_key=None):
    """Register the decoder node directly with the coordinator API"""
    if not coordinator_url:
        return False
        
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        logger.info(f"Registering with coordinator at {coordinator_url}")
        response = requests.post(
            f"{coordinator_url}/admin/add_decoder",
            json=node_entry,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Successfully registered with coordinator API")
            return True
        else:
            logger.error(f"Failed to register with coordinator API: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error registering with coordinator: {e}")
        return False

def register_with_file(node_entry, pool_path):
    """Register the decoder node in the local JSON file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        
        # Load existing pool or create new one
        if os.path.exists(pool_path):
            with open(pool_path, "r") as f:
                try:
                    pool_data = json.load(f)
                    if isinstance(pool_data, dict) and "nodes" in pool_data:
                        nodes = pool_data["nodes"]
                    elif isinstance(pool_data, list):
                        nodes = pool_data
                    else:
                        nodes = []
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {pool_path}, creating new file")
                    nodes = []
        else:
            nodes = []
            
        # Add new node
        nodes.append(node_entry)
        
        # Save to file
        with open(pool_path, "w") as f:
            if isinstance(pool_data, dict) and "nodes" in pool_data:
                pool_data["nodes"] = nodes
                json.dump(pool_data, f, indent=2)
            else:
                json.dump({"nodes": nodes, "ab_tests": []}, f, indent=2)
                
        logger.info(f"Node registered in file {pool_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to register with file: {e}")
        return False

def main():
    print("\n=== Universal Translation System: Decoder Node Registration ===\n")
    
    parser = argparse.ArgumentParser(description="Register a decoder node.")
    parser.add_argument("--endpoint", type=str, help="Decoder endpoint (e.g., https://your-decoder.com)")
    parser.add_argument("--region", type=str, default="us-east-1", help="Region (e.g., us-east-1)")
    parser.add_argument("--gpu_type", type=str, default="T4", help="GPU type (e.g., T4, A100)")
    parser.add_argument("--capacity", type=int, default=100, help="Capacity (requests/sec)")
    parser.add_argument("--redis-url", type=str, help="Redis URL (e.g., redis://localhost:6379/0)")
    parser.add_argument("--coordinator-url", type=str, help="Coordinator API URL (e.g., https://coordinator.example.com)")
    parser.add_argument("--api-key", type=str, help="API key for coordinator authentication")
    parser.add_argument("--tags", type=str, help="Comma-separated tags for this node (e.g., 'production,high-memory')")
    args = parser.parse_args()

    # Get Redis URL from environment if not provided
    redis_url = args.redis_url or os.environ.get("REDIS_URL")
    coordinator_url = args.coordinator_url or os.environ.get("COORDINATOR_URL")
    api_key = args.api_key or os.environ.get("COORDINATOR_API_KEY")
    
    node_id = str(uuid.uuid4())
    
    endpoint = args.endpoint or prompt("Decoder endpoint (e.g., https://your-decoder.com)")
    region = args.region or prompt("Region (e.g., us-east-1)", "us-east-1")
    gpu_type = args.gpu_type or prompt("GPU type (e.g., T4, A100)", "T4")
    capacity = args.capacity or int(prompt("Capacity (requests/sec)", "100"))
    tags = args.tags or prompt("Tags (comma-separated, e.g., 'production,high-memory')", "")
    
    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else []

    if not validate_https(endpoint):
        logger.error("❌ Endpoint must use HTTPS.")
        return 1
        
    print(f"Checking health of endpoint {endpoint}...")
    if not check_health(endpoint):
        logger.error(f"❌ Health check failed for {endpoint}/health. Make sure your decoder is running and accessible.")
        return 1

    node_entry = {
        "node_id": node_id,
        "endpoint": endpoint,
        "region": region,
        "gpu_type": gpu_type,
        "capacity": int(capacity),
        "tags": tag_list,
        "healthy": True,  # Initially assume it's healthy since we just checked
        "load": 0,
        "uptime": 0
    }

    # Try different registration methods in order of preference
    registration_success = False
    
    # 1. Try coordinator API first (most direct)
    if coordinator_url:
        print(f"Attempting to register directly with coordinator at {coordinator_url}...")
        if register_with_coordinator(node_entry, coordinator_url, api_key):
            registration_success = True
            print("✅ Successfully registered with coordinator API")
        else:
            print("⚠️ Failed to register with coordinator API, trying Redis...")
    
    # 2. Try Redis next
    if not registration_success and redis_url:
        print(f"Attempting to register with Redis at {redis_url}...")
        if register_with_redis(node_entry, redis_url):
            registration_success = True
            print("✅ Successfully registered with Redis")
        else:
            print("⚠️ Failed to register with Redis, falling back to file...")
    
    # 3. Fall back to file-based registration
    if not registration_success:
        pool_path = os.path.join("configs", "decoder_pool.json")
        print(f"Registering with local file at {pool_path}...")
        if register_with_file(node_entry, pool_path):
            registration_success = True
            print(f"✅ Node registered in file {pool_path}")
        else:
            print("❌ All registration methods failed")
            return 1
    
    print("\n✅ Node registered! Details:")
    print(json.dumps(node_entry, indent=2))
    
    if registration_success:
        print("\nNext steps:")
        if not coordinator_url:
            print("1. Submit a PR to add this node to the official decoder pool")
            print("2. Or, provide this configuration to your coordinator service")
        print(f"Node ID: {node_id}")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())