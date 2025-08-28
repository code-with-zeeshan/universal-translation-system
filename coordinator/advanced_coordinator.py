# coordinator/advanced_coordinator.py
import etcd3
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time
import logging
import os
import threading

import random
from datetime import datetime, timedelta
from functools import wraps

import httpx
import jwt
from fastapi import (FastAPI, Request, Depends, HTTPException, Form, Header, Response, status, APIRouter)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jinja2 import Template
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import Counter, Gauge, generate_latest
from pydantic import BaseModel, Field
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from starlette.middleware.sessions import SessionMiddleware

# Import utility modules
from utils.auth import APIKeyManager
from utils.rate_limiter import RateLimiter
from utils.exceptions import ConfigurationError, NetworkError
from utils.security import validate_model_source, safe_load_model

# --- Configuration and Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration paths
POOL_PATH = os.environ.get("POOL_CONFIG_PATH", os.path.join("configs", "decoder_pool.json"))

# Version information
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

# Security keys and tokens
SECRET_KEY = os.environ.get("COORDINATOR_SECRET", "a-very-secret-key-for-cookies")
JWT_SECRET = os.environ.get("COORDINATOR_JWT_SECRET", "a-super-secret-jwt-key")
AUTH_TOKEN = os.environ.get("COORDINATOR_TOKEN", "changeme123")
INTERNAL_AUTH_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", "internal-secret-token-for-service-auth")

# API configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "5100"))
API_WORKERS = int(os.environ.get("API_WORKERS", "1"))
API_TITLE = os.environ.get("API_TITLE", "Universal Translation Coordinator")

# Add these constants
ETCD_HOST = os.environ.get("ETCD_HOST", "localhost")
ETCD_PORT = int(os.environ.get("ETCD_PORT", "2379"))
USE_ETCD = os.environ.get("USE_ETCD", "false").lower() == "true"
SERVICE_TTL = int(os.environ.get("SERVICE_TTL", "60"))  # seconds
ETCD_PREFIX = os.environ.get("ETCD_PREFIX", "/universal-translation/decoders/")

# Mirroring configuration
try:
    MIRROR_INTERVAL_SECONDS = int(os.environ.get("COORDINATOR_MIRROR_INTERVAL", "60"))
except ValueError:
    logger.warning("Invalid COORDINATOR_MIRROR_INTERVAL; defaulting to 60s")
    MIRROR_INTERVAL_SECONDS = 60
# Enforce a sensible minimum
if MIRROR_INTERVAL_SECONDS < 5:
    logger.warning(f"COORDINATOR_MIRROR_INTERVAL too low ({MIRROR_INTERVAL_SECONDS}s); clamping to 5s")
    MIRROR_INTERVAL_SECONDS = 5
logger.info(f"Coordinator mirror interval: {MIRROR_INTERVAL_SECONDS}s")

# Initialize utilities
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)

# --- Pydantic Models for API Schema ---
class DecoderNodeSchema(BaseModel):
    node_id: str
    endpoint: str
    region: str
    gpu_type: str
    capacity: int
    healthy: bool = Field(default=False)
    load: int = Field(default=0)
    uptime: int = Field(default=0)

class StatusSchema(BaseModel):
    model_version: str
    decoder_pool_size: int
    healthy_decoders: int
    decoders: List[DecoderNodeSchema]

# --- FastAPI App Setup ---
app = FastAPI(
    title="Universal Translation Coordinator",
    version=MODEL_VERSION,
    description="Coordinates requests across a pool of decoder nodes."
)

# Service discovery API endpoints
@app.post("/api/v1/register")
async def register_decoder(node: DecoderNodeSchema):
    """Register a decoder node with the pool."""
    try:
        node_id = await pool.register_node(node.dict())
        return {"success": True, "node_id": node_id}
    except Exception as e:
        logger.error(f"Error registering node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/unregister/{node_id}")
async def unregister_decoder(node_id: str):
    """Unregister a decoder node from the pool."""
    try:
        await pool.unregister_node(node_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error unregistering node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/node/{node_id}")
async def get_node(node_id: str):
    """Get information about a specific node."""
    node = await pool.get_node_by_id(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@app.put("/api/v1/node/{node_id}/status")
async def update_node_status(
    node_id: str, 
    healthy: bool = Form(...), 
    load: int = Form(...)
):
    """Update a node's status."""
    try:
        await pool.update_node_status(node_id, healthy, load)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating node status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenTelemetry Tracing ---
trace.set_tracer_provider(TracerProvider(resource=Resource.create({SERVICE_NAME: "coordinator-service"})))
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)

# --- Prometheus Metrics ---
requests_total = Counter('coordinator_requests_total', 'Total requests received', ['endpoint', 'group'])
requests_errors = Counter('coordinator_requests_errors', 'Total errors', ['endpoint', 'group'])
decoder_active = Gauge('coordinator_decoder_active', 'Active decoders')
decoder_load = Gauge('coordinator_decoder_load', 'Current load per decoder', ['node_id'])

# --- Live Config Reloading ---
# Use a thread-safe event to signal a reload request from the watchdog thread to the main async loop
CONFIG_NEEDS_RELOAD = threading.Event()

class PoolReloadHandler(FileSystemEventHandler):
    """Handles file system events for the decoder pool configuration."""
    def on_modified(self, event):
        if event.src_path.endswith(os.path.basename(POOL_PATH)):
            logger.info(f"Configuration file {event.src_path} changed. Scheduling reload.")
            # Set the event to signal the main loop to reload
            CONFIG_NEEDS_RELOAD.set()

# --- Decoder Pool Management ---
class DecoderPool:
    def __init__(self, pool_path=POOL_PATH, redis_url=None):
        self.pool_path = pool_path
        self.lock = asyncio.Lock()
        self.nodes: List[Dict] = [] # Renamed from 'pool' for clarity
        self.ab_tests: List[Dict] = []
        self.client = httpx.AsyncClient(timeout=2.0)
        
        # Redis configuration
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self.use_redis = False

        # Service discovery with etcd
        self.use_etcd = USE_ETCD
        self.etcd_client = None
        self.service_ttl = SERVICE_TTL
        self.lease = None
        self.node_watchers = {}
        self.discovery_thread = None
        
        if self.use_etcd:
            self._setup_etcd()
        
        # Load initial configuration
        self._load_config()
        
        # Start background health check
        self.health_check_task = None
        
        # Try to initialize Redis if URL is provided
        if self.redis_url:
            try:
                # Import Redis manager
                from utils.redis_manager import RedisManager
                self.redis_manager = RedisManager.get_instance()
                
                # Test Redis connection
                if self.redis_manager.get_client():
                    self.use_redis = True
                    logger.info(f"Redis connection established via RedisManager")
                else:
                    logger.warning("Failed to connect to Redis via RedisManager")
            except ImportError:
                logger.warning("Redis manager not available. Install with 'pip install redis'")
            except Exception as e:
                logger.error(f"Failed to initialize Redis manager: {e}")
        
        # Initial load of configuration
        self.reload_sync()
        
    def _setup_etcd(self):
        """Set up etcd client for service discovery"""
        try:
            self.etcd_client = etcd3.client(host=ETCD_HOST, port=ETCD_PORT)
            self.lease = self.etcd_client.lease(self.service_ttl)
            
            # Start discovery thread
            self.discovery_thread = threading.Thread(
                target=self._run_discovery_loop,
                daemon=True
            )
            self.discovery_thread.start()
            
            logger.info(f"Connected to etcd at {ETCD_HOST}:{ETCD_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to etcd: {e}")
            self.use_etcd = False
    
    def _run_discovery_loop(self):
        """Background thread for service discovery"""
        while True:
            try:
                # Refresh lease
                self.lease.refresh()
                
                # Discover nodes
                self._discover_nodes()
                
                # Sleep
                time.sleep(self.service_ttl / 2)
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(5)  # Backoff on error
    
    def _discover_nodes(self):
        """Discover decoder nodes from etcd"""
        if not self.use_etcd or not self.etcd_client:
            return
            
        try:
            # Get all nodes
            nodes = []
            for value, metadata in self.etcd_client.get_prefix(ETCD_PREFIX):
                if value:
                    try:
                        node_data = json.loads(value.decode('utf-8'))
                        nodes.append(node_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid node data: {value}")
            
            # Update nodes (thread-safe)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._update_nodes_from_discovery(nodes))
            loop.close()
        except Exception as e:
            logger.error(f"Error discovering nodes: {e}")
    
    async def _update_nodes_from_discovery(self, discovered_nodes: List[Dict]):
        """Update nodes from discovery (thread-safe)"""
        async with self.lock:
            # Map existing nodes by ID
            existing_nodes = {node['node_id']: node for node in self.nodes}
            
            # Update existing nodes and add new ones
            updated_nodes = []
            for node in discovered_nodes:
                node_id = node['node_id']
                if node_id in existing_nodes:
                    # Preserve load and health status
                    node['load'] = existing_nodes[node_id].get('load', 0)
                    node['healthy'] = existing_nodes[node_id].get('healthy', False)
                    updated_nodes.append(node)
                else:
                    # New node, initialize with default values
                    node['load'] = 0
                    node['healthy'] = False
                    updated_nodes.append(node)
            
            # Keep nodes that are in the config file but not discovered
            for node in self.nodes:
                node_id = node['node_id']
                if not any(n['node_id'] == node_id for n in discovered_nodes):
                    # Only keep if it's from the config file
                    if node.get('from_config', False):
                        updated_nodes.append(node)
            
            self.nodes = updated_nodes
            logger.info(f"Updated nodes from discovery: {len(self.nodes)} nodes")
    
    async def register_node(self, node: Dict):
        """Register a decoder node with the pool"""
        node_id = node.get('node_id')
        if not node_id:
            node_id = str(uuid.uuid4())
            node['node_id'] = node_id
        
        # Add to local pool
        async with self.lock:
            # Check if node already exists
            for i, existing in enumerate(self.nodes):
                if existing['node_id'] == node_id:
                    # Update existing node
                    self.nodes[i] = {**existing, **node, 'healthy': True}
                    logger.info(f"Updated existing node: {node_id}")
                    break
            else:
                # Add new node
                node['healthy'] = True
                node['load'] = 0
                node['from_config'] = False
                self.nodes.append(node)
                logger.info(f"Added new node: {node_id}")
        
        # Register with etcd if enabled
        if self.use_etcd and self.etcd_client and self.lease:
            try:
                key = f"{ETCD_PREFIX}{node_id}"
                value = json.dumps(node)
                self.etcd_client.put(key, value, lease=self.lease)
                logger.info(f"Registered node with etcd: {node_id}")
            except Exception as e:
                logger.error(f"Failed to register node with etcd: {e}")
        
        return node_id
    
    async def unregister_node(self, node_id: str):
        """Unregister a decoder node from the pool"""
        # Remove from local pool
        async with self.lock:
            self.nodes = [node for node in self.nodes if node['node_id'] != node_id]
            logger.info(f"Removed node: {node_id}")
        
        # Unregister from etcd if enabled
        if self.use_etcd and self.etcd_client:
            try:
                key = f"{ETCD_PREFIX}{node_id}"
                self.etcd_client.delete(key)
                logger.info(f"Unregistered node from etcd: {node_id}")
            except Exception as e:
                logger.error(f"Failed to unregister node from etcd: {e}")    

    def reload_sync(self):
        """Synchronously reload configuration (used during initialization)"""
        try:
            if self.use_redis:
                # Try Redis first using the manager
                from utils.redis_manager import RedisManager
                rm = RedisManager.get_instance()
                
                nodes_data = rm.get("decoder_pool:nodes")
                ab_tests_data = rm.get("decoder_pool:ab_tests")
                
                if nodes_data:
                    self.nodes = nodes_data
                    logger.info(f"Loaded {len(self.nodes)} nodes from Redis")
                else:
                    # If Redis has no data, try loading from disk
                    logger.info("No nodes found in Redis, trying disk")
                    self._load_from_disk()
                    # If we loaded from disk, save to Redis for next time
                    if self.nodes:
                        rm.set("decoder_pool:nodes", self.nodes)
                        logger.info(f"Saved {len(self.nodes)} nodes to Redis")
                
                if ab_tests_data:
                    self.ab_tests = ab_tests_data
                    logger.info(f"Loaded {len(self.ab_tests)} A/B tests from Redis")
                else:
                    # If Redis has no A/B test data, use what we loaded from disk
                    if not self.ab_tests:  # Only if we haven't loaded them yet
                        self._load_ab_tests_from_disk()
                    # Save to Redis for next time
                    if self.ab_tests:
                        rm.set("decoder_pool:ab_tests", self.ab_tests)
                        logger.info(f"Saved {len(self.ab_tests)} A/B tests to Redis")
                
                # Always mirror to disk so fallback stays up-to-date
                try:
                    os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
                    with open(self.pool_path, "w") as f:
                        json.dump({
                            "nodes": self.nodes,
                            "ab_tests": self.ab_tests
                        }, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to mirror decoder pool to disk: {e}")
            else:
                # No Redis, load from disk
                self._load_from_disk()
        except Exception as e:
            logger.error(f"Error during sync reload: {e}")
            # Fallback to disk
            self._load_from_disk()

    def _load_from_disk(self):
        """Synchronously loads the pool nodes and AB tests configuration from the JSON file."""
        try:
            if os.path.exists(self.pool_path):
                with open(self.pool_path, "r") as f:
                    config_data = json.load(f)
                    # Load both nodes and A/B tests
                    self.nodes = config_data.get("nodes", [])
                    self.ab_tests = config_data.get("ab_tests", [])
                logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.ab_tests)} A/B tests from disk")
            else:
                self.nodes, self.ab_tests = [], []
                logger.warning(f"Pool configuration file {self.pool_path} not found, using empty configuration")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse decoder pool file {self.pool_path}: {e}")
            self.nodes, self.ab_tests = [], []

    def _load_ab_tests_from_disk(self):
        """Load only A/B tests from disk (used when Redis has nodes but no A/B tests)"""
        try:
            if os.path.exists(self.pool_path):
                with open(self.pool_path, "r") as f:
                    config_data = json.load(f)
                    self.ab_tests = config_data.get("ab_tests", [])
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load A/B tests from disk: {e}")
            self.ab_tests = []

    def mirror_redis_to_disk(self) -> bool:
        """Mirror the latest Redis state to disk without mutating in-memory state."""
        try:
            if not self.use_redis:
                return False
            try:
                from utils.redis_manager import RedisManager
                rm = getattr(self, "redis_manager", None) or RedisManager.get_instance()
            except Exception:
                return False
            if not rm or not rm.get_client():
                return False

            nodes_data = rm.get("decoder_pool:nodes") or []
            ab_tests_data = rm.get("decoder_pool:ab_tests") or []

            os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
            with open(self.pool_path, "w") as f:
                json.dump({"nodes": nodes_data, "ab_tests": ab_tests_data}, f, indent=2)
            logger.info(f"Mirrored Redis decoder pool to disk: nodes={len(nodes_data)}, ab_tests={len(ab_tests_data)}")
            return True
        except Exception as e:
            logger.warning(f"Mirror Redis->disk failed: {e}")
            return False

    async def _load_from_redis(self):
        """Asynchronously load configuration from Redis (via RedisManager)"""
        try:
            from utils.redis_manager import RedisManager
            rm = RedisManager.get_instance()
            if not self.use_redis or not rm.get_client():
                logger.warning("Redis not configured, falling back to disk")
                self._load_from_disk()
                return

            # Get nodes and A/B tests from Redis (already deserialized)
            nodes_data = rm.get("decoder_pool:nodes")
            ab_tests_data = rm.get("decoder_pool:ab_tests")

            if nodes_data:
                self.nodes = nodes_data
                logger.info(f"Loaded {len(self.nodes)} nodes from Redis")
            else:
                # If Redis has no data, try loading from disk
                logger.info("No nodes found in Redis, trying disk")
                self._load_from_disk()
                # If we loaded from disk, save to Redis for next time
                if self.nodes:
                    rm.set("decoder_pool:nodes", self.nodes)
                    logger.info(f"Saved {len(self.nodes)} nodes to Redis")

            if ab_tests_data:
                self.ab_tests = ab_tests_data
                logger.info(f"Loaded {len(self.ab_tests)} A/B tests from Redis")
            else:
                # If Redis has no A/B test data, use what we loaded from disk
                if not self.ab_tests:  # Only if we haven't loaded them yet
                    self._load_ab_tests_from_disk()
                # Save to Redis for next time
                if self.ab_tests:
                    rm.set("decoder_pool:ab_tests", self.ab_tests)
                    logger.info(f"Saved {len(self.ab_tests)} A/B tests to Redis")

            # Mirror to disk so file fallback is always up-to-date
            try:
                os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
                with open(self.pool_path, "w") as f:
                    json.dump({"nodes": self.nodes, "ab_tests": self.ab_tests}, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to mirror decoder pool to disk: {e}")
        except Exception as e:
            logger.error(f"Failed to load from Redis: {e}")
            # Fallback to disk
            self._load_from_disk()

    async def reload(self):
        """Asynchronously reloads the configuration and checks the health of the new pool."""
        async with self.lock:
            if self.use_redis:
                logger.info("Reloading decoder pool configuration from Redis...")
                await self._load_from_redis()
            else:
                logger.info("Reloading decoder pool configuration from disk...")
                self._load_from_disk()
                
        # After reloading, immediately check the health of the new/updated pool
        await self.check_health()
        logger.info("Decoder pool reloaded and health checked.")

    async def check_health(self):
        async with self.lock:
            # Health check logic now iterates over self.nodes
            tasks = [self._check_single_node(node) for node in self.nodes]
            await asyncio.gather(*tasks)
            
            active_count = sum(1 for n in self.nodes if n.get('healthy'))
            decoder_active.set(active_count)
            for node in self.nodes:
                decoder_load.labels(node_id=node['node_id']).set(node.get('load', 0))

    async def _check_single_node(self, node: Dict):
        try:
            # Perform health and adapter checks concurrently
            health_resp, adapters_resp = await asyncio.gather(
                self.client.get(f"{node['endpoint']}/health"),
                self.client.get(f"{node['endpoint']}/loaded_adapters"),
                return_exceptions=True # Don't let one failure stop the other
            )
            # Process health response
            if isinstance(health_resp, httpx.Response) and health_resp.status_code == 200:
                node['healthy'] = True
                # In a real scenario, parse metrics from health endpoint (load/uptime)
                node['load'] = random.randint(0, node.get('capacity', 100))
                node['uptime'] = node.get('uptime', 0) + 10
            else:
                node['healthy'] = False
                node['load'] = -1
                node['uptime'] = 0

            # Process loaded adapters response
            if isinstance(adapters_resp, httpx.Response) and adapters_resp.status_code == 200:
                node['hot_adapters'] = adapters_resp.json()
            else:
                node['hot_adapters'] = []

        except Exception as e:
            node['healthy'] = False
            node['load'] = -1
            node['uptime'] = 0
            node['hot_adapters'] = []
            logger.warning(f"Health check failed for node {node['node_id']}: {e}")

    def get_healthy_decoders(self) -> List[Dict]:
        return [n for n in self.nodes if n.get('healthy')]

    async def add_decoder(self, node: Dict):
        async with self.lock:
            self.nodes.append(node)
            await self._save()

    async def remove_decoder(self, node_id: str):
        async with self.lock:
            self.nodes = [n for n in self.nodes if n['node_id'] != node_id]
            await self._save()

    async def _save(self):
        """Save configuration to Redis and/or disk"""
        # First try to save to Redis if configured
        try:
            if self.use_redis:
                from utils.redis_manager import RedisManager
                rm = RedisManager.get_instance()
                if rm.get_client():
                    rm.set("decoder_pool:nodes", self.nodes)
                    rm.set("decoder_pool:ab_tests", self.ab_tests)
                    logger.info(f"Saved {len(self.nodes)} nodes and {len(self.ab_tests)} A/B tests to Redis")
        except Exception as e:
            logger.error(f"Failed to save to Redis: {e}")
            # Will fall back to disk
        
        # Always save to disk (as backup or primary if Redis not configured)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
            
            with open(self.pool_path, "w") as f:
                json.dump({
                "nodes": self.nodes,
                "ab_tests": self.ab_tests
            }, f, indent=2)

    def get_nodes_by_tags(self, tags: List[str], nodes_list: List[Dict]) -> List[Dict]:
        """Filters a list of nodes to find those matching ALL given tags."""
        return [
            node for node in nodes_list
            if all(tag in node.get("tags", []) for tag in tags)
        ]    

    async def get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """Get a node by ID"""
        async with self.lock:
            for node in self.nodes:
                if node['node_id'] == node_id:
                    return node
        return None
    
    async def update_node_status(self, node_id: str, healthy: bool, load: int):
        """Update a node's status"""
        async with self.lock:
            for i, node in enumerate(self.nodes):
                if node['node_id'] == node_id:
                    self.nodes[i]['healthy'] = healthy
                    self.nodes[i]['load'] = load
                    break        

    # --- THE CORE A/B LOGIC ---
    def pick_best_node(
        self, source_lang: str, target_lang: str, 
        adapter_name: str, 
        is_zero_shot: bool = False
    ) -> Optional[Dict]:
        """
        Intelligently picks the best node for a request, now with A/B testing logic.
        
        Priority:
        1. Healthy node with the adapter already loaded (least loaded among them).
        2. If none, any healthy node (least loaded among them).
        """
        healthy_nodes = self.get_healthy_decoders()
        if not healthy_nodes:
            return None

        # --- MODIFIED: Handle Zero-Shot routing first ---
        if is_zero_shot:
            # For zero-shot, find a node that has at least one of the pivot adapters hot.
            source_adapter, target_adapter = get_pivot_adapters(source_lang, target_lang, None)
            hot_candidates = [n for n in healthy_nodes if source_adapter in n.get('hot_adapters', []) or target_adapter in n.get('hot_adapters', [])]
            if hot_candidates:
                return min(hot_candidates, key=lambda n: n.get('load', float('inf')))
            # Fallback to any healthy node if none have the adapters hot.

        # 1. Check for an active A/B test for this language pair
        pair_str = f"{source_lang}-{target_lang}"
        active_test = next((
            test for test in self.ab_tests 
            if test.get("is_active") and test.get("language_pair") == pair_str
        ), None)

        if active_test:
            # A/B Test is active for this pair. Decide which group this request falls into.
            if random.randint(1, 100) <= active_test.get("traffic_percentage", 0):
                # --- EXPERIMENT GROUP ---
                logger.info(f"A/B Test '{active_test['name']}': Routing to EXPERIMENT group.")
                exp_tags = active_test.get("experiment_group_tags", [])
                candidate_nodes = self.get_nodes_by_tags(exp_tags, healthy_nodes)
                if candidate_nodes:
                    # Pick the least loaded from the experimental pool
                    return min(candidate_nodes, key=lambda n: n.get('load', float('inf')))
                else:
                    logger.warning(f"A/B Test '{active_test['name']}': No healthy experimental nodes found! Falling back to control.")
            
            # --- CONTROL GROUP (or fallback from experiment) ---
            control_tags = active_test.get("control_group_tags", [])
            candidate_nodes = self.get_nodes_by_tags(control_tags, healthy_nodes)
            if candidate_nodes:
                # Standard intelligent routing for the control group
                return self._route_intelligently(candidate_nodes, adapter_name)
            else:
                logger.error(f"A/B Test '{active_test['name']}': No healthy control nodes found! This is critical.")
                return None # Or fallback to any healthy node    

        # 2. No active A/B test, proceed with standard intelligent routing
        return self._route_intelligently(healthy_nodes, adapter_name)

    def _route_intelligently(self, nodes_list: List[Dict], adapter_name: str) -> Optional[Dict]:
        """The intelligent routing logic."""
        # Find nodes with the adapter already "hot" in their cache
        hot_nodes = [n for n in nodes_list if adapter_name in n.get('hot_adapters', [])]
        if hot_nodes:
            # If we have hot nodes, pick the least loaded among them
            logger.info(f"Found {len(hot_nodes)} hot nodes for adapter '{adapter_name}'. Routing to least loaded.")
            return min(hot_nodes, key=lambda n: n.get('load', float('inf')))
        # Fallback - no hot nodes found, so pick the least loaded overall.
        # This node will have to do a cold load of the adapter.
        logger.warning(f"No hot nodes found for adapter '{adapter_name}'. Falling back to least loaded node.")    
        return min(nodes_list, key=lambda n: n.get('load', float('inf'))) if nodes_list else None 
    
    # Helper in DecoderPool to check for direct support
    def is_direct_pair(self, source_lang: str, target_lang: str) -> bool:
        """
        Checks if a language pair is directly supported (i.e., not zero-shot).
        This is a placeholder. In production, this would check against a config
        of trained language pairs.
        """
        # For now, assume any pair involving English is directly supported.
        return source_lang == 'en' or target_lang == 'en'   

pool = DecoderPool()

# --- Authentication Dependencies ---
bearer_scheme = HTTPBearer()

def get_current_user(token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Dependency for JWT-based API authentication."""
    try:
        jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_session(request: Request) -> bool:
    """Dependency to check for a valid session cookie for the dashboard."""
    try:
        token = request.cookies.get("session_token")
        if not token:
            return False
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub") == "admin"
    except (jwt.PyJWTError, KeyError):
        return False

# --- Background Tasks ---
async def background_tasks():
    """Runs periodic health checks and handles config reloads and periodic mirroring."""
    last_mirror = 0.0
    while True:
        now = time.time()
        # Check if a reload has been signaled by the watchdog
        if CONFIG_NEEDS_RELOAD.is_set():
            await pool.reload()
            CONFIG_NEEDS_RELOAD.clear()  # Reset the event after handling

        # Periodically mirror Redis state to disk every MIRROR_INTERVAL_SECONDS
        if now - last_mirror >= MIRROR_INTERVAL_SECONDS:
            try:
                pool.mirror_redis_to_disk()
            except Exception as e:
                logger.debug(f"Periodic mirror failed: {e}")
            last_mirror = now

        logger.info("Running background health check...")
        await pool.check_health()
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    # Start the background task
    asyncio.create_task(background_tasks())

    # Start the watchdog observer to monitor the config file
    observer = Observer()
    pool_dir = os.path.dirname(POOL_PATH)
    if not os.path.exists(pool_dir):
        os.makedirs(pool_dir)
    observer.schedule(PoolReloadHandler(), pool_dir, recursive=False)
    observer.start()
    logger.info(f"Started watching {pool_dir} for configuration changes.")    

# --- API Endpoints ---
api_router = APIRouter(prefix="/api", tags=["API"])

@api_router.get("/status", response_model=StatusSchema)
async def get_status():
    """Get the current status of the decoder pool."""
    with tracer.start_as_current_span("get_status"):
        healthy_decoders = pool.get_healthy_decoders()
        return {
            "model_version": MODEL_VERSION,
            "decoder_pool_size": len(pool.nodes),
            "healthy_decoders": len(healthy_decoders),
            "decoders": [DecoderNodeSchema(**n) for n in pool.nodes]
        }

@api_router.post("/decode")
async def decode_proxy(
    request: Request,
    x_source_language: str = Header(...), 
    x_target_language: str = Header(...), 
    x_domain: Optional[str] = Header(None)
):
    """Proxy a decode request to the least loaded healthy decoder."""
    # Add API key validation
    api_key = request.headers.get('X-API-Key')
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    requests_total.labels(endpoint='/api/decode', group='standard').inc()
    with tracer.start_as_current_span("decode_proxy") as span:

        # Check if this is a direct, supported pair
        # This requires a new helper function in the pool
        is_direct_pair = pool.is_direct_pair(x_source_language, x_target_language)

        if not is_direct_pair:
            # --- ZERO-SHOT PIVOT LOGIC ---
            logger.info(f"Unsupported pair {x_source_language}->{x_target_language}. Attempting zero-shot pivot.")
            span.set_attribute("translation_type", "zero_shot_pivot")
            
            # Use the new zero-shot handler
            return await handle_zero_shot_request(request, x_source_language, x_target_language, x_domain)

        # --- STANDARD ROUTING LOGIC (for directly supported pairs) ---
        span.set_attribute("translation_type", "direct")
        adapter_name = f"{x_target_language}_{x_domain}" if x_domain else x_target_language

        # New A/B aware intelligent picker
        node = pool.pick_best_node(x_source_language, x_target_language, adapter_name)

        if not node:
            requests_errors.labels(endpoint='/api/decode').inc()
            raise HTTPException(status_code=503, detail="No healthy decoders available")
        
        # Tag the request with its group for metrics 
        request_group = "control"
        if "experimental" in node.get("tags", []):
            request_group = "experiment"
        span.set_attribute("ab_test_group", request_group)
        span.set_attribute("routed_node_id", node['node_id'])
        span.set_attribute("is_hot_route", adapter_name in node.get('hot_adapters', []))

        try:
            content = await request.body()
            # Pass domain header to the decoder node
            headers = {
                'Content-Type': request.headers['Content-Type'], 
                'X-Target-Language': x_target_language,
                'X-Domain': x_domain or ''
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{node['endpoint']}/decode", content=content, headers=headers, timeout=10)
                resp.raise_for_status()
            return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            requests_errors.labels(endpoint='/api/decode', group=request_group).inc()
            logger.error(f"Failed to contact decoder {node['node_id']}: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to contact decoder: {e}")

# New helper function for zero-shot requests 
async def handle_zero_shot_request(request: Request, source_lang: str, target_lang: str, domain: Optional[str]):
    """Orchestrates a zero-shot translation request."""
    
    # 1. Define the pivot adapters
    source_adapter_name, target_adapter_name = get_pivot_adapters(source_lang, target_lang, domain)
    
    # 2. Pick a healthy, low-load node to perform the composition
    node = pool.pick_best_node(source_lang, target_lang, "", is_zero_shot=True)
    if not node:
        raise HTTPException(status_code=503, detail="No healthy decoders available for zero-shot task.")

    # 3. Instruct the chosen node to create the composed adapter
    composition_payload = {
        "source_adapter": source_adapter_name,
        "target_adapter": target_adapter_name
    }
    try:
        async with httpx.AsyncClient() as client:
            # This call requires authentication, which we need to handle
            # For now, assuming an internal auth mechanism
            compose_resp = await client.post(
                f"{node['endpoint']}/compose_adapter", 
                json=composition_payload,
                timeout=15.0,
                headers={"X-Internal-Auth": INTERNAL_AUTH_TOKEN}
            )
            compose_resp.raise_for_status()
            composed_adapter_name = compose_resp.json()["composed_adapter_name"]
    except Exception as e:
        logger.error(f"Failed to instruct node {node['node_id']} to compose adapters: {e}")
        raise HTTPException(status_code=500, detail="Failed to create zero-shot adapter.")

    # 4. Now, proxy the original request to the SAME node, telling it to use the new virtual adapter
    try:
        content = await request.body()
        headers = {
            'Content-Type': request.headers['Content-Type'], 
            # IMPORTANT: We tell the decoder to use the composed adapter
            'X-Target-Language': composed_adapter_name,
            'X-Domain': domain or ''
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{node['endpoint']}/decode", content=content, headers=headers, timeout=10)
            resp.raise_for_status()
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
    except Exception as e:
        logger.error(f"Proxying failed for zero-shot request: {e}")
        raise HTTPException(status_code=502, detail="Zero-shot proxy failed.")

def get_pivot_adapters(source_lang, target_lang, domain):
    # This logic assumes you train adapters like 'en-es', 'en-de', etc.
    # and name them 'es', 'de' for simplicity.
    source_adapter = f"{source_lang}_{domain}" if domain else source_lang
    target_adapter = f"{target_lang}_{domain}" if domain else target_lang
    return source_adapter, target_adapter        

# --- Admin API Endpoints ---
admin_router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(get_current_user)])

@admin_router.post("/add_decoder")
async def add_decoder(node: DecoderNodeSchema):
    """Register a new decoder node."""
    await pool.add_decoder(node.dict())
    return {"status": "added", "node": node}

@admin_router.post("/remove_decoder")
async def remove_decoder(node_id: str = Form(...)):
    """De-register a decoder node."""
    await pool.remove_decoder(node_id)
    return {"status": "removed", "node_id": node_id}

# --- Dashboard and UI ---
DASHBOARD_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Coordinator Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; background: #f4f4f9; color: #333; }
        h1 { color: #4a4a4a; }
        table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background: #6a89cc; color: white; }
        tr:nth-child(even) { background: #f2f2f2; }
        .healthy { color: #2ecc71; font-weight: bold; }
        .unhealthy { color: #e74c3c; font-weight: bold; }
        .admin-form { background: white; padding: 1em; margin-bottom: 1em; border-radius: 5px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
        input, button { padding: 8px; margin-right: 5px; border: 1px solid #ccc; border-radius: 3px; }
        button { background: #6a89cc; color: white; cursor: pointer; border: none; }
        button:hover { background: #4a69bb; }
    </style>
</head>
<body>
    <h1>Coordinator Dashboard</h1>
    {% if not logged_in %}
    <form method="post" action="/login" class="admin-form">
        <input type="password" name="token" placeholder="Admin Token" required />
        <button type="submit">Login</button>
    </form>
    {% else %}
    <form method="post" action="/logout" class="admin-form">
        <button type="submit">Logout</button>
    </form>
    {% endif %}
    <table>
        <tr><th>Node ID</th><th>Endpoint</th><th>Region</th><th>GPU</th><th>Capacity</th><th>Load</th><th>Status</th></tr>
        {% for node in nodes %}
        <tr>
            <td>{{ node.node_id }}</td>
            <td><a href="{{ node.endpoint }}" target="_blank">{{ node.endpoint }}</a></td>
            <td>{{ node.region }}</td>
            <td>{{ node.gpu_type }}</td>
            <td>{{ node.capacity }}</td>
            <td>{{ node.load }}</td>
            <td class="{{ 'healthy' if node.healthy else 'unhealthy' }}">{{ 'Healthy' if node.healthy else 'Unhealthy' }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
""")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, logged_in: bool = Depends(get_session)):
    """Serves the main dashboard UI."""
    nodes_data = [DecoderNodeSchema(**n) for n in pool.pool]
    return DASHBOARD_TEMPLATE.render(nodes=nodes_data, logged_in=logged_in)

@app.post("/login")
async def login(token: str = Form(...)):
    """Handles admin login."""
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if token == AUTH_TOKEN:
        # Create a JWT session token
        session_token = jwt.encode(
            {"sub": "admin", "exp": datetime.utcnow() + timedelta(hours=1)},
            SECRET_KEY,
            algorithm="HS256"
        )
        response.set_cookie(key="session_token", value=session_token, httponly=True, samesite="strict")
    return response

@app.post("/logout")
async def logout():
    """Handles admin logout."""
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("session_token")
    return response

# --- Metrics Endpoint ---
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")

# --- Include Routers ---
app.include_router(api_router)
app.include_router(admin_router)

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, workers=API_WORKERS)