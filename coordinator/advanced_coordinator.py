# coordinator/advanced_coordinator.py
import asyncio
import json
import logging
import os
import threading
import time
import random
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional

import httpx
import jwt
from fastapi import (FastAPI, Request, Depends, HTTPException, Form, Header, Response, status, APIRouter)
from fastapi.responses import HTMLResponse, RedirectResponse
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

# Import utility modules
from utils.auth import APIKeyManager
from utils.rate_limiter import RateLimiter
from utils.security import validate_model_source, safe_load_model

# --- Configuration and Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POOL_PATH = os.path.join("configs", "decoder_pool.json")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
SECRET_KEY = os.environ.get("COORDINATOR_SECRET", "a-very-secret-key-for-cookies")
JWT_SECRET = os.environ.get("COORDINATOR_JWT_SECRET", "a-super-secret-jwt-key")
AUTH_TOKEN = os.environ.get("COORDINATOR_TOKEN", "changeme123")
INTERNAL_AUTH_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", "internal-secret-token-for-service-auth")

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
    def __init__(self, pool_path=POOL_PATH):
        self.pool_path = pool_path
        self.lock = asyncio.Lock()
        self.nodes: List[Dict] = [] # Renamed from 'pool' for clarity
        self.ab_tests: List[Dict] = []
        self.client = httpx.AsyncClient(timeout=2.0)
        self.reload_sync()

    def _load_from_disk(self):
        """Synchronously loads the pool nodes and AB tests configuration from the JSON file."""
        try:
            if os.path.exists(self.pool_path):
                with open(self.pool_path, "r") as f:
                    config_data = json.load(f)
                    # Load both nodes and A/B tests
                    self.nodes = config_data.get("nodes", [])
                    self.ab_tests = config_data.get("ab_tests", [])
            else:
                self.nodes, self.ab_tests = [], []
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse decoder pool file {self.pool_path}: {e}")
            self.nodes, self.ab_tests = [], []

    async def reload(self):
        """Asynchronously reloads the configuration and checks the health of the new pool."""
        async with self.lock:
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
    """Runs periodic health checks and handles config reloads."""
    while True:
        # Check if a reload has been signaled by the watchdog
        if CONFIG_NEEDS_RELOAD.is_set():
            await pool.reload()
            CONFIG_NEEDS_RELOAD.clear()  # Reset the event after handling

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
    uvicorn.run(app, host="0.0.0.0", port=5100)