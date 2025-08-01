# coordinator/advanced_coordinator.py
import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional

import httpx
import jwt
from fastapi import (FastAPI, Request, Depends, HTTPException, Form,
                     Header, Response, status, APIRouter)
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

# --- Configuration and Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POOL_PATH = os.path.join("configs", "decoder_pool.json")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
SECRET_KEY = os.environ.get("COORDINATOR_SECRET", "a-very-secret-key-for-cookies")
JWT_SECRET = os.environ.get("COORDINATOR_JWT_SECRET", "a-super-secret-jwt-key")
AUTH_TOKEN = os.environ.get("COORDINATOR_TOKEN", "changeme123")

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
requests_total = Counter('coordinator_requests_total', 'Total requests received', ['endpoint'])
requests_errors = Counter('coordinator_requests_errors', 'Total errors', ['endpoint'])
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
        self.pool: List[Dict] = []
        self.client = httpx.AsyncClient(timeout=2.0)
        self.reload_sync()

    def _load_from_disk(self):
        """Synchronously loads the pool configuration from the JSON file."""
        try:
            if os.path.exists(self.pool_path):
                with open(self.pool_path, "r") as f:
                    self.pool = json.load(f)
            else:
                self.pool = []
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse decoder pool file {self.pool_path}: {e}")
            self.pool = []

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
            tasks = [self._check_single_node(node) for node in self.pool]
            await asyncio.gather(*tasks)
            
            active_count = sum(1 for n in self.pool if n.get('healthy'))
            decoder_active.set(active_count)
            for node in self.pool:
                decoder_load.labels(node_id=node['node_id']).set(node.get('load', 0))

    async def _check_single_node(self, node: Dict):
        try:
            resp = await self.client.get(f"{node['endpoint']}/health")
            node['healthy'] = resp.status_code == 200
            if node['healthy']:
                # In a real scenario, you'd parse metrics for load/uptime
                node['load'] = random.randint(0, node.get('capacity', 100))
                node['uptime'] = node.get('uptime', 0) + 10
        except (httpx.RequestError, httpx.HTTPStatusError):
            node['healthy'] = False
            node['load'] = -1
            node['uptime'] = 0

    def get_healthy_decoders(self) -> List[Dict]:
        return [n for n in self.pool if n.get('healthy')]

    async def add_decoder(self, node: Dict):
        async with self.lock:
            self.pool.append(node)
            await self._save()

    async def remove_decoder(self, node_id: str):
        async with self.lock:
            self.pool = [n for n in self.pool if n['node_id'] != node_id]
            await self._save()

    async def _save(self):
        with open(self.pool_path, "w") as f:
            json.dump(self.pool, f, indent=2)

    def pick_least_loaded(self) -> Optional[Dict]:
        healthy = self.get_healthy_decoders()
        if not healthy:
            return None
        return min(healthy, key=lambda n: n.get('load', float('inf')))

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
            "decoder_pool_size": len(pool.pool),
            "healthy_decoders": len(healthy_decoders),
            "decoders": pool.pool
        }

@api_router.post("/decode")
async def decode_proxy(request: Request, x_target_language: str = Header(...)):
    """Proxy a decode request to the least loaded healthy decoder."""
    requests_total.labels(endpoint='/api/decode').inc()
    with tracer.start_as_current_span("decode_proxy") as span:
        node = pool.pick_least_loaded()
        if not node:
            requests_errors.labels(endpoint='/api/decode').inc()
            raise HTTPException(status_code=503, detail="No healthy decoders available")
        
        span.set_attribute("routed_node_id", node['node_id'])
        try:
            content = await request.body()
            headers = {'Content-Type': request.headers['Content-Type'], 'X-Target-Language': x_target_language}
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{node['endpoint']}/decode", content=content, headers=headers, timeout=10)
                resp.raise_for_status()
            return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            requests_errors.labels(endpoint='/api/decode').inc()
            logger.error(f"Failed to contact decoder {node['node_id']}: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to contact decoder: {e}")

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