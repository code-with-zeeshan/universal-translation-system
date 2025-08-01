# monitoring/health_service.py

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uvicorn
import logging

# Import the main system class to access the health monitor
# We use a TYPE_CHECKING block to avoid circular import issues at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from integration.connect_all_systems import UniversalTranslationSystem, SystemHealthMonitor

# --- Pydantic Models for API Response Schemas ---
# This gives you automatic data validation and API documentation (in /docs)

class ComponentHealth(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None

class HealthStatusResponse(BaseModel):
    status: str
    timestamp: float
    components: Dict[str, Any]
    resources: Dict[str, Any]

# --- FastAPI Application ---

app = FastAPI(
    title="Universal Translation System Health API",
    description="Provides health and status monitoring for the translation system.",
    version="1.0.0"
)

# This will hold the reference to the main system's health monitor
system_health_monitor: Optional["SystemHealthMonitor"] = None

# --- API Endpoints ---

@app.get("/health", response_model=HealthStatusResponse, tags=["Health"])
async def get_system_health():
    """
    Provides a comprehensive health check of the entire system,
    including all components and resource usage.
    """
    if not system_health_monitor:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    # Use the existing SystemHealthMonitor to get the status
    health_status = await system_health_monitor.check_health()
    return health_status

@app.get("/health/liveness", tags=["Health"])
async def get_liveness():
    """
    A simple liveness probe to check if the service is running.
    Kubernetes uses this to know whether to restart the container.
    """
    return {"status": "alive"}

@app.get("/health/readiness", tags=["Health"])
async def get_readiness():
    """
    A readiness probe to check if the service is ready to accept traffic.
    Kubernetes uses this to know whether to send traffic to the container.
    """
    if not system_health_monitor:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    health_status = await system_health_monitor.check_health()
    if health_status.get("status") != "healthy":
        raise HTTPException(status_code=503, detail=f"System is not ready: {health_status}")
        
    return {"status": "ready"}

# --- Service Runner ---

def start_health_service(system: "UniversalTranslationSystem", host: str = "0.0.0.0", port: int = 8081):
    """
    Starts the FastAPI health check service.
    This function is designed to be run in a separate thread.
    """
    global system_health_monitor
    # Store the reference to the system's health monitor
    system_health_monitor = system.health_monitor
    
    logging.info(f"Starting FastAPI health service on http://{host}:{port}")
    
    # Use uvicorn to run the FastAPI app
    uvicorn.run(app, host=host, port=port, log_level="warning")