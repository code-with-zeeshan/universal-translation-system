# coordinator/main.py
"""
Explicit uvicorn launcher for local runs and clarity. Dockerfile already uses uvicorn directly.
"""
import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("COORDINATOR_HOST", "0.0.0.0")
    port = int(os.environ.get("COORDINATOR_PORT", "5100"))
    uvicorn.run("coordinator.advanced_coordinator:app", host=host, port=port, reload=False)