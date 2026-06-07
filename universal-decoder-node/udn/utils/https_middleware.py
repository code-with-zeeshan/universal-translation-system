# universal-decoder-node/universal_decoder_node/utils/https_middleware.py
import logging
from typing import Callable, Dict, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os

logger = logging.getLogger(__name__)

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce HTTPS by redirecting HTTP requests to HTTPS.
    
    This middleware checks if a request is using HTTP and redirects it to HTTPS
    if the environment is configured to enforce HTTPS.
    """
    
    def __init__(
        self, 
        app: FastAPI, 
        enforce_https: bool = None,
        https_port: int = 443,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize the HTTPS redirect middleware.
        
        Args:
            app: FastAPI application
            enforce_https: Whether to enforce HTTPS. If None, reads from environment.
            https_port: HTTPS port to redirect to
            exclude_paths: List of paths to exclude from HTTPS enforcement
        """
        super().__init__(app)
        # Read from environment if not explicitly provided
        if enforce_https is None:
            enforce_https = os.environ.get("ENFORCE_HTTPS", "false").lower() == "true"
        
        self.enforce_https = enforce_https
        self.https_port = https_port
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        
        if self.enforce_https:
            logger.info("HTTPS enforcement is enabled. HTTP requests will be redirected to HTTPS.")
        else:
            logger.info("HTTPS enforcement is disabled. HTTP requests will be allowed.")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and enforce HTTPS if enabled.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Skip HTTPS enforcement for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Skip HTTPS enforcement if disabled or already using HTTPS
        if not self.enforce_https or request.url.scheme == "https":
            return await call_next(request)
        
        # Redirect to HTTPS
        https_url = request.url.replace(scheme="https")
        
        # Use custom port if not the default 443
        if self.https_port != 443:
            https_url = https_url.replace(port=self.https_port)
        
        logger.info(f"Redirecting HTTP request to HTTPS: {https_url}")
        return RedirectResponse(url=str(https_url), status_code=301)

def add_https_middleware(
    app: FastAPI, 
    enforce_https: bool = None,
    https_port: int = 443,
    exclude_paths: Optional[list] = None
) -> None:
    """
    Add HTTPS enforcement middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        enforce_https: Whether to enforce HTTPS. If None, reads from environment.
        https_port: HTTPS port to redirect to
        exclude_paths: List of paths to exclude from HTTPS enforcement
    """
    app.add_middleware(
        HTTPSRedirectMiddleware,
        enforce_https=enforce_https,
        https_port=https_port,
        exclude_paths=exclude_paths
    )
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response