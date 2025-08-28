# utils/service_discovery.py
"""
Service discovery client for the Universal Translation System.
This module provides a client for registering and discovering decoder nodes.

Environment overrides:
- COORDINATOR_URL or SERVICE_DISCOVERY_COORDINATOR_URL: coordinator base URL (for from_env)
- SERVICE_DISCOVERY_HEARTBEAT_INTERVAL: heartbeat seconds (default 30)
- SERVICE_DISCOVERY_TIMEOUT: HTTP client timeout seconds (default 5)
"""

import httpx
import json
import os
import logging
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from .exceptions import NetworkError, ConfigurationError

logger = logging.getLogger(__name__)

class ServiceDiscoveryClient:
    """Client for service discovery."""
    
    def __init__(
        self,
        coordinator_url: str,
        node_info: Dict[str, Any],
        auto_register: bool = True,
        heartbeat_interval: int = 30,
        timeout: int = 5
    ):
        """
        Initialize service discovery client.
        
        Args:
            coordinator_url: URL of the coordinator service
            node_info: Information about this node
            auto_register: Whether to automatically register with the coordinator
            heartbeat_interval: Interval between heartbeats in seconds
            timeout: Timeout for HTTP requests in seconds
        """
        self.coordinator_url = coordinator_url.rstrip('/')
        self.node_info = node_info
        self.auto_register = auto_register
        # Allow env overrides for timing
        self.heartbeat_interval = int(os.environ.get("SERVICE_DISCOVERY_HEARTBEAT_INTERVAL", heartbeat_interval))
        self.timeout = int(os.environ.get("SERVICE_DISCOVERY_TIMEOUT", timeout))
        
        self.node_id = node_info.get('node_id')
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        self.heartbeat_task = None
        self.running = False
        
        # Status callback
        self.status_callback = None
        
        if auto_register:
            # Start in a background thread
            threading.Thread(target=self._start_registration, daemon=True).start()

    @classmethod
    def from_env(
        cls,
        node_info: Dict[str, Any],
        auto_register: bool = True
    ) -> "ServiceDiscoveryClient":
        """Construct a client using environment variables.
        Requires COORDINATOR_URL or SERVICE_DISCOVERY_COORDINATOR_URL.
        Optionally uses SERVICE_DISCOVERY_HEARTBEAT_INTERVAL and SERVICE_DISCOVERY_TIMEOUT.
        """
        coordinator_url = os.environ.get("COORDINATOR_URL") or os.environ.get("SERVICE_DISCOVERY_COORDINATOR_URL")
        if not coordinator_url:
            raise ConfigurationError("COORDINATOR_URL must be set for service discovery")
        heartbeat = int(os.environ.get("SERVICE_DISCOVERY_HEARTBEAT_INTERVAL", "30"))
        timeout = int(os.environ.get("SERVICE_DISCOVERY_TIMEOUT", "5"))
        return cls(coordinator_url, node_info, auto_register=auto_register, heartbeat_interval=heartbeat, timeout=timeout)
    
    def _start_registration(self):
        """Start registration in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.register())
        loop.close()
    
    async def register(self) -> str:
        """
        Register with the coordinator.
        
        Returns:
            Node ID
        """
        try:
            url = f"{self.coordinator_url}/api/v1/register"
            response = await self.client.post(url, json=self.node_info)
            
            if response.status_code != 200:
                raise NetworkError(f"Failed to register: {response.text}")
                
            data = response.json()
            self.node_id = data.get('node_id')
            self.node_info['node_id'] = self.node_id
            
            logger.info(f"Registered with coordinator: {self.node_id}")
            
            # Start heartbeat
            self.start_heartbeat()
            
            return self.node_id
        except Exception as e:
            logger.error(f"Error registering with coordinator: {e}")
            raise
    
    async def unregister(self) -> bool:
        """
        Unregister from the coordinator.
        
        Returns:
            True if successful
        """
        if not self.node_id:
            logger.warning("Not registered, cannot unregister")
            return False
            
        try:
            url = f"{self.coordinator_url}/api/v1/unregister/{self.node_id}"
            response = await self.client.delete(url)
            
            if response.status_code != 200:
                raise NetworkError(f"Failed to unregister: {response.text}")
                
            logger.info(f"Unregistered from coordinator: {self.node_id}")
            
            # Stop heartbeat
            self.stop_heartbeat()
            
            return True
        except Exception as e:
            logger.error(f"Error unregistering from coordinator: {e}")
            return False
    
    def start_heartbeat(self):
        """Start sending heartbeats."""
        if self.heartbeat_task:
            return
            
        self.running = True
        
        # Start in a background thread
        threading.Thread(target=self._run_heartbeat_loop, daemon=True).start()
        
        logger.info(f"Started heartbeat (interval: {self.heartbeat_interval}s)")
    
    def stop_heartbeat(self):
        """Stop sending heartbeats."""
        self.running = False
        logger.info("Stopped heartbeat")
    
    def _run_heartbeat_loop(self):
        """Run heartbeat loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # Send heartbeat
                loop.run_until_complete(self._send_heartbeat())
                
                # Sleep
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)  # Backoff on error
        
        loop.close()
    
    async def _send_heartbeat(self):
        """Send a heartbeat to the coordinator."""
        if not self.node_id:
            logger.warning("Not registered, cannot send heartbeat")
            return
            
        try:
            # Get current status
            status = await self._get_status()
            
            url = f"{self.coordinator_url}/api/v1/node/{self.node_id}/status"
            response = await self.client.put(
                url,
                data={
                    'healthy': status['healthy'],
                    'load': status['load']
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send heartbeat: {response.text}")
            else:
                logger.debug(f"Sent heartbeat: {status}")
        except Exception as e:
            logger.warning(f"Error sending heartbeat: {e}")
    
    async def _get_status(self) -> Dict[str, Any]:
        """
        Get current status for heartbeat.
        
        Returns:
            Status dictionary with 'healthy' and 'load' keys
        """
        # Default status
        status = {
            'healthy': True,
            'load': 0
        }
        
        # Call status callback if set
        if self.status_callback:
            try:
                custom_status = await self.status_callback()
                if isinstance(custom_status, dict):
                    status.update(custom_status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
                status['healthy'] = False
        
        return status
    
    def set_status_callback(self, callback: Callable[[], Dict[str, Any]]):
        """
        Set a callback for getting node status.
        
        Args:
            callback: Async function that returns a status dictionary
        """
        self.status_callback = callback
    
    async def discover_nodes(self) -> List[Dict[str, Any]]:
        """
        Discover all registered nodes.
        
        Returns:
            List of node dictionaries
        """
        try:
            url = f"{self.coordinator_url}/api/v1/status"
            response = await self.client.get(url)
            
            if response.status_code != 200:
                raise NetworkError(f"Failed to discover nodes: {response.text}")
                
            data = response.json()
            return data.get('decoders', [])
        except Exception as e:
            logger.error(f"Error discovering nodes: {e}")
            raise
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node dictionary or None if not found
        """
        try:
            url = f"{self.coordinator_url}/api/v1/node/{node_id}"
            response = await self.client.get(url)
            
            if response.status_code == 404:
                return None
                
            if response.status_code != 200:
                raise NetworkError(f"Failed to get node: {response.text}")
                
            return response.json()
        except Exception as e:
            logger.error(f"Error getting node: {e}")
            raise
    
    async def close(self):
        """Close the client and unregister."""
        self.stop_heartbeat()
        
        if self.node_id:
            await self.unregister()
            
        await self.client.aclose()
