# monitoring/vocabulary_monitor.py
"""
Standalone vocabulary monitoring service.
Run this to expose vocabulary metrics on a separate port.
"""

import time
import logging
from prometheus_client import start_http_server
from metrics_collector import vocab_collector, collect_vocabulary_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run vocabulary monitoring service."""
    # Start Prometheus metrics server
    port = 9001
    start_http_server(port)
    logger.info(f"Vocabulary metrics server started on port {port}")
    
    # Collect metrics every 30 seconds
    while True:
        try:
            logger.info("Collecting vocabulary metrics...")
            collect_vocabulary_metrics()
        except Exception as e:
            logger.error(f"Error in vocabulary monitoring: {e}")
        
        time.sleep(30)


if __name__ == '__main__':
    main()