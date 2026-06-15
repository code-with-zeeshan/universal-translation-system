# monitoring/metrics_collector.py
"""
Vocabulary-specific metrics collector.
Shared metrics are imported from monitoring.metrics to avoid conflicts.
"""
from prometheus_client import Gauge, REGISTRY
from monitoring.metrics import (
    TRANSLATION_REQUESTS, TRANSLATION_LATENCY,
    SYSTEM_GPU_UTILIZATION, COORDINATOR_ACTIVE_CONNECTIONS,
)

import logging
import sys
import os
from pathlib import Path

try:
    from runtime.vocabulary.manager import UnifiedVocabularyManager, VocabularyMode
    HAS_VOCAB_MANAGER = True
except ImportError:
    HAS_VOCAB_MANAGER = False
    logging.warning("VocabularyManager not available - vocabulary metrics disabled")

# Re-export aliases for backward compatibility
translation_requests = TRANSLATION_REQUESTS
translation_latency = TRANSLATION_LATENCY
gpu_utilization = SYSTEM_GPU_UTILIZATION
active_connections = COORDINATOR_ACTIVE_CONNECTIONS

# Vocabulary metrics (unique to this collector)
vocabulary_pack_info = Gauge(
    'vocabulary_pack_info',
    'Vocabulary pack version information',
    ['pack_name', 'version', 'status']
)

vocabulary_packs_total = Gauge(
    'vocabulary_packs_total',
    'Total number of vocabulary packs',
    ['status']
)

vocabulary_pack_size_mb = Gauge(
    'vocabulary_pack_size_mb',
    'Size of vocabulary pack in MB',
    ['pack_name', 'version']
)

vocabulary_pack_tokens = Gauge(
    'vocabulary_pack_tokens',
    'Number of tokens in vocabulary pack',
    ['pack_name', 'version', 'token_type']
)

# Logging setup (centralized)
from utils.logging_config import setup_logging
from utils.common_utils import RuntimeDirectoryManager
setup_logging(log_dir=str(RuntimeDirectoryManager().logs_dir), log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("monitoring")


class VocabularyMetricsCollector:
    """Collects metrics about vocabulary packs."""
    
    def __init__(self, vocab_dir: str = 'vocabs'):
        self.vocab_dir = Path(vocab_dir)
        self.vocab_manager = None
        
    def collect_vocabulary_metrics(self):
        """Collect and update vocabulary metrics."""
        if not HAS_VOCAB_MANAGER or not self.vocab_manager:
            logger.warning("Vocabulary metrics collection skipped - VocabularyManager not available")
            return
            
        try:
            # Get all available vocabulary packs
            available_packs = self._get_available_packs()
            loaded_packs = self._get_loaded_packs()
            
            # Update total counts
            vocabulary_packs_total.labels(status='available').set(len(available_packs))
            vocabulary_packs_total.labels(status='loaded').set(len(loaded_packs))
            
            # Update per-pack metrics
            for pack_name, pack_info in available_packs.items():
                version = pack_info.get('version', 'unknown')
                status = 'loaded' if pack_name in loaded_packs else 'available'
                
                # Set pack info (version and status)
                vocabulary_pack_info.labels(
                    pack_name=pack_name,
                    version=version,
                    status=status
                ).set(1)  # Gauge with value 1 indicates existence
                
                # Set pack size if available
                if 'size_mb' in pack_info:
                    vocabulary_pack_size_mb.labels(
                        pack_name=pack_name,
                        version=version
                    ).set(pack_info['size_mb'])
                
                # Set token counts if available
                if 'token_counts' in pack_info:
                    for token_type, count in pack_info['token_counts'].items():
                        vocabulary_pack_tokens.labels(
                            pack_name=pack_name,
                            version=version,
                            token_type=token_type
                        ).set(count)
                        
            logger.info(f"Collected vocabulary metrics: {len(available_packs)} available, {len(loaded_packs)} loaded")
            
        except Exception as e:
            logger.error(f"Error collecting vocabulary metrics: {e}")
            # Set error status
            vocabulary_pack_info.labels(
                pack_name='error',
                version='error',
                status='error'
            ).set(0)
    
    def _get_available_packs(self) -> dict:
        """Get information about all available vocabulary packs."""
        packs = {}
        
        try:
            # Scan for msgpack files
            for pack_file in self.vocab_dir.glob('*_v*.msgpack'):
                pack_name, version = self._parse_pack_filename(pack_file)
                if pack_name:
                    pack_info = {
                        'version': version,
                        'file_path': str(pack_file),
                        'size_mb': pack_file.stat().st_size / (1024 * 1024)
                    }
                    
                    # Try to load pack to get token counts
                    try:
                        pack_data = self._load_pack_info(pack_file)
                        if pack_data:
                            pack_info['token_counts'] = {
                                'regular': len(pack_data.get('tokens', {})),
                                'subword': len(pack_data.get('subwords', {})),
                                'special': len(pack_data.get('special_tokens', {}))
                            }
                    except Exception as e:
                        logger.debug(f"Could not load pack info for {pack_file}: {e}")
                    
                    packs[pack_name] = pack_info
                    
        except Exception as e:
            logger.error(f"Error scanning vocabulary packs: {e}")
            
        return packs
    
    def _get_loaded_packs(self) -> dict:
        """Get information about currently loaded vocabulary packs."""
        if not self.vocab_manager:
            return {}
            
        try:
            # Access loaded_packs from vocabulary manager
            loaded = {}
            for name, pack in self.vocab_manager.loaded_packs.items():
                loaded[name] = {
                    'version': getattr(pack, 'version', 'unknown'),
                    'size': getattr(pack, 'size', 0)
                }
            return loaded
        except Exception as e:
            logger.error(f"Error getting loaded packs: {e}")
            return {}
    
    def _parse_pack_filename(self, pack_file: Path) -> tuple:
        """Parse pack name and version from filename."""
        try:
            # Expected format: packname_v1.2.msgpack
            stem = pack_file.stem  # Remove .msgpack
            parts = stem.split('_v')
            if len(parts) >= 2:
                pack_name = '_'.join(parts[:-1])
                version = parts[-1]
                return pack_name, version
        except Exception:
            pass
        return None, None
    
    def _load_pack_info(self, pack_file: Path) -> dict:
        """Load basic info from pack file."""
        try:
            import msgpack
            with open(pack_file, 'rb') as f:
                # Read only enough to get basic info (e.g., first 1KB)
                # In a real scenario, you'd parse the msgpack stream for metadata
                # without loading the entire vocabulary.
                data = msgpack.unpackb(f.read(1024), raw=False, strict_map_key=False)
                return data
        except Exception:
            return None


# Create global vocabulary metrics collector
vocab_collector = VocabularyMetricsCollector() if HAS_VOCAB_MANAGER else None


def collect_vocabulary_metrics():
    """Global function to collect vocabulary metrics."""
    if vocab_collector:
        vocab_collector.collect_vocabulary_metrics()


from monitoring.metrics import track_translation_request  # noqa: F401


# Example usage in your decoder or encoder
def get_metrics_summary():
    """Get a summary of current metrics for logging or API response."""
    summary = {
        'translation_requests': {
            'total': translation_requests._value.sum() if hasattr(translation_requests, '_value') else 0
        },
        'active_connections': active_connections._value if hasattr(active_connections, '_value') else 0,
    }
    
    if vocab_collector:
        available_packs = vocab_collector._get_available_packs()
        loaded_packs = vocab_collector._get_loaded_packs()
        summary['vocabulary'] = {
            'available_packs': list(available_packs.keys()),
            'loaded_packs': list(loaded_packs.keys()),
            'total_available': len(available_packs),
            'total_loaded': len(loaded_packs)
        }
    
    return summary