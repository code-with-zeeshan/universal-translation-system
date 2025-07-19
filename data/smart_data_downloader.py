# data/smart_data_downloader.py
"""
Smart data strategy module - Refactored for better integration
Defines language pair priorities and download strategies
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import shared utilities with fallback
try:
    from data_utils import ConfigManager
    from utils.common_utils import StandardLogger
    INTEGRATED_MODE = True
except ImportError:
    import logging
    INTEGRATED_MODE = False
    
    class StandardLogger:
        @staticmethod
        def get_logger(name):
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(name)


@dataclass
class LanguagePair:
    """Structured representation of language pairs with metadata"""
    source: str
    target: str
    priority: str  # 'high', 'medium', 'low'
    expected_size: int
    data_sources: List[str] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = ['opus-100', 'nllb-seed', 'ccmatrix']
    
    @property
    def pair_string(self) -> str:
        """Get string representation of pair"""
        return f"{self.source}-{self.target}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'pair': self.pair_string,
            'priority': self.priority,
            'expected_size': self.expected_size,
            'sources': self.data_sources
        }


class SmartDataStrategy:
    """Optimize language pair selection strategy with configuration integration"""
    
    def __init__(self, config_path: str = 'data/config.yaml'):
        self.logger = StandardLogger.get_logger(__name__)
        
        # Load configuration if available
        if INTEGRATED_MODE:
            self.config = ConfigManager.load_config(config_path)
            self.languages = ConfigManager.get_languages()
            self.training_distribution = ConfigManager.get_training_distribution()
        else:
            # Fallback configuration
            self.languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
                             'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv']
            self.training_distribution = self._get_default_distribution()
        
        # Define priority rules
        self.priority_rules = {
            'high': ['en-es', 'en-fr', 'en-de', 'en-zh', 'en-ru'],
            'medium': ['en-ja', 'en-ar', 'en-pt', 'en-it', 'en-hi', 'en-ko',
                      'es-pt', 'es-fr', 'de-fr', 'zh-ja', 'ar-fr', 'ru-uk'],
            'low': ['en-tr', 'en-th', 'en-vi', 'en-pl', 'en-uk', 'en-nl', 'en-id', 'en-sv']
        }
        
        # Define data source preferences by pair type
        self.source_preferences = {
            'en-*': ['opus-100', 'nllb-seed', 'ccmatrix', 'wmt'],
            'european': ['opus-100', 'wmt', 'opus_books'],
            'asian': ['opus-100', 'nllb-seed', 'ted_talks'],
            'default': ['opus-100', 'nllb-seed', 'tatoeba']
        }
    
    def get_required_pairs(self) -> List[LanguagePair]:
        """
        Return strategically selected language pairs based on configuration
        
        Returns:
            List of LanguagePair objects sorted by priority
        """
        required_pairs = []
        
        # Process configured pairs from training distribution
        for pair_str, expected_size in self.training_distribution.items():
            if '-' in pair_str:
                source, target = pair_str.split('-')
                priority = self._determine_priority(pair_str)
                data_sources = self._get_data_sources(source, target)
                
                required_pairs.append(LanguagePair(
                    source=source,
                    target=target,
                    priority=priority,
                    expected_size=expected_size,
                    data_sources=data_sources
                ))
        
        # Sort by priority and size
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        required_pairs.sort(
            key=lambda p: (priority_order[p.priority], -p.expected_size)
        )
        
        self.logger.info(f"✅ Strategy defined for {len(required_pairs)} language pairs")
        self._log_strategy_summary(required_pairs)
        
        return required_pairs
    
    def _determine_priority(self, pair_str: str) -> str:
        """Determine priority of a language pair"""
        for priority, pairs in self.priority_rules.items():
            if pair_str in pairs:
                return priority
        
        # Default priority based on languages involved
        source, target = pair_str.split('-')
        if source == 'en' and target in ['es', 'fr', 'de', 'zh', 'ru']:
            return 'high'
        elif source == 'en' or target == 'en':
            return 'medium'
        else:
            return 'low'
    
    def _get_data_sources(self, source: str, target: str) -> List[str]:
        """Get recommended data sources for a language pair"""
        pair_str = f"{source}-{target}"
        
        # English-centric pairs
        if source == 'en' or target == 'en':
            return self.source_preferences['en-*']
        
        # European language pairs
        european_langs = ['es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl']
        if source in european_langs and target in european_langs:
            return self.source_preferences['european']
        
        # Asian language pairs
        asian_langs = ['zh', 'ja', 'ko', 'th', 'vi', 'id']
        if source in asian_langs and target in asian_langs:
            return self.source_preferences['asian']
        
        # Default sources
        return self.source_preferences['default']
    
    def _log_strategy_summary(self, pairs: List[LanguagePair]) -> None:
        """Log summary of the strategy"""
        high_priority = [p for p in pairs if p.priority == 'high']
        medium_priority = [p for p in pairs if p.priority == 'medium']
        low_priority = [p for p in pairs if p.priority == 'low']
        
        total_expected = sum(p.expected_size for p in pairs)
        
        self.logger.info("📊 STRATEGY SUMMARY:")
        self.logger.info(f"  High priority: {len(high_priority)} pairs")
        self.logger.info(f"  Medium priority: {len(medium_priority)} pairs")
        self.logger.info(f"  Low priority: {len(low_priority)} pairs")
        self.logger.info(f"  Total expected sentences: {total_expected:,}")
    
    def _get_default_distribution(self) -> Dict[str, int]:
        """Get default distribution for standalone mode"""
        return {
            'en-es': 2000000, 'en-fr': 2000000, 'en-de': 2000000,
            'en-zh': 1500000, 'en-ru': 1500000,
            'en-ja': 1000000, 'en-ar': 1000000, 'en-pt': 1000000, 'en-it': 1000000,
            'en-hi': 500000, 'en-ko': 500000, 'en-tr': 500000,
            'en-th': 300000, 'en-vi': 300000, 'en-pl': 300000,
            'en-uk': 300000, 'en-nl': 300000, 'en-id': 300000, 'en-sv': 300000,
            'es-pt': 200000, 'zh-ja': 200000, 'fr-es': 200000,
            'de-fr': 200000, 'ru-uk': 200000
        }
    
    def get_download_schedule(self) -> List[Dict[str, any]]:
        """
        Get optimized download schedule with parallel download groups
        
        Returns:
            List of download batches that can be processed in parallel
        """
        pairs = self.get_required_pairs()
        
        # Group pairs that can be downloaded in parallel
        schedule = []
        
        # Batch 1: High priority English pairs
        batch1 = [p for p in pairs if p.priority == 'high' and p.source == 'en']
        if batch1:
            schedule.append({
                'batch_name': 'High Priority English Pairs',
                'pairs': batch1,
                'parallel': True,
                'max_workers': 4
            })
        
        # Batch 2: Direct pairs (non-English)
        batch2 = [p for p in pairs if p.priority in ['high', 'medium'] and 
                  p.source != 'en' and p.target != 'en']
        if batch2:
            schedule.append({
                'batch_name': 'Direct Language Pairs',
                'pairs': batch2,
                'parallel': True,
                'max_workers': 2
            })
        
        # Batch 3: Medium priority pairs
        batch3 = [p for p in pairs if p.priority == 'medium' and 
                  (p.source == 'en' or p.target == 'en')]
        if batch3:
            schedule.append({
                'batch_name': 'Medium Priority Pairs',
                'pairs': batch3,
                'parallel': True,
                'max_workers': 3
            })
        
        # Batch 4: Low priority pairs
        batch4 = [p for p in pairs if p.priority == 'low']
        if batch4:
            schedule.append({
                'batch_name': 'Low Priority Pairs',
                'pairs': batch4,
                'parallel': True,
                'max_workers': 2
            })
        
        return schedule
    
    def estimate_download_size(self) -> Dict[str, float]:
        """
        Estimate total download size
        
        Returns:
            Dictionary with size estimates in GB
        """
        pairs = self.get_required_pairs()
        
        # Rough estimates based on experience
        bytes_per_sentence = {
            'high': 150,  # Longer, higher quality sentences
            'medium': 120,
            'low': 100
        }
        
        estimates = {
            'high_priority_gb': 0,
            'medium_priority_gb': 0,
            'low_priority_gb': 0,
            'total_gb': 0
        }
        
        for pair in pairs:
            size_gb = (pair.expected_size * bytes_per_sentence[pair.priority]) / (1024**3)
            estimates[f'{pair.priority}_priority_gb'] += size_gb
            estimates['total_gb'] += size_gb
        
        return estimates
    
    def export_strategy(self, output_file: str = 'data/download_strategy.json') -> None:
        """Export strategy to JSON file for documentation"""
        import json
        
        pairs = self.get_required_pairs()
        schedule = self.get_download_schedule()
        estimates = self.estimate_download_size()
        
        strategy_data = {
            'total_pairs': len(pairs),
            'language_pairs': [p.to_dict() for p in pairs],
            'download_schedule': [
                {
                    'batch_name': batch['batch_name'],
                    'pair_count': len(batch['pairs']),
                    'pairs': [p.pair_string for p in batch['pairs']],
                    'parallel': batch['parallel'],
                    'max_workers': batch['max_workers']
                }
                for batch in schedule
            ],
            'size_estimates': estimates
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(strategy_data, f, indent=2)
        
        self.logger.info(f"📄 Strategy exported to {output_file}")


def main():
    """Main entry point for standalone execution"""
    strategy = SmartDataStrategy()
    
    # Get and display strategy
    pairs = strategy.get_required_pairs()
    
    print("\n📊 LANGUAGE PAIR STRATEGY:")
    print("-" * 60)
    
    for priority in ['high', 'medium', 'low']:
        priority_pairs = [p for p in pairs if p.priority == priority]
        if priority_pairs:
            print(f"\n{priority.upper()} PRIORITY ({len(priority_pairs)} pairs):")
            for pair in priority_pairs[:5]:  # Show first 5
                print(f"  {pair.pair_string}: {pair.expected_size:,} sentences")
            if len(priority_pairs) > 5:
                print(f"  ... and {len(priority_pairs) - 5} more")
    
    # Get download schedule
    schedule = strategy.get_download_schedule()
    print("\n📅 DOWNLOAD SCHEDULE:")
    print("-" * 60)
    for i, batch in enumerate(schedule, 1):
        print(f"\nBatch {i}: {batch['batch_name']}")
        print(f"  Pairs: {len(batch['pairs'])}")
        print(f"  Parallel: {batch['parallel']} (max {batch['max_workers']} workers)")
    
    # Get size estimates
    estimates = strategy.estimate_download_size()
    print("\n💾 SIZE ESTIMATES:")
    print("-" * 60)
    for key, value in estimates.items():
        print(f"  {key}: {value:.2f} GB")
    
    # Export strategy
    strategy.export_strategy()


if __name__ == "__main__":
    main()