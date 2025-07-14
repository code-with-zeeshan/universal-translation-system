# data/smart_data_downloader.py
from typing import List
from dataclasses import dataclass
import logging

@dataclass
class LanguagePair:
    """Structured representation of language pairs"""
    source: str
    target: str
    priority: str
    expected_size: int

class SmartDataStrategy:
    """Optimize language pair selection strategy with modern practices"""
    
    def __init__(self):
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_dir /'data_pipeline.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_required_pairs(self) -> List[LanguagePair]:
        """Return strategically selected language pairs"""
        languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
                     'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv']
        
        required_pairs = []
        
        # English-centric pairs
        for lang in languages[1:]:
            required_pairs.append(LanguagePair(
                source='en',
                target=lang,
                priority='high' if lang in ['es', 'fr', 'de', 'zh', 'ru'] else 'medium' if lang in ['ja', 'ar', 'pt', 'it'] else 'low',
                expected_size=2000000 if lang in ['es', 'fr', 'de'] else 1500000 if lang in ['zh', 'ru'] else 1000000 if lang in ['ja', 'ar', 'pt', 'it'] else 500000
            ))
        
        # High-traffic direct pairs
        direct_pairs = [
            ('es', 'pt', 'medium', 200000),
            ('es', 'fr', 'medium', 200000),
            ('de', 'fr', 'medium', 200000),
            ('zh', 'ja', 'medium', 200000),
            ('ar', 'fr', 'medium', 200000),
            ('ru', 'uk', 'medium', 200000)
        ]
        
        for source, target, priority, size in direct_pairs:
            if (source, target) not in [(p.source, p.target) for p in required_pairs]:
                required_pairs.append(LanguagePair(source, target, priority, size))
        
        self.logger.info(f"✅ Selected {len(required_pairs)} language pairs")
        return required_pairs