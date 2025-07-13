# data/smart_data_downloader.py
class SmartDataStrategy:
    """
    Instead of 190 pairs, we need:
    1. English-centric: 19 pairs (en-X)
    2. High-traffic direct: ~10-20 pairs (es-pt, zh-ja, etc.)
    Total: ~30-40 pairs (not 190!)
    """
    
    def get_required_pairs(self):
        languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
                     'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv']
        
        required_pairs = []
        
        # 1. English-centric pairs (MUST HAVE)
        for lang in languages[1:]:  # Skip 'en'
            required_pairs.append(('en', lang))
        
        # 2. High-traffic direct pairs (NICE TO HAVE)
        direct_pairs = [
            ('es', 'pt'),  # Spanish-Portuguese
            ('es', 'fr'),  # Spanish-French  
            ('de', 'fr'),  # German-French
            ('zh', 'ja'),  # Chinese-Japanese
            ('ar', 'fr'),  # Arabic-French (North Africa)
            ('hi', 'en'),  # Hindi-English (already covered)
            ('ru', 'uk'),  # Russian-Ukrainian
        ]
        
        for pair in direct_pairs:
            if pair not in required_pairs:
                required_pairs.append(pair)
        
        return required_pairs  # Only ~26 pairs!