# run_system.py
#!/usr/bin/env python
"""
Main entry point for the Universal Translation System
Ensures proper setup and runs the system
"""
import argparse
import logging
import sys
from pathlib import Path
from utils.logging_config import setup_logging
setup_logging(log_dir="logs", log_level="INFO")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.final_integration import SystemIntegrator
from integration.connect_all_systems import integrate_full_pipeline



logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Universal Translation System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'translate', 'setup'], 
                       default='setup', help='Operating mode')
    parser.add_argument('--config', type=str, default='config/integration_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate system setup')
    
    args = parser.parse_args()
    
    # Validate system
    logger.info("🔍 Validating system setup...")
    if not SystemIntegrator.validate_system_ready():
        logger.error("System validation failed. Please fix the issues above.")
        return 1
    
    if args.validate_only:
        logger.info("✅ Validation complete")
        return 0
    
    # Run based on mode
    if args.mode == 'setup':
        logger.info("🚀 Initializing Universal Translation System...")
        system = integrate_full_pipeline(args.config)
        
        if system:
            logger.info("✅ System initialized successfully!")
            logger.info("Run with --mode train to start training")
        else:
            logger.error("❌ System initialization failed")
            return 1
            
    elif args.mode == 'train':
        logger.info("🎯 Starting training...")
        from training.train_universal_system import main as train_main
        train_main()
        
    elif args.mode == 'evaluate':
        logger.info("📊 Starting evaluation...")
        from evaluation.evaluate_model import main as evaluate_main
        evaluate_main(args.config) # Assuming evaluate_model.py takes a config
        
    elif args.mode == 'translate':
        logger.info("🌐 Starting translation service...")
        # Placeholder for translation service. This would likely involve:
        # 1. Loading a trained model (e.g., from `models/production`)
        # 2. Initializing an inference service (e.g., using FastAPI from `cloud_decoder`)
        # 3. Exposing an API endpoint for translation requests.
        logger.info("Translation service not yet implemented. Please refer to documentation for manual setup.")
        return 1 # Indicate that this mode is not fully implemented
    
    return 0

if __name__ == "__main__":
    sys.exit(main())