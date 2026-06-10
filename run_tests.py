#!/usr/bin/env python3
"""
Test runner that replicates conftest.py behavior for non-pytest environments.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("UTS_HMAC_KEY", "a" * 40)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Execute conftest.py in the proper context
conftest_path = Path(__file__).parent / 'tests' / 'conftest.py'
with open(conftest_path) as f:
    code = compile(f.read(), str(conftest_path), 'exec')
    exec(code, {'__file__': str(conftest_path), '__name__': 'conftest'})

# Now import and run all test modules
from unittest import TestLoader, TextTestRunner, TestSuite

import tests.test_encoder_quantizer
import tests.test_quality_comparator
import tests.test_hardware_profile
import tests.test_training_analytics
import tests.test_training_utils
import tests.test_system_health
import tests.test_translation_api
import tests.test_pipeline_data_flow

loader = TestLoader()
suite = TestSuite()
for mod in [
    tests.test_encoder_quantizer,
    tests.test_quality_comparator,
    tests.test_hardware_profile,
    tests.test_training_analytics,
    tests.test_training_utils,
    tests.test_system_health,
    tests.test_translation_api,
    tests.test_pipeline_data_flow,
]:
    suite.addTests(loader.loadTestsFromModule(mod))

runner = TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
