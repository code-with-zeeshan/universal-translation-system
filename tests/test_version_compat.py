import json
import unittest
from pathlib import Path

class TestVersionCompatibility(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('version-config.json').read_text(encoding='utf-8'))

    def test_core_fields_present(self):
        core = self.cfg.get('core', {})
        for field in ['version','minSupportedVersion','apiVersion']:
            self.assertIn(field, core)
            self.assertTrue(core[field])

    def test_component_compatibility_ranges(self):
        comps = self.cfg.get('components', {})
        required = ['encoder','decoder','coordinator']
        for k in required:
            self.assertIn(k, comps)
            self.assertIn('compatibleWith', comps[k])
            cw = comps[k]['compatibleWith']
            self.assertIsInstance(cw, dict)
            # minimal sanity: ranges include >= and < semantics
            for dep, rng in cw.items():
                self.assertRegex(rng, r">=\d+\.\d+\.\d+ <\d+\.\d+\.\d+")

    def test_versions_consistent_with_workflow(self):
        # Basic smoke: ensure major version alignment across core and key components
        core_ver = self.cfg['core']['version'].split('.')[0]
        comps = self.cfg.get('components', {})
        majors = {name: v.get('version','0.0.0').split('.')[0] for name, v in comps.items()}
        # Ensure core major matches key libs to avoid glaring mismatches
        for name in ['encoder','decoder','coordinator','python-package']:
            if name in majors:
                self.assertEqual(core_ver, majors[name])

if __name__ == '__main__':
    unittest.main()