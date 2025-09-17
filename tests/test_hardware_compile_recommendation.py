import unittest
from main import HardwareConfig

class TestHardwareCompileRecommendation(unittest.TestCase):
    def test_h100(self):
        enabled, mode = HardwareConfig.get_compile_recommendation(["NVIDIA H100 PCIe"])
        self.assertTrue(enabled)
        self.assertEqual(mode, "max-autotune")

    def test_a100(self):
        enabled, mode = HardwareConfig.get_compile_recommendation(["NVIDIA A100-SXM4-40GB"])
        self.assertTrue(enabled)
        self.assertEqual(mode, "max-autotune")

    def test_v100(self):
        enabled, mode = HardwareConfig.get_compile_recommendation(["Tesla V100-SXM2-16GB"])
        self.assertTrue(enabled)
        self.assertEqual(mode, "reduce-overhead")

    def test_t4(self):
        enabled, mode = HardwareConfig.get_compile_recommendation(["Tesla T4"])
        self.assertTrue(enabled)
        self.assertEqual(mode, "reduce-overhead")

    def test_consumer_3090(self):
        enabled, mode = HardwareConfig.get_compile_recommendation(["NVIDIA GeForce RTX 3090"])
        self.assertTrue(enabled)
        self.assertEqual(mode, "reduce-overhead")

    def test_unknown_or_cpu(self):
        enabled, mode = HardwareConfig.get_compile_recommendation(["Some Unknown GPU"])
        self.assertFalse(enabled)
        self.assertEqual(mode, "default")
        enabled2, mode2 = HardwareConfig.get_compile_recommendation([])
        self.assertFalse(enabled2)
        self.assertEqual(mode2, "default")

if __name__ == "__main__":
    unittest.main()