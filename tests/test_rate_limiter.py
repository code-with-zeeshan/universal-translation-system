"""Tests for utils.rate_limiter"""
import time
import unittest
from unittest.mock import patch
from utils.rate_limiter import RateLimiter


class TestRateLimiter(unittest.TestCase):
    def test_init_defaults(self):
        rl = RateLimiter()
        self.assertEqual(rl.requests_per_minute, 60)
        self.assertEqual(rl.requests_per_hour, 1000)

    def test_is_allowed_returns_true_for_first_request(self):
        rl = RateLimiter(requests_per_minute=10)
        allowed, msg = rl.is_allowed("client-1")
        self.assertTrue(allowed)
        self.assertEqual(msg, "")

    def test_is_allowed_respects_per_minute_limit(self):
        rl = RateLimiter(requests_per_minute=3)
        for _ in range(3):
            allowed, msg = rl.is_allowed("client-2")
            self.assertTrue(allowed, f"Expected allowed, got {msg}")
        allowed, msg = rl.is_allowed("client-2")
        self.assertFalse(allowed)
        self.assertIn("Rate limit exceeded", msg)

    def test_is_allowed_independent_per_client(self):
        rl = RateLimiter(requests_per_minute=2)
        rl.is_allowed("client-a")
        rl.is_allowed("client-a")
        allowed, _ = rl.is_allowed("client-b")
        self.assertTrue(allowed)

    def test_window_slides_after_time(self):
        rl = RateLimiter(requests_per_minute=2)
        now = 1000.0
        with patch("utils.rate_limiter.time.time") as mock_time:
            mock_time.return_value = now
            rl.is_allowed("client-3")
            rl.is_allowed("client-3")
            allowed, _ = rl.is_allowed("client-3")
            self.assertFalse(allowed)
            mock_time.return_value = now + 61.0
            allowed, _ = rl.is_allowed("client-3")
            self.assertTrue(allowed)

    def test_slowapi_limiter_creation(self):
        with patch("utils.rate_limiter._slowapi_available", True):
            with patch("utils.rate_limiter.SlowapiLimiter") as MockLimiter:
                rl = RateLimiter(requests_per_minute=30, requests_per_hour=500)
                self.assertIsNotNone(rl._limiter)
                MockLimiter.assert_called_once()

    def test_slowapi_unavailable_fallback(self):
        with patch("utils.rate_limiter._slowapi_available", False):
            rl = RateLimiter(requests_per_minute=5)
            self.assertIsNone(rl._limiter)
            allowed, _ = rl.is_allowed("fallback-client")
            self.assertTrue(allowed)


if __name__ == "__main__":
    unittest.main()
