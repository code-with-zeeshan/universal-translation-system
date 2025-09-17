import unittest
from utils.sensitive_filter import SensitiveDataFilter

class TestSensitiveDataFilter(unittest.TestCase):
    def setUp(self):
        self.f = SensitiveDataFilter()

    def test_redact_email(self):
        s = "Contact me at user@example.com for details."
        out = self.f.sanitize(s)
        self.assertNotIn("user@example.com", out)
        self.assertIn("<redacted_email>", out)

    def test_redact_phone(self):
        s = "Call +1 (415) 555-1234 now!"
        out = self.f.sanitize(s)
        self.assertNotIn("415", out)
        self.assertIn("<redacted_phone>", out)

    def test_redact_token(self):
        s = 'token="ABCD1234EFGH5678" should not leak'
        out = self.f.sanitize(s)
        self.assertIn("<redacted_secret>", out)

    def test_redact_jwt(self):
        s = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
        out = self.f.sanitize(s)
        self.assertIn("<redacted_jwt>", out)

    def test_redact_aws_keys(self):
        s = 'AKIAABCDEFGHIJKLMNOP and secret ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD'
        out = self.f.sanitize(s)
        self.assertNotIn('AKIAABCDEFGHIJKLMNOP', out)
        # May redact generic secret pattern
        self.assertIn('<redacted_secret>', out) or self.assertIn('<redacted_aws_access_key>', out)

    def test_redact_cards(self):
        s = 'my card is 4242 4242 4242 4242'
        out = self.f.sanitize(s)
        self.assertIn('<redacted_card>', out)

    def test_no_change_on_clean_text(self):
        s = "hello world"
        out = self.f.sanitize(s)
        self.assertEqual(s, out)

if __name__ == "__main__":
    unittest.main()