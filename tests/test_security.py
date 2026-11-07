from __future__ import annotations

from agents_app.security import (
    SlidingWindowRateLimiter,
    key_fingerprint,
    redact_pii,
    verify_api_key,
)


def test_redacts_email_and_secret():
    text = "mail me at a.b+c@example.co.uk with token sk-abcdefghijklmnop"
    out, count = redact_pii(text)
    assert "[EMAIL]" in out
    assert "[SECRET]" in out
    assert "a.b+c@example.co.uk" not in out
    assert count == 2


def test_redacts_only_luhn_valid_cards():
    valid, _ = redact_pii("card 4111 1111 1111 1111 please")
    invalid, _ = redact_pii("number 4111 1111 1111 1112 please")
    assert "[CARD]" in valid
    assert "[CARD]" not in invalid


def test_redacts_government_ids_and_phone():
    out, _ = redact_pii("PAN ABCDE1234F aadhaar 1234 5678 9012 call +91 98765 43210")
    assert "[PAN]" in out
    assert "[AADHAAR]" in out
    assert "[PHONE]" in out


def test_empty_text_is_unchanged():
    assert redact_pii("") == ("", 0)


def test_api_key_comparison_is_exact():
    configured = ("first-key", "second-key")
    assert verify_api_key("second-key", configured) == "second-key"
    assert verify_api_key("wrong", configured) is None
    assert verify_api_key(None, configured) is None


def test_fingerprint_is_stable_and_not_the_key():
    fingerprint = key_fingerprint("super-secret")
    assert fingerprint == key_fingerprint("super-secret")
    assert fingerprint != key_fingerprint("other-secret")
    assert "super-secret" not in fingerprint


def test_rate_limiter_allows_up_to_limit_then_blocks():
    limiter = SlidingWindowRateLimiter(limit=2, window_s=60)
    assert limiter.allow("customer") == (True, 0)
    assert limiter.allow("customer") == (True, 0)
    allowed, retry_after = limiter.allow("customer")
    assert allowed is False
    assert retry_after >= 1
    assert limiter.allow("other") == (True, 0)


def test_rate_limiter_reset_clears_windows():
    limiter = SlidingWindowRateLimiter(limit=1)
    assert limiter.allow("customer")[0] is True
    limiter.reset()
    assert limiter.allow("customer")[0] is True
