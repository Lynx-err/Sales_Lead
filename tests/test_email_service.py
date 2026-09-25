import pytest
from email_service import EmailDeliveryService


def test_email_validation():
    """Tests email address syntax validation."""
    service = EmailDeliveryService(demo_mode=True)
    assert service.is_valid_email("prospect@company.com") is True
    assert service.is_valid_email("john.doe+sales@test.co.uk") is True
    assert service.is_valid_email("invalid-address") is False
    assert service.is_valid_email("no-at-domain.com") is False


def test_test_domain_guard():
    """Tests identification of placeholder/Kaggle domains."""
    service = EmailDeliveryService(demo_mode=True)
    assert service.is_placeholder_or_sample_domain("david52@valenzuela.com") is True
    assert service.is_placeholder_or_sample_domain("user@example.com") is True
    assert service.is_placeholder_or_sample_domain("realuser@google.com") is False


def test_demo_mode_simulation():
    """Tests that demo mode safely simulates delivery and produces audit receipts."""
    service = EmailDeliveryService(demo_mode=True)
    receipt = service.send_email(
        recipient="prospect@company.com",
        subject="LeadSense AI Demo",
        body="Hello, this is a simulated sales lead message.",
    )
    assert receipt["status"] == "simulated"
    assert "sim-" in receipt["message_id"]
    assert receipt["recipient"] == "prospect@company.com"
    assert len(service.get_audit_log()) == 1


def test_invalid_email_rejection():
    """Tests that malformed email addresses are cleanly rejected with failed status."""
    service = EmailDeliveryService(demo_mode=True)
    receipt = service.send_email(
        recipient="malformed-email",
        subject="Test Subject",
        body="Test Body",
    )
    assert receipt["status"] == "failed"
    assert "Invalid recipient email" in receipt["error"]


def test_mime_message_creation():
    """Tests generation of RFC 822 compliant MIME multipart objects."""
    service = EmailDeliveryService(demo_mode=True)
    msg = service.create_mime_message(
        sender="sales@leadsense.ai",
        recipient="buyer@retailer.com",
        subject="Exclusive Wholesale Terms",
        body="Dear Buyer,\nHere are the terms.",
    )
    assert msg["From"] == "sales@leadsense.ai"
    assert msg["To"] == "buyer@retailer.com"
    assert msg["Subject"] == "Exclusive Wholesale Terms"
    assert "Date" in msg
