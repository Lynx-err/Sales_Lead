import os
from typing import Any, Dict, Optional

from config import get_app_config
from email_service import EmailDeliveryService


class GmailService(EmailDeliveryService):
    """Thin wrapper around the existing email service with explicit Gmail config checks."""

    def __init__(self, demo_mode: Optional[bool] = None):
        super().__init__(demo_mode=demo_mode)
        self.config = get_app_config()

    def is_configured(self) -> bool:
        return self.config.has_gmail_credentials

    def get_status(self) -> Dict[str, Any]:
        return {
            "demo_mode": self.demo_mode,
            "gmail_configured": self.is_configured(),
            "sender_email": self.config.gmail_sender_email,
            "credentials_path": self.config.gmail_credentials_path,
            "token_path": self.config.gmail_token_path,
        }


def get_gmail_service() -> GmailService:
    return GmailService(demo_mode=get_app_config().demo_mode)


def build_gmail_status() -> Dict[str, Any]:
    return get_gmail_service().get_status()
