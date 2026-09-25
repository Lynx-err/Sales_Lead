import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini/gemini-2.0-flash"


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str
    gemini_model: str
    demo_mode: bool
    gmail_sender_email: str
    gmail_app_password: str
    gmail_credentials_path: str
    gmail_token_path: str

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def has_gmail_credentials(self) -> bool:
        return bool(
            self.gmail_app_password.strip() or os.path.exists(self.gmail_credentials_path) or os.path.exists(self.gmail_token_path)
        )


def get_app_config() -> AppConfig:
    return AppConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=(os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip(),
        demo_mode=str(os.getenv("DEMO_MODE", "true")).lower() in {"1", "true", "yes", "on"},
        gmail_sender_email=os.getenv("GMAIL_SENDER_EMAIL", "sales-ops@leadsense.ai").strip(),
        gmail_app_password=os.getenv("GMAIL_APP_PASSWORD", "").strip(),
        gmail_credentials_path=os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json").strip(),
        gmail_token_path=os.getenv("GMAIL_TOKEN_PATH", "token.json").strip(),
    )


def validate_gemini_config() -> dict:
    cfg = get_app_config()
    if not cfg.has_gemini_key:
        return {"status": "missing_api_key", "model": cfg.gemini_model, "configured": False}
    return {"status": "ready", "model": cfg.gemini_model, "configured": True}


def is_gmail_configured() -> bool:
    return get_app_config().has_gmail_credentials
