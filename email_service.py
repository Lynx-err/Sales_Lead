import os
import re
import base64
import uuid
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()


class EmailDeliveryService:
    """
    Enterprise-grade Gmail & GCP Email Service with built-in Safety Guards and Demo Mode.
    Supports GCP Gmail API (OAuth2) and Gmail SMTP (App Password), with automated mock simulation.
    """

    def __init__(self, demo_mode: Optional[bool] = None):
        # Demo mode is ON by default for safety unless explicitly disabled in environment
        env_demo = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
        self.demo_mode = demo_mode if demo_mode is not None else env_demo
        self.audit_log: List[Dict[str, Any]] = []

    def set_demo_mode(self, enabled: bool) -> None:
        """Dynamically toggle demo mode."""
        self.demo_mode = enabled

    def is_demo_mode(self) -> bool:
        """Returns True if the service is operating in safe demo/mock mode."""
        return self.demo_mode

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validates basic email address formatting."""
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return bool(re.match(pattern, email.strip()))

    @staticmethod
    def is_placeholder_or_sample_domain(email: str) -> bool:
        """Identifies Kaggle or fictitious demonstration domains to prevent spamming."""
        domain = email.split("@")[-1].lower() if "@" in email else ""
        test_domains = {
            "example.com", "example.org", "test.com", "sample.org",
            "valenzuela.com", "duncan-foster.org", "harmon.com",
            "alvarez.com", "robinson.org"
        }
        return domain in test_domains

    def create_mime_message(
        self, sender: str, recipient: str, subject: str, body: str
    ) -> MIMEMultipart:
        """Creates an RFC 822 compliant MIME message."""
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
        msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        sender_email: Optional[str] = None,
        force_live: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends an email or simulates delivery depending on demo_mode and credentials.
        Returns a dictionary with delivery receipt details.
        """
        now = datetime.now(timezone.utc).isoformat()
        sender = sender_email or os.getenv("GMAIL_SENDER_EMAIL", "sales-ops@leadsense.ai")

        # Validate recipient
        if not self.is_valid_email(recipient):
            record = {
                "status": "failed",
                "recipient": recipient,
                "subject": subject,
                "timestamp": now,
                "error": f"Invalid recipient email format: '{recipient}'",
                "method": "none",
            }
            self.audit_log.append(record)
            return record

        # Safe Demo Mode or Synthetic domain protection
        is_synthetic = self.is_placeholder_or_sample_domain(recipient)
        should_simulate = self.demo_mode or (not force_live) or is_synthetic

        if should_simulate:
            simulated_id = f"sim-{uuid.uuid4().hex[:12]}"
            reason = "Demo Mode is enabled" if self.demo_mode else "Safety guard for test domain"
            record = {
                "status": "simulated",
                "message_id": simulated_id,
                "recipient": recipient,
                "sender": sender,
                "subject": subject,
                "timestamp": now,
                "method": "safe_simulation",
                "notes": f"Email successfully verified and simulated ({reason}). No real message sent.",
            }
            self.audit_log.append(record)
            return record

        # Live Sending Attempt via GCP Gmail API or Gmail SMTP
        try:
            # 1. Try Gmail API if credentials exist
            api_result = self._send_via_gmail_api(sender, recipient, subject, body)
            if api_result:
                record = {**api_result, "recipient": recipient, "subject": subject, "timestamp": now}
                self.audit_log.append(record)
                return record

            # 2. Fall back to Gmail SMTP if App Password is provided
            smtp_result = self._send_via_smtp(sender, recipient, subject, body)
            if smtp_result:
                record = {**smtp_result, "recipient": recipient, "subject": subject, "timestamp": now}
                self.audit_log.append(record)
                return record

            # If no live credentials configured, return clear instructions
            record = {
                "status": "failed",
                "recipient": recipient,
                "subject": subject,
                "timestamp": now,
                "error": "No live Gmail credentials found (requires GCP credentials.json or GMAIL_APP_PASSWORD). Set DEMO_MODE=true for testing.",
                "method": "unconfigured",
            }
            self.audit_log.append(record)
            return record

        except Exception as e:
            record = {
                "status": "failed",
                "recipient": recipient,
                "subject": subject,
                "timestamp": now,
                "error": str(e),
                "method": "error",
            }
            self.audit_log.append(record)
            return record

    def _send_via_gmail_api(
        self, sender: str, recipient: str, subject: str, body: str
    ) -> Optional[Dict[str, Any]]:
        """Sends email using official Google Cloud Gmail API Client (OAuth2)."""
        creds_path = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
        token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")

        if not (os.path.exists(creds_path) or os.path.exists(token_path)):
            return None

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path)
            else:
                return None

            service = build("gmail", "v1", credentials=creds)
            msg = self.create_mime_message(sender, recipient, subject, body)
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
            response = service.users().messages().send(userId="me", body={"raw": raw}).execute()

            return {
                "status": "sent",
                "message_id": response.get("id", f"gmail-{uuid.uuid4().hex[:8]}"),
                "method": "gcp_gmail_api",
                "notes": "Sent via Google Cloud Platform Gmail API",
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Gmail API error: {str(e)}",
                "method": "gcp_gmail_api",
            }

    def _send_via_smtp(
        self, sender: str, recipient: str, subject: str, body: str
    ) -> Optional[Dict[str, Any]]:
        """Sends email using Gmail SMTP with an Application-Specific Password."""
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        sender_email = os.getenv("GMAIL_SENDER_EMAIL", sender)

        if not app_password or not sender_email:
            return None

        msg = self.create_mime_message(sender_email, recipient, subject, body)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, [recipient], msg.as_string())

        return {
            "status": "sent",
            "message_id": f"smtp-{uuid.uuid4().hex[:12]}",
            "method": "gmail_smtp",
            "notes": "Sent via Gmail SMTP (SSL 465)",
        }

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Returns history of all sent, simulated, and attempted email transmissions."""
        return self.audit_log


# Global singleton instance for clean access
default_email_service = EmailDeliveryService()
