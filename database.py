import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent / "leads.db"


def connect_db(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(db_path: str | Path = DB_PATH) -> None:
    with connect_db(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_name TEXT,
                company TEXT,
                email TEXT,
                qualification TEXT,
                bant_score INTEGER,
                email_type TEXT,
                result_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_name TEXT,
                company TEXT,
                email TEXT UNIQUE,
                classification TEXT,
                intent TEXT,
                sentiment TEXT,
                qualification TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lead_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER,
                interaction_type TEXT,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bant_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER,
                score INTEGER,
                budget INTEGER,
                authority INTEGER,
                need INTEGER,
                timeline INTEGER,
                qualification TEXT,
                rationale TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER,
                subject TEXT,
                body TEXT,
                validation_status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _coalesce(lead: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = lead.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def upsert_lead(lead: Dict[str, Any], db_path: str | Path = DB_PATH) -> int:
    initialize_db(db_path)
    email = _coalesce(lead, "Email", "email")
    company = _coalesce(lead, "Company", "company")
    lead_name = _coalesce(lead, "Name", "name")

    with connect_db(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM leads WHERE email = ? OR (lead_name = ? AND company = ?)",
            (email or None, lead_name or None, company or None),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE leads
                SET lead_name = ?, company = ?, email = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (lead_name or "Unknown", company or "Unknown", email or "", int(existing["id"])),
            )
            return int(existing["id"])

        cursor = conn.execute(
            """
            INSERT INTO leads (lead_name, company, email, classification, intent, sentiment, qualification)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_name or "Unknown",
                company or "Unknown",
                email or "",
                None,
                None,
                None,
                None,
            ),
        )
        return int(cursor.lastrowid)


def record_lead_interaction(lead_id: int, interaction_type: str, details: Any, db_path: str | Path = DB_PATH) -> None:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        conn.execute(
            "INSERT INTO lead_interactions (lead_id, interaction_type, details) VALUES (?, ?, ?)",
            (lead_id, interaction_type, json.dumps(details, default=str) if isinstance(details, (dict, list)) else str(details)),
        )


def record_bant_score(lead_id: int, bant_data: Dict[str, Any], db_path: str | Path = DB_PATH) -> None:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO bant_scores (lead_id, score, budget, authority, need, timeline, qualification, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_id,
                int(bant_data.get("bant_score", 0) or 0),
                int(bant_data.get("budget_score", 0) or 0),
                int(bant_data.get("authority_score", 0) or 0),
                int(bant_data.get("need_score", 0) or 0),
                int(bant_data.get("timeline_score", 0) or 0),
                bant_data.get("qualification") or "Unqualified",
                bant_data.get("rationale") or "",
            ),
        )


def record_generated_email(lead_id: int, email_data: Dict[str, Any], validation_status: str, db_path: str | Path = DB_PATH) -> None:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        conn.execute(
            "INSERT INTO generated_emails (lead_id, subject, body, validation_status) VALUES (?, ?, ?, ?)",
            (
                lead_id,
                email_data.get("subject") or "",
                email_data.get("body") or "",
                validation_status,
            ),
        )


def get_lead_history(lead_id: int, db_path: str | Path = DB_PATH) -> List[Dict[str, Any]]:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        rows = conn.execute(
            "SELECT interaction_type, details, created_at FROM lead_interactions WHERE lead_id = ? ORDER BY created_at ASC",
            (lead_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_bant_history(lead_id: int, db_path: str | Path = DB_PATH) -> List[Dict[str, Any]]:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        rows = conn.execute(
            "SELECT score, budget, authority, need, timeline, qualification, rationale, created_at FROM bant_scores WHERE lead_id = ? ORDER BY created_at ASC",
            (lead_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_lead_result(result: Dict[str, Any], db_path: str | Path = DB_PATH) -> int:
    initialize_db(db_path)
    lead = result.get("lead_data", {})
    score = result.get("bant_score", {})
    email = result.get("email_response") or {}
    classification = result.get("classification") or {}
    intent = result.get("intent") or {}
    sentiment = result.get("sentiment") or {}
    lead_id = upsert_lead(lead, db_path)

    with connect_db(db_path) as conn:
        conn.execute(
            """
            UPDATE leads
            SET classification = ?, intent = ?, sentiment = ?, qualification = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                classification.get("classification") if isinstance(classification, dict) else None,
                intent.get("intent") if isinstance(intent, dict) else None,
                sentiment.get("sentiment") if isinstance(sentiment, dict) else None,
                score.get("qualification") or "Unqualified",
                lead_id,
            ),
        )

    record_lead_interaction(lead_id, "lead_received", lead, db_path)
    if score:
        record_bant_score(lead_id, score, db_path)
    if type(email) is dict and email:
        record_generated_email(lead_id, email, (result.get("email_validation") or {}).get("status", "PASS") if isinstance(result.get("email_validation"), dict) else "PASS", db_path)
        record_lead_interaction(lead_id, "email_generated", email, db_path)
    if classification:
        record_lead_interaction(lead_id, "classification", classification, db_path)
    if intent:
        record_lead_interaction(lead_id, "intent", intent, db_path)
    if sentiment:
        record_lead_interaction(lead_id, "sentiment", sentiment, db_path)
    if result.get("next_best_action"):
        record_lead_interaction(lead_id, "next_best_action", result.get("next_best_action"), db_path)
    if result.get("email_validation"):
        record_lead_interaction(lead_id, "email_validation", result.get("email_validation"), db_path)

    with connect_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO processed_leads (
                lead_name,
                company,
                email,
                qualification,
                bant_score,
                email_type,
                result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead.get("Name") or lead.get("name") or "Unknown",
                lead.get("Company") or lead.get("company") or "Unknown",
                lead.get("Email") or lead.get("email") or "",
                score.get("qualification") or "Unqualified",
                score.get("bant_score") or 0,
                email.get("email_type") or "",
                json.dumps(result, default=str),
            ),
        )
        return int(cursor.lastrowid)


def get_recent_leads(limit: int = 20, db_path: str | Path = DB_PATH) -> List[Dict[str, Any]]:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM processed_leads
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_lead_by_id(lead_id: int, db_path: str | Path = DB_PATH) -> Optional[Dict[str, Any]]:
    initialize_db(db_path)
    with connect_db(db_path) as conn:
        row = conn.execute("SELECT * FROM processed_leads WHERE id = ?", (lead_id,)).fetchone()
    return dict(row) if row is not None else None
