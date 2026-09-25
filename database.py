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


def save_lead_result(result: Dict[str, Any], db_path: str | Path = DB_PATH) -> int:
    initialize_db(db_path)
    lead = result.get("lead_data", {})
    score = result.get("bant_score", {})
    email = result.get("email_response") or {}
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
                str(result),
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
