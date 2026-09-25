import os
import csv
import json
import time
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from crew import process_lead_pipeline
from email_service import default_email_service

INPUT_CSV = "leads.csv"
OUTPUT_CSV = "leads_scored.csv"
OUTPUT_JSON = "leads_scored.json"


def load_leads(path: str) -> List[Dict[str, Any]]:
    """Loads lead records from a CSV file."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def process_leads_batch(
    leads: List[Dict[str, Any]],
    threshold: int = 60,
    send_email: bool = False,
    limit: int = 0,
    pacing_seconds: float = 1.0,
) -> List[Dict[str, Any]]:
    """Processes a batch of leads through the LeadSense AI multi-agent pipeline."""
    target_leads = leads[:limit] if limit > 0 else leads
    total = len(target_leads)
    results = []

    print(f"\n=======================================================")
    print(f"  LeadSense AI — Multi-Agent Batch Lead Qualification  ")
    print(f"  Processing {total} leads | Threshold: {threshold} | Email Auto-Send: {send_email}")
    print(f"=======================================================\n")

    for i, lead in enumerate(target_leads, start=1):
        name = lead.get("Name", "Unknown")
        company = lead.get("Company", "Unknown")
        print(f"[{i}/{total}] Analyzing Lead: {name} ({company})...")

        try:
            res = process_lead_pipeline(
                lead=lead,
                threshold=threshold,
                send_email=send_email,
                email_service=default_email_service,
            )
            score = res["bant_score"]["bant_score"]
            status = res["bant_score"]["qualification"]
            b = res["bant_score"]["budget_score"]
            a = res["bant_score"]["authority_score"]
            n = res["bant_score"]["need_score"]
            t = res["bant_score"]["timeline_score"]
            email_status = res["email_delivery_status"]
            print(f"   -> Result: {status} (BANT: {score}/100 | B:{b} A:{a} N:{n} T:{t}) | Email: {email_status}")
            results.append(res)
        except Exception as e:
            print(f"   -> Error processing lead {name}: {e}")
            # Resilient fallback error record
            err_record = {
                "lead_data": lead,
                "bant_score": {
                    "lead_name": name,
                    "budget_score": 0,
                    "authority_score": 0,
                    "need_score": 0,
                    "timeline_score": 0,
                    "budget_signal": "error",
                    "authority_signal": "error",
                    "need_signal": "error",
                    "timeline_signal": "error",
                    "bant_score": 0,
                    "qualification": "Unqualified",
                    "rationale": f"Processing encountered error: {e}",
                    "key_pain_points": [],
                    "recommended_action": "Manual review required",
                },
                "email_response": None,
                "email_delivery_status": "failed",
                "email_sent_at": None,
                "delivery_notes": str(e),
            }
            results.append(err_record)

        if i < total and pacing_seconds > 0:
            time.sleep(pacing_seconds)

    return results


def save_results(results: List[Dict[str, Any]], csv_path: str, json_path: str) -> None:
    """Exports structured results to JSON and flat CSV formats."""
    # 1. Save Full Fidelity JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Flatten for CSV export
    flat_rows = []
    for r in results:
        lead = r.get("lead_data", {})
        score = r.get("bant_score", {})
        email = r.get("email_response") or {}

        row = {
            # Original lead attributes
            **lead,
            # BANT dimension scores and signals
            "BANT_Total_Score": score.get("bant_score", 0),
            "Qualification": score.get("qualification", "Unqualified"),
            "Budget_Score": score.get("budget_score", 0),
            "Authority_Score": score.get("authority_score", 0),
            "Need_Score": score.get("need_score", 0),
            "Timeline_Score": score.get("timeline_score", 0),
            "Budget_Signal": score.get("budget_signal", ""),
            "Authority_Signal": score.get("authority_signal", ""),
            "Need_Signal": score.get("need_signal", ""),
            "Timeline_Signal": score.get("timeline_signal", ""),
            "Scoring_Rationale": score.get("rationale", ""),
            "Recommended_Action": score.get("recommended_action", ""),
            # Email information
            "Email_Subject": email.get("subject", ""),
            "Email_Type": email.get("email_type", ""),
            "Email_Call_To_Action": email.get("call_to_action", ""),
            "Email_Body": email.get("body", ""),
            "Email_Delivery_Status": r.get("email_delivery_status", "draft"),
            "Email_Sent_At": r.get("email_sent_at", ""),
        }
        flat_rows.append(row)

    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    qualified_count = sum(
        1 for r in results if r.get("bant_score", {}).get("qualification") == "Qualified"
    )
    print(f"\nProcessing Complete!")
    print(f"Summary: {qualified_count}/{len(results)} leads qualified.")
    print(f"Saved results to: {csv_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(description="LeadSense AI Lead Qualification Engine")
    parser.add_argument("--input", default=INPUT_CSV, help="Path to input leads CSV")
    parser.add_argument("--output-csv", default=OUTPUT_CSV, help="Path to output CSV")
    parser.add_argument("--output-json", default=OUTPUT_JSON, help="Path to output JSON")
    parser.add_argument("--threshold", type=int, default=60, help="BANT qualification threshold (0-100)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of leads to process (0 = all)")
    parser.add_argument("--send-email", action="store_true", help="Automatically trigger email delivery / safe simulation")
    parser.add_argument("--pacing", type=float, default=1.0, help="Pause between leads in seconds")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    leads = load_leads(args.input)
    results = process_leads_batch(
        leads=leads,
        threshold=args.threshold,
        send_email=args.send_email,
        limit=args.limit,
        pacing_seconds=args.pacing,
    )
    save_results(results, args.output_csv, args.output_json)


if __name__ == "__main__":
    main()
