import csv
import json
import time

from dotenv import load_dotenv
load_dotenv()

from crew import score_lead

INPUT_CSV = "leads.csv"          # export of your beauty_bant_500_realistic_final dataset
OUTPUT_CSV = "leads_scored.csv"
OUTPUT_JSON = "leads_scored.json"


def load_leads(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    leads = load_leads(INPUT_CSV)
    print(f"Loaded {len(leads)} leads from {INPUT_CSV}")

    scored_leads = []
    for i, lead in enumerate(leads, start=1):
        print(f"Scoring lead {i}/{len(leads)}: {lead.get('Name', 'unknown')}")
        try:
            score = score_lead(lead)
        except Exception as e:
            print(f"  Failed to score lead {i}: {e}")
            score = {
                "lead_name": lead.get("Name", "unknown"),
                "budget_signal": "error",
                "authority_signal": "error",
                "need_signal": "error",
                "timeline_signal": "error",
                "bant_score": 0,
                "qualification": "Unqualified",
                "rationale": f"Scoring failed: {e}",
            }

        scored_leads.append({**lead, **score})

        # Gentle pacing - stay comfortably under Gemini free-tier rate limits.
        # Remove or lower this once you're on a paid tier / batching properly.
        time.sleep(1)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(scored_leads, f, indent=2)

    fieldnames = list(scored_leads[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_leads)

    qualified = sum(1 for l in scored_leads if l["qualification"] == "Qualified")
    print(f"\nDone. {qualified}/{len(scored_leads)} leads marked Qualified.")
    print(f"Results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
