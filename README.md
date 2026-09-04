# Agent 1 — BANT Lead Scorer

Scores raw inbound leads (Name, Job Title, Company, Industry, Size, Notes, ...)
using the BANT framework, via a single CrewAI agent backed by Gemini.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste in your Gemini API key
```

Export your leads dataset (e.g. `beauty_bant_500_realistic_final`) as `leads.csv`
in this folder. It just needs a header row - column names don't need to match
exactly, since the agent reads whatever fields are present.

## Run

```bash
python main.py
```

This scores every row and writes:
- `leads_scored.csv` — original columns + BANT fields, easy to open in Sheets/Excel
- `leads_scored.json` — same data, full fidelity, easier to feed into Agent 2 later

## What each file does

| File | Purpose |
|---|---|
| `models.py` | Pydantic schema Agent 1 must return (score, qualification, per-dimension signals, rationale) |
| `agents.py` | Defines Agent 1 and its Gemini connection |
| `tasks.py` | The scoring instructions given to Agent 1 for one lead |
| `crew.py` | Wires the agent + task into a Crew, exposes `score_lead()` |
| `main.py` | Loads `leads.csv`, scores every row, saves results |

## Notes on the threshold

`qualification` is currently set to "Qualified" at `bant_score >= 60` (see
`tasks.py`). Treat this as a starting point — once you validate against a
labeled dataset (see earlier discussion on validation sets), tune this
threshold to whatever cutoff best matches real outcomes.

## Cost / rate limiting

`main.py` scores leads one at a time with a 1-second pause between calls, to
stay under Gemini's free-tier rate limits. For 500 leads this is slow
(~10+ minutes) but safe. Once this works, look into `crew.kickoff_for_each()`
or batching multiple leads per LLM call to speed this up.
