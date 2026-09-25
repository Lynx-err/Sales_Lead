# LeadSense AI

LeadSense AI is an intelligent BANT-based sales lead qualification and outreach system built for lead triage, qualification, personalized email generation, and safe demo-ready delivery. The project combines rule-based BANT logic with CrewAI + Gemini, structured Pydantic validation, a Streamlit dashboard, and optional Gmail delivery.

## Overview

The system follows the end-to-end workflow below:

Lead acquisition → lead preprocessing → BANT analysis → lead classification → email generation → email sending → Streamlit dashboard

## Problem Statement

Sales teams receive inbound leads with incomplete and inconsistent information. Without a consistent qualification process, teams waste time on leads that are not budgeted, not authorized, or not urgent enough. LeadSense AI addresses this by scoring each lead using the BANT framework and generating a tailored outreach or nurture email.

## Architecture

- `agents.py`: CrewAI agent definitions for BANT scoring and email drafting
- `tasks.py`: Structured instructions for each agent
- `crew.py`: orchestration pipeline with deterministic final qualification enforcement
- `models.py`: Pydantic models for lead, score, and email validation
- `database.py`: SQLite persistence for processed leads
- `email_service.py`: safe email sending and demo-mode behavior
- `gmail_service.py`: Gmail status wrapper for optional Gmail configuration
- `app.py`: Streamlit dashboard UI
- `main.py`: CSV batch processor
- `config.py`: environment-based configuration and validation

## BANT Methodology

Each lead is evaluated across four dimensions:

- Budget
- Authority
- Need
- Timeline

Scoring:

- 0-25 per dimension
- total score ranges from 0-100
- threshold is 60 by default for qualification
- if budget or authority is missing or unaddressed, the final score is capped at 55 in Python before qualification is decided

## Multi-Agent Architecture

- Agent 1: BANT Lead Qualification Specialist
  - Extracts evidence from a lead record
  - Produces a score and rationale
  - Enforces gating rules in a deterministic Python layer
- Agent 2: Sales Email Agent
  - Creates a personalized sales pitch for qualified leads
  - Creates a follow-up or nurture email for unqualified leads
- Agent 3: Follow-up/Nurture agent is represented through the email workflow logic for unqualified leads

## Dataset

The repository includes a sample CSV at `leads.csv` for demonstration. The pipeline accepts CSV files with common lead fields such as Name, Job Title, Company, Email, Phone, Industry, Size, and Notes.

## Installation

```bash
cd Sales_Lead
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Gemini Configuration

Update `.env` with a valid Gemini key and model:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini/gemini-2.0-flash
```

If the key is missing or the LLM fails, the application falls back to deterministic rule-based evaluation and clearly indicates configuration issues.

## Gmail Configuration

Gmail is optional. The application works in demo mode without credentials.

```env
DEMO_MODE=true
GMAIL_SENDER_EMAIL=your_address@gmail.com
GMAIL_APP_PASSWORD=your_app_password
```

Live Gmail sending requires either:

- Gmail SMTP app password, or
- Google OAuth credentials + token files

The project intentionally blocks sending unless the user explicitly confirms the action.

## Demo Mode

The project runs successfully without Gemini and without Gmail by using:

- fallback BANT scoring
- simulated email delivery
- copyable generated email drafts

This keeps the demo usable for presentations and testing.

## Streamlit Usage

```bash
streamlit run app.py
```

The app includes:

- dashboard metrics
- single-lead analysis
- batch lead processing
- email review hub
- settings and about pages

## Batch Processing

```bash
python main.py --input leads.csv --output-csv leads_scored.csv --output-json leads_scored.json
```

The batch processor saves results to CSV and JSON while continuing through failed leads without stopping the rest of the batch.

## Project Structure

```text
Sales_Lead/
├── app.py
├── main.py
├── agents.py
├── tasks.py
├── crew.py
├── models.py
├── email_service.py
├── gmail_service.py
├── database.py
├── config.py
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── leads.csv
├── IMPLEMENTATION_PROGRESS.md
└── tests/
```

## Troubleshooting

- Missing `GEMINI_API_KEY`: the app shows configuration status and continues with rule-based scoring.
- Invalid model name: check `GEMINI_MODEL` in `.env` and use a supported Gemini identifier.
- Malformed LLM output: the pipeline validates the response and falls back safely.
- Empty or malformed CSV: the loader reports a clean error and avoids crashing the whole run.
- Gmail unavailable: demo mode remains enabled and the email can still be copied or simulated.

## Future Scope

- Add CSV upload from the UI
- Add full OAuth Gmail onboarding flow
- Add more robust recruiter/CRM integrations
- Add multi-dataset validation and threshold tuning
- Expand the dashboard with lead history and export tools

## Security

Secrets are never committed. The project includes explicit ignore rules for `.env`, credentials, tokens, and private key files.

