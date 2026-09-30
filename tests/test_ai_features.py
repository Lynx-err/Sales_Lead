import pytest
import crew
import database

from crew import (
    classify_email,
    detect_intent,
    analyze_sentiment,
    get_next_best_action,
    validate_generated_email,
    process_lead_pipeline,
)


def _use_test_database(monkeypatch, tmp_path):
    db_path = tmp_path / "leads.db"
    monkeypatch.setattr(crew, "save_lead_result", lambda result: database.save_lead_result(result, db_path))
    monkeypatch.setattr(crew, "get_lead_history", lambda lead_id: database.get_lead_history(lead_id, db_path))
    monkeypatch.setattr(crew, "get_bant_history", lambda lead_id: database.get_bant_history(lead_id, db_path))


def test_spam_classification_stops_bant():
    lead = {
        "Name": "Test User",
        "Company": "Example",
        "Email": "user@example.com",
        "Notes": "You have won a free iPhone. Click here now to claim your reward!"
    }
    result = classify_email(lead)
    assert result["classification"] == "SPAM"
    assert result["confidence"] >= 0.5


def test_non_sales_classification_stops_bant():
    lead = {
        "Name": "Support User",
        "Company": "Client Corp",
        "Email": "support@company.com",
        "Notes": "Please reset my account password and help with login issue."
    }
    result = classify_email(lead)
    assert result["classification"] == "NON_SALES"


def test_genuine_lead_classification_runs_bant():
    lead = {
        "Name": "Maria Olson",
        "Job Title": "Head of Merchandising",
        "Company": "Gray, Olson and Anderson Beauty",
        "Email": "maria@example.com",
        "Notes": "We are looking for a CRM solution for our company and have a $75k budget for implementation within 2 months."
    }
    result = classify_email(lead)
    assert result["classification"] == "GENUINE_LEAD"


def test_intent_detection_for_purchase_inquiry():
    text = "We are evaluating vendors for an upcoming purchase and want pricing for enterprise rollout."
    result = detect_intent(text)
    assert result["intent"] == "PURCHASE_INTENT"
    assert result["confidence"] >= 0.5


def test_sentiment_detection_positive_and_negative():
    positive = analyze_sentiment("We are excited to move forward and would love to discuss pricing.")
    negative = analyze_sentiment("We are frustrated by delays and need a better solution quickly.")
    assert positive["sentiment"] == "POSITIVE"
    assert negative["sentiment"] in {"NEGATIVE", "NEUTRAL"}


def test_next_best_action_recommends_from_signal():
    lead = {
        "Name": "Cora",
        "Company": "Northwind Retail",
        "Notes": "We have a confirmed $120k budget and need a CRM rollout in 6 weeks."
    }
    action = get_next_best_action(lead, {"qualification": "Qualified", "bant_score": 82})
    assert action["action"] in {"SCHEDULE_SALES_CALL", "REQUEST_DEMO"}
    assert action["priority"] in {"HIGH", "MEDIUM"}


def test_hallucination_checker_fails_for_unsupported_claim():
    email = {
        "subject": "Demo offer",
        "body": "We offer a 30-day free trial on all enterprise plans.",
    }
    result = validate_generated_email(email, {"lead_data": {}, "bant_score": {}})
    assert result["is_valid"] is False
    assert result["status"] == "FAIL"


def test_hallucination_checker_passes_for_supported_claim():
    email = {
        "subject": "Next steps on your rollout",
        "body": "We can review your requirements and arrange a 15-minute discovery call.",
    }
    result = validate_generated_email(email, {"lead_data": {"Notes": "We need a demo and have budget."}, "bant_score": {"qualification": "Qualified"}})
    assert result["is_valid"] is True
    assert result["status"] == "PASS"


def test_pipeline_tracks_history_and_bant_evolution(monkeypatch, tmp_path):
    _use_test_database(monkeypatch, tmp_path)
    lead = {
        "Name": "Emma Lee",
        "Company": "Oak & Co.",
        "Email": "emma@oakco.com",
        "Notes": "We need a CRM solution for our store network and have budget available. We are evaluating vendors for an upcoming purchase and want pricing."
    }
    result = process_lead_pipeline(lead, threshold=60, send_email=False, use_llm=False)
    assert result["classification"]["classification"] == "GENUINE_LEAD"
    assert result["intent"]["intent"] != "UNKNOWN"
    assert result["sentiment"]["sentiment"] in {"POSITIVE", "NEUTRAL", "NEGATIVE"}
    assert result["next_best_action"]["action"]
    assert result["email_validation"]["status"] in {"PASS", "FAIL"}
    assert result["email_response"] is not None
    assert result["lead_history"]
    assert result["bant_history"]
    assert result["bant_history"][-1]["score"] == result["bant_score"]["bant_score"]


@pytest.mark.parametrize(
    ("notes", "expected_classification"),
    [
        ("You have won a free iPhone. Click here now to claim your reward!", "SPAM"),
        ("Please reset my account password and help with login issue.", "NON_SALES"),
    ],
)
def test_pipeline_filters_non_sales_before_bant(notes, expected_classification, monkeypatch, tmp_path):
    _use_test_database(monkeypatch, tmp_path)
    monkeypatch.setattr(crew, "score_lead", lambda *args, **kwargs: pytest.fail("BANT scoring should be skipped"))
    lead = {
        "Name": "Filtered User",
        "Company": "Example Corp",
        "Email": "filtered@example.com",
        "Notes": notes,
    }

    result = process_lead_pipeline(lead, send_email=False, use_llm=False)

    assert result["classification"]["classification"] == expected_classification
    assert result["bant_score"]["bant_score"] == 0
    assert result["email_response"] is None
    assert result["email_validation"]["status"] == "SKIPPED"
    assert result["lead_history"]
    assert result["bant_history"]
