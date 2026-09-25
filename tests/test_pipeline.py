import pytest
from crew import _heuristic_bant_fallback, _heuristic_email_fallback, process_lead_pipeline
from email_service import EmailDeliveryService


def test_heuristic_bant_scoring_qualified():
    """Tests high-signal qualified lead scoring."""
    lead = {
        "Name": "Maria Olson",
        "Job Title": "Head of Merchandising",
        "Company": "Gray, Olson and Anderson Beauty",
        "Notes": "Our indicative budget for this initiative is $250k. We're targeting to move forward within 2 months. I will be the final sign-off on supplier selection.",
    }
    result = _heuristic_bant_fallback(lead, threshold=60)
    assert result["bant_score"] >= 60
    assert result["qualification"] == "Qualified"
    assert result["budget_score"] >= 20
    assert result["authority_score"] >= 20
    assert result["timeline_score"] >= 20


def test_heuristic_bant_scoring_gating_rule():
    """
    Tests gating rule: If budget or authority is missing (<= 5),
    score cannot exceed 55 and lead cannot be Qualified.
    """
    lead_no_budget = {
        "Name": "Unfunded Prospect",
        "Job Title": "Intern",
        "Company": "Unknown Co",
        "Notes": "We urgently need lipsticks within 2 months immediately!",
    }
    result = _heuristic_bant_fallback(lead_no_budget, threshold=60)
    assert result["bant_score"] <= 55
    assert result["qualification"] == "Unqualified"


def test_heuristic_email_generation_qualified():
    """Tests pitch email generation for qualified leads."""
    lead = {
        "Name": "Zachary Ford",
        "Company": "Andrade Group Beauty",
        "Email": "david52@valenzuela.com",
    }
    bant_info = {"qualification": "Qualified"}
    email = _heuristic_email_fallback(lead, bant_info)
    assert email["email_type"] == "sales_pitch"
    assert "Andrade Group Beauty" in email["subject"]
    assert "discovery call" in email["call_to_action"].lower() or "call" in email["call_to_action"].lower()


def test_heuristic_email_generation_unqualified():
    """Tests nurture email generation for unqualified leads."""
    lead = {
        "Name": "Julie Hurst",
        "Company": "Snyder, Allen and Alexander Beauty",
        "Email": "millereric@harmon.com",
    }
    bant_info = {"qualification": "Unqualified"}
    email = _heuristic_email_fallback(lead, bant_info)
    assert email["email_type"] == "nurturing_followup"
    assert "future planning" in email["subject"].lower() or "connecting" in email["subject"].lower()


def test_end_to_end_pipeline_simulation():
    """Tests full pipeline orchestration with mock/simulation email service."""
    test_service = EmailDeliveryService(demo_mode=True)
    lead = {
        "Name": "Maria Olson",
        "Job Title": "Head of Merchandising",
        "Company": "Gray, Olson and Anderson Beauty",
        "Email": "amandacortez@duncan-foster.org",
        "Notes": "Our indicative budget is $250k. We're targeting to move forward within 2 months.",
    }
    res = process_lead_pipeline(lead, threshold=60, send_email=True, email_service=test_service)
    assert "bant_score" in res
    assert "email_response" in res
    assert res["email_delivery_status"] == "simulated"
    assert res["bant_score"]["bant_score"] >= 60
