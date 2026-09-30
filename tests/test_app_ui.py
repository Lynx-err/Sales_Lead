from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app():
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def test_new_lead_form_starts_blank():
    app = load_app()

    assert not app.exception
    assert next(widget for widget in app.text_input if widget.label == "Lead Name").value == ""
    assert next(widget for widget in app.text_input if widget.label == "Company Name").value == ""
    assert next(widget for widget in app.text_area if widget.label == "Lead notes").value == ""


def test_csv_upload_accepts_common_header_aliases():
    app = load_app()
    app.file_uploader[0].set_value((
        "new_leads.csv",
        b"Lead Name,Company Name,Email Address,Message\nJordan Reed,Northwind,jordan@example.com,We have budget and need a CRM solution this quarter.\n",
        "text/csv",
    ))
    app.run()

    assert not app.exception
    assert any("1 leads from new_leads.csv" in item.value for item in app.caption)
    assert not next(button for button in app.button if button.label == "Run batch").disabled


def test_csv_upload_requires_name_and_notes():
    app = load_app()
    app.file_uploader[0].set_value(("invalid.csv", b"Company\nNorthwind\n", "text/csv"))
    app.run()

    assert not app.exception
    assert any("CSV must include these columns: Name, Notes" in item.value for item in app.error)
    assert next(button for button in app.button if button.label == "Run batch").disabled
