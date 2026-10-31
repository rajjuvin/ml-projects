import html
import json
from importlib import resources
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from readmission import __version__
from readmission.app import create_app
from readmission.predict import DECLARATIONS, Predictor
from readmission.schema import EXAMPLE_ENCOUNTER, MAX_ROWS, MAX_UPLOAD_BYTES

SECRET = "SECRET-VALUE-123"
SAMPLE = resources.files("readmission").joinpath("web/sample_encounters.csv").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def client(model_dir):
    return TestClient(create_app(model_dir), raise_server_exceptions=False)


def post_file(client, text):
    return client.post("/api/predict-file", content=text.encode("utf-8"), headers={"Content-Type": "text/csv"})


def header_and_rows():
    lines = SAMPLE.strip().splitlines()
    return lines[0], lines[1:]


def test_health_and_schema(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and 0 < health["threshold"] < 1
    assert health["app_version"] == __version__ and health["model_created_utc"]  # tells a deploy or rollback apart
    described = client.get("/api/schema").json()
    assert described["example"] == json.loads(json.dumps(EXAMPLE_ENCOUNTER))
    assert len(described["required_columns"]) == 22
    diagnoses = {field["name"]: field["optional"] for field in described["codes"]}
    assert diagnoses == {"diag_1": False, "diag_2": True, "diag_3": True}


def test_page_assets_and_privacy_headers(client):
    page = client.get("/")
    assert page.status_code == 200
    # the project name, objective and every declaration are in the page itself, outside the switching views
    header = page.text.split("</header>")[0]
    assert "Hospital Readmission Risk Prediction" in header and "within 30 days" in header
    for declaration in DECLARATIONS:
        assert html.escape(declaration.split(". ")[0]) in header
    assert "Do not enter real patient information" in header and "{{" not in page.text
    for path in ("/static/app.js", "/static/style.css", "/sample.csv"):
        assert client.get(path).status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    assert client.get("/docs").status_code == 404  # no interactive API docs are exposed


def test_predict_returns_only_row_score_flag_and_text(client):
    response = client.post("/api/predict", json=EXAMPLE_ENCOUNTER)
    assert response.status_code == 200
    data = response.json()
    (result,) = data["results"]
    assert set(result) == {"row", "risk_score", "flagged", "message"}
    assert result["row"] == 1 and 0 <= result["risk_score"] <= 1 and isinstance(result["flagged"], bool)
    assert any("not a calibrated probability" in note for note in data["notes"])
    assert any("not stored" in declaration for declaration in data["declarations"])
    for submitted in ("Caucasian", "InternalMedicine", "428", "Steady"):
        assert submitted not in response.text  # the response never echoes what was entered


def test_medications_left_out_count_as_no(client):
    without = {key: value for key, value in EXAMPLE_ENCOUNTER.items() if key != "medications"}
    without.update({"diabetesMed": "No", "change": "No"})  # consistent with listing no medication
    explicit = {**without, "medications": {"insulin": "No", "metformin": "No", "glipizide": "No"}}
    assert (client.post("/api/predict", json=without).json()["results"]
            == client.post("/api/predict", json=explicit).json()["results"])


def test_predict_rejects_bad_input_without_echoing_it(client, caplog):
    bad = {**EXAMPLE_ENCOUNTER, "age": SECRET, "race": ""}
    response = client.post("/api/predict", json=bad)
    assert response.status_code == 422
    assert sorted(response.json()["errors"]) == ["age: not an allowed value", "race: required"]
    assert SECRET not in response.text and SECRET not in caplog.text


def test_predict_rejects_unusable_requests(client):
    assert client.post("/api/predict", content=b"{not json").status_code == 400
    assert client.post("/api/predict", json=[1, 2]).status_code == 400
    assert client.post("/api/predict", json={**EXAMPLE_ENCOUNTER, "medications": "x"}).status_code == 400
    unknown = client.post("/api/predict", json={**EXAMPLE_ENCOUNTER, "medications": {"aspirin": "Up"}})
    assert unknown.status_code == 422 and "aspirin" in unknown.json()["errors"][0]
    assert client.post("/api/predict", content=b" " * 20_000).status_code == 413


def test_medication_answers_must_agree_with_the_listed_drugs(client):
    no_drugs = {key: value for key, value in EXAMPLE_ENCOUNTER.items() if key != "medications"}
    response = client.post("/api/predict", json=no_drugs)  # prescribed = Yes but nothing listed
    assert response.status_code == 422
    assert response.json()["errors"] == ["diabetesMed: Yes needs at least one medication that is not No."]

    header, rows = header_and_rows()
    columns = header.split(",")
    values = rows[1].split(",")  # prescribed = No, no drugs: valid
    values[columns.index("insulin")] = "Steady"
    errors = post_file(client, "\n".join([header, rows[0], ",".join(values)])).json()["errors"]
    assert errors == ["Row 2, diabetesMed: No cannot be combined with a medication that is not No."]


def test_optional_diagnoses_and_new_gender_values_can_be_scored(client):
    without = {key: value for key, value in EXAMPLE_ENCOUNTER.items() if key not in ("diag_2", "diag_3")}
    assert client.post("/api/predict", json=without).status_code == 200
    for gender in ("Non-binary", "Other", "Not declared"):
        assert client.post("/api/predict", json={**EXAMPLE_ENCOUNTER, "gender": gender}).status_code == 200


def test_a_file_may_leave_out_the_optional_diagnosis_columns(client):
    header, rows = header_and_rows()
    columns = header.split(",")
    keep = [i for i, name in enumerate(columns) if name not in ("diag_2", "diag_3")]
    text = "\n".join(",".join(line.split(",")[i] for i in keep) for line in [header, *rows])
    assert post_file(client, text).status_code == 200


def test_the_sample_file_scores_every_row(client):
    response = post_file(client, SAMPLE)
    assert response.status_code == 200
    assert [result["row"] for result in response.json()["results"]] == [1, 2, 3]


def test_a_file_with_the_row_limit_is_accepted_and_one_more_is_not(client):
    header, rows = header_and_rows()
    at_limit = "\n".join([header, *(rows * 4)[:MAX_ROWS]])
    assert post_file(client, at_limit).status_code == 200
    too_many = "\n".join([header, *(rows * 4)[:MAX_ROWS + 1]])
    response = post_file(client, too_many)
    assert response.status_code == 422 and "more than 10 rows" in response.json()["errors"][0]


def test_file_size_is_capped(client):
    response = post_file(client, "x" * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 413


def test_file_header_problems_are_listed_once(client):
    header, rows = header_and_rows()
    columns = header.split(",")
    broken = ",".join(["patient_nbr" if c == "age" else c for c in columns])
    errors = post_file(client, "\n".join([broken, *rows])).json()["errors"]
    assert errors == ["Unknown columns: patient_nbr.", "Missing columns: age."]
    assert post_file(client, "").status_code == 400
    assert "no rows" in post_file(client, header).json()["errors"][0]


def test_file_row_errors_name_the_row_and_column_only(client):
    header, rows = header_and_rows()
    columns = header.split(",")
    values = rows[1].split(",")
    values[columns.index("age")] = SECRET
    response = post_file(client, "\n".join([header, rows[0], ",".join(values)]))
    assert response.status_code == 422
    assert response.json()["errors"] == ["Row 2, age: not an allowed value"]
    assert SECRET not in response.text


def test_file_rows_with_the_wrong_number_of_values_are_rejected(client):
    header, rows = header_and_rows()
    response = post_file(client, "\n".join([header, rows[0] + ",extra", rows[1].rsplit(",", 1)[0]]))
    assert response.json()["errors"] == [
        "Row 1: the number of values does not match the header.",
        "Row 2: the number of values does not match the header.",
    ]


def test_a_file_that_is_not_text_is_rejected(client):
    response = client.post("/api/predict-file", content=b"\xff\xfe\x00\x01", headers={"Content-Type": "text/csv"})
    assert response.status_code == 400


def test_nothing_is_written_or_logged(model_dir, in_tmp_cwd, caplog):
    client = TestClient(create_app(model_dir), raise_server_exceptions=False)
    header, rows = header_and_rows()
    columns = header.split(",")
    values = rows[0].split(",")
    values[columns.index("payer_code")] = SECRET
    post_file(client, "\n".join([header, ",".join(values)]))
    client.post("/api/predict", json={**EXAMPLE_ENCOUNTER, "payer_code": SECRET})
    client.post("/api/predict", json=EXAMPLE_ENCOUNTER)
    assert list(in_tmp_cwd.iterdir()) == []
    assert SECRET not in caplog.text


def test_unexpected_errors_give_a_generic_message(model_dir, monkeypatch):
    def explode(self, rows):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(Predictor, "score", explode)
    client = TestClient(create_app(model_dir), raise_server_exceptions=False)
    response = client.post("/api/predict", json=EXAMPLE_ENCOUNTER)
    assert response.status_code == 500 and SECRET not in response.text


def test_flag_depends_on_the_threshold(model_dir):
    predictor = Predictor.load(model_dir)
    predictor.threshold = 0.3
    assert predictor.result(1, 0.3)["flagged"] is True
    assert predictor.result(1, 0.299)["flagged"] is False
    assert "below the cutoff of 0.30" in predictor.result(1, 0.1)["message"]
    assert predictor.result(1, 0.5)["message"].startswith("Flagged for follow-up review")


REAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.mark.skipif(not (REAL_MODEL_DIR / "readmission_model.joblib").exists(),
                    reason="needs the trained model (local only; CI has no model file)")
def test_example_and_sample_work_with_the_real_model():
    client = TestClient(create_app(REAL_MODEL_DIR), raise_server_exceptions=False)
    assert client.post("/api/predict", json=EXAMPLE_ENCOUNTER).status_code == 200
    assert post_file(client, SAMPLE).status_code == 200


@pytest.mark.skipif(not (REAL_MODEL_DIR / "readmission_model.joblib").exists(),
                    reason="needs the trained model (local only; CI has no model file)")
def test_non_binary_other_and_not_declared_are_scored_like_the_models_rare_gender_category():
    predictor = Predictor.load(REAL_MODEL_DIR)
    row, _ = predictor.schema.clean({key: value for key, value in EXAMPLE_ENCOUNTER.items() if key != "medications"})
    scores = {}
    # the last is the dataset's own rare value
    for gender in ("Non-binary", "Other", "Not declared", "Unknown/Invalid"):
        scores[gender] = predictor.score([{**row, "gender": gender}])[0]
    assert scores["Non-binary"] == scores["Other"] == scores["Not declared"] == scores["Unknown/Invalid"]
