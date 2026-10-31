import csv
import io
from importlib import resources

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from readmission import config
from readmission.app import create_app
from readmission.predict import Predictor
from readmission.samples import DATA_SHEET, GUIDE_SHEET, build_sample_workbook
from readmission.schema import REQUIRED_COLUMNS

SAMPLE_CSV = resources.files("readmission").joinpath("web/sample_encounters.csv").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workbook(model_dir):
    schema = Predictor.load(model_dir).schema
    return load_workbook(io.BytesIO(build_sample_workbook(schema.guide(), SAMPLE_CSV)))


def test_workbook_has_the_data_sheet_first_and_a_field_guide(workbook):
    assert workbook.sheetnames == [DATA_SHEET, GUIDE_SHEET]
    assert workbook.active.title == DATA_SHEET  # opens on the sheet to be saved as CSV


def test_data_sheet_holds_the_same_encounters_as_the_csv(workbook):
    sheet_rows = list(workbook[DATA_SHEET].values)
    csv_rows = list(csv.reader(io.StringIO(SAMPLE_CSV)))
    assert list(sheet_rows[0]) == csv_rows[0]
    assert len(sheet_rows) == len(csv_rows) == 4
    assert sheet_rows[1][0] == 4 and sheet_rows[1][csv_rows[0].index("diag_1")] == "428"  # numbers vs text kept


def test_field_guide_explains_every_column(workbook):
    rows = {row[0]: row for row in workbook[GUIDE_SHEET].values if row[0] in config.FEATURE_COLUMNS}
    assert set(rows) == set(config.FEATURE_COLUMNS)
    for column, (_, required, meaning, allowed, example) in rows.items():
        assert required == ("yes" if column in REQUIRED_COLUMNS else "no")
        assert meaning and allowed and example
    assert "Steady = dose unchanged" in rows["insulin"][2]
    assert "Ch (yes, changed)" in rows["change"][3]
    assert "1 - Emergency" in rows["admission_type_id"][3]


def test_field_guide_starts_with_instructions(workbook):
    lines = [row[0] for row in workbook[GUIDE_SHEET].values if row[0]]
    assert any("Save the 'Sample data' sheet as CSV" in line for line in lines)
    assert any("Do not enter real patient information" in line for line in lines)


def test_the_data_sheet_saved_as_csv_is_accepted_by_the_app(workbook, model_dir):
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(workbook[DATA_SHEET].values)
    client = TestClient(create_app(model_dir), raise_server_exceptions=False)
    response = client.post("/api/predict-file", content=buffer.getvalue().encode(),
                           headers={"Content-Type": "text/csv"})
    assert response.status_code == 200 and len(response.json()["results"]) == 3


def test_workbook_download(model_dir):
    client = TestClient(create_app(model_dir), raise_server_exceptions=False)
    response = client.get("/sample.xlsx")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument")
    assert 'filename="sample_encounters.xlsx"' in response.headers["content-disposition"]
    assert load_workbook(io.BytesIO(response.content)).sheetnames == [DATA_SHEET, GUIDE_SHEET]
