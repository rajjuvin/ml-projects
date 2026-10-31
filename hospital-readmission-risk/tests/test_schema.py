import pytest

from readmission import config
from readmission.predict import Predictor
from readmission.schema import EXAMPLE_ENCOUNTER, REQUIRED_COLUMNS, InputSchema


@pytest.fixture(scope="module")
def schema(model_dir):
    return Predictor.load(model_dir).schema


def example_row():
    values = {key: value for key, value in EXAMPLE_ENCOUNTER.items() if key != "medications"}
    return {**values, **EXAMPLE_ENCOUNTER["medications"]}


def test_the_example_encounter_is_valid(schema):
    row, errors = schema.clean(example_row())
    assert errors == []
    assert set(row) == set(config.FEATURE_COLUMNS)  # medications left out are filled in as "No"
    assert row["acarbose"] == "No"


def test_text_numbers_ids_and_markers_from_a_csv_are_accepted(schema):
    raw = {key: str(value) for key, value in example_row().items()}
    raw.update({"weight": "?", "max_glu_serum": "None", "diag_1": " v57 ", "citoglipton": "No"})
    row, errors = schema.clean(raw)
    assert errors == []
    assert row["weight"] == "not_recorded" and row["max_glu_serum"] == "not_measured"
    assert row["diag_1"] == "V57"
    assert row["admission_type_id"] == 1 and row["time_in_hospital"] == 4
    assert row["sitagliptin"] == "No"  # the source's misspelled column name is accepted


@pytest.mark.parametrize("field, value", [
    ("time_in_hospital", 0), ("time_in_hospital", 99), ("time_in_hospital", "abc"), ("time_in_hospital", True),
    ("time_in_hospital", 2.5), ("age", "[999-1000)"), ("gender", "Robot"), ("admission_type_id", 99),
    ("diag_1", "not a code!"), ("insulin", "Sideways"), ("A1Cresult", "maybe"),
])
def test_bad_values_are_rejected_without_repeating_them(schema, field, value):
    _, errors = schema.clean({**example_row(), field: value})
    assert len(errors) == 1 and errors[0].startswith(f"{field}:")
    if isinstance(value, str):
        assert value not in errors[0]


def test_missing_required_fields_are_reported_by_name(schema):
    values = example_row()
    del values["age"]
    values["race"] = ""
    _, errors = schema.clean(values)
    assert sorted(errors) == ["age: required", "race: required"]


def test_unknown_fields_are_reported_and_truncated(schema):
    _, errors = schema.clean({**example_row(), "patient_nbr": 5, "x" * 100: 1})
    assert "unknown field 'patient_nbr'" in errors
    assert f"unknown field '{'x' * 40}'" in errors


def test_optional_columns_are_the_medications_and_the_later_diagnoses():
    assert set(config.FEATURE_COLUMNS) - set(REQUIRED_COLUMNS) == set(config.MEDICATION_COLUMNS) | {"diag_2", "diag_3"}
    assert len(config.MEDICATION_COLUMNS) == 23 and len(REQUIRED_COLUMNS) == 22


@pytest.mark.parametrize("blank", ["", "  ", None, "?"])
def test_secondary_and_additional_diagnoses_may_be_blank(schema, blank):
    row, errors = schema.clean({**example_row(), "diag_2": blank, "diag_3": blank})
    assert errors == [] and row["diag_2"] == "?" and row["diag_3"] == "?"  # the dataset's marker for "not recorded"


def test_diagnoses_may_be_left_out_entirely(schema):
    values = example_row()
    del values["diag_2"], values["diag_3"]
    row, errors = schema.clean(values)
    assert errors == [] and row["diag_2"] == row["diag_3"] == "?"


def test_the_primary_diagnosis_is_still_required_and_codes_are_still_checked(schema):
    values = {**example_row(), "diag_1": "", "diag_2": "not a code!"}
    assert sorted(schema.clean(values)[1]) == ["diag_1: required", "diag_2: not an allowed value"]


def no_medications():
    return {key: value for key, value in example_row().items() if key not in ("insulin", "metformin")}


def test_prescribed_yes_needs_a_drug_and_prescribed_no_allows_none(schema):
    yes_without = {**no_medications(), "diabetesMed": "Yes", "change": "No"}
    assert schema.clean(yes_without)[1] == ["diabetesMed: Yes needs at least one medication that is not No."]
    no_with = {**no_medications(), "diabetesMed": "No", "change": "No", "glipizide": "Steady"}
    assert schema.clean(no_with)[1] == ["diabetesMed: No cannot be combined with a medication that is not No."]
    assert schema.clean({**no_medications(), "diabetesMed": "No", "change": "No"})[1] == []
    assert schema.clean({**no_medications(), "diabetesMed": "Yes", "change": "No", "glipizide": "Steady"})[1] == []


@pytest.mark.parametrize("level", ["Up", "Down"])
def test_a_dose_change_needs_change_to_be_yes(schema, level):
    values = {**example_row(), "insulin": level}
    assert schema.clean({**values, "change": "Ch"})[1] == []
    assert schema.clean({**values, "change": "No"})[1] == [
        "change: a medication marked Up or Down needs the answer Yes, changed."]


def test_changed_yes_is_allowed_without_a_dose_change(schema):
    # a drug can be switched without any dose going Up or Down, so this direction is not enforced
    values = {**no_medications(), "diabetesMed": "Yes", "change": "Ch", "insulin": "Steady"}
    assert schema.clean(values)[1] == []


def test_gender_offers_female_male_non_binary_other_and_not_declared(schema):
    options = [option["value"] for option in next(
        field for field in schema.describe()["choices"] if field["name"] == "gender")["options"]]
    assert options == ["Female", "Male", "Non-binary", "Other", "Not declared"]
    for value in options:
        assert schema.clean({**example_row(), "gender": value})[1] == []
    assert schema.clean({**example_row(), "gender": "Unknown/Invalid"})[1] == ["gender: not an allowed value"]


def test_description_orders_and_labels_options(schema):
    described = schema.describe()
    choices = {field["name"]: field["options"] for field in described["choices"]}
    assert choices["weight"][0]["label"] == "Not recorded"
    assert choices["admission_type_id"][0]["label"] == "1 - Emergency"
    assert {option["value"]: option["label"] for option in choices["change"]} == {
        "Ch": "Yes, changed", "No": "No change"}
    assert described["limits"] == {"max_rows": 10, "max_upload_bytes": 50_000}
    assert all(len(med["levels"]) > 1 for med in described["medications"]["options"])


def test_every_field_has_guidance_text(schema):
    described = schema.describe()
    fields = described["numeric"] + described["choices"] + described["codes"]
    assert len(fields) == 24
    assert all(field["hint"].strip() for field in fields)
    assert described["medications"]["hint"]


def test_guide_has_one_row_per_column_or_one_for_all_medications(schema):
    full = schema.guide()
    assert [row["column"] for row in full] == config.FEATURE_COLUMNS
    grouped = schema.guide(group_medications=True)
    assert len(grouped) == len(config.FEATURE_COLUMNS) - len(config.MEDICATION_COLUMNS) + 1
    assert all(row["meaning"] and row["allowed"] for row in full + grouped)
    assert schema.describe()["guide"] == grouped


def test_allowed_values_come_from_the_model(schema, model_dir):
    assert isinstance(schema, InputSchema)
    assert "Caucasian" in schema.allowed["race"]
