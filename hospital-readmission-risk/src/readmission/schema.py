"""The model's inputs: allowed values, labels and validation, shared by the web form and the API.

Allowed values for the categorical fields are read from the trained model itself (the categories its
one-hot encoder saw), so the form and the model cannot drift apart. Error messages name the column but do
not repeat a value the user entered (an unrecognized field name is shown back, shortened to 40 characters).
"""

import json
import re
from importlib import resources

from readmission import config

MAX_ROWS = 10
MAX_UPLOAD_BYTES = 50_000
MAX_JSON_BYTES = 10_000
DEFAULT_MEDICATION_LEVEL = "No"

LABELS = {
    "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures (not including lab tests)",
    "num_medications": "Number of distinct medications given",
    "number_outpatient": "Outpatient visits in the year before",
    "number_emergency": "Emergency visits in the year before",
    "number_inpatient": "Inpatient stays in the year before",
    "number_diagnoses": "Number of diagnoses entered",
    "race": "Race",
    "gender": "Gender",
    "age": "Age group",
    "weight": "Weight range (lb)",
    "admission_type_id": "Admission type",
    "discharge_disposition_id": "Discharge destination",
    "admission_source_id": "Admission source",
    "payer_code": "Payer code",
    "medical_specialty": "Admitting physician specialty",
    "diag_1": "Primary diagnosis, ICD-9 code",
    "diag_2": "Secondary diagnosis, ICD-9 code",
    "diag_3": "Additional diagnosis, ICD-9 code",
    "max_glu_serum": "Glucose serum test result",
    "A1Cresult": "HbA1c test result",
    "change": "Diabetes medication changed during the stay?",
    "diabetesMed": "Any diabetes medication prescribed?",
}
# One line of guidance shown under each field: a question or description of what to enter.
HINTS = {
    "time_in_hospital": "How many days was the patient in hospital for this stay?",
    "num_lab_procedures": "How many lab tests were performed during the stay?",
    "num_procedures": "How many procedures other than lab tests (for example surgeries or scans) were performed?",
    "num_medications": "How many different medications were given during the stay?",
    "number_outpatient": "How many outpatient visits did the patient have in the year before this stay?",
    "number_emergency": "How many emergency visits did the patient have in the year before this stay?",
    "number_inpatient": "How many times was the patient admitted as an inpatient in the year before this stay?",
    "number_diagnoses": "How many diagnoses were entered for the stay?",
    "race": "Race as recorded. Choose Not recorded if it is unknown.",
    "gender": "Gender as recorded. The training data has no records for Non-binary, Other or Not declared, so these "
              "are scored like an unknown gender, not on any pattern of their own. The model was trained on public "
              "data to demonstrate AI/ML learning.",
    "age": "Age group of the patient at the time of the stay.",
    "weight": "Weight range in pounds. Choose Not recorded if unknown (most records have no weight).",
    "admission_type_id": "How was the patient admitted, for example as an emergency or an elective admission?",
    "discharge_disposition_id": "Where did the patient go after leaving the hospital?",
    "admission_source_id": "Where did the admission come from, for example the emergency room or a physician referral?",
    "payer_code": "Code for who pays for the care, as recorded. Choose Not recorded if unknown.",
    "medical_specialty": "Specialty of the admitting physician, as recorded. Choose Not recorded if unknown.",
    "diag_1": "What was the main reason for the stay? Enter its ICD-9 diagnosis code, for example 250.83 or 428.",
    "diag_2": "Secondary diagnosis as an ICD-9 code, for example 401. Leave blank if the stay had only one diagnosis.",
    "diag_3": "Additional diagnosis as an ICD-9 code, for example 250. Leave blank if the stay had fewer than three "
              "diagnoses.",
    "max_glu_serum": "Was a glucose serum test done, and what was the result? Choose Not measured if none was done.",
    "A1Cresult": "Was an HbA1c (average blood sugar) test done, and what was the result? "
                 "Choose Not measured if none was done.",
    "change": "Was any diabetes medication changed (the drug or its dose) during the stay? If a medication below is "
              "marked Up or Down, choose Yes, changed.",
    "diabetesMed": "Was any diabetes medication prescribed during the stay? Choose Yes only if you list at least "
                   "one medication below, and No only if you list none.",
}
MEDICATION_HINT = ("Every medication you do not list counts as No (not prescribed). Add a medication only if it was "
                   "prescribed: Steady = dose unchanged, Up = dose increased, Down = dose decreased.")

LEVEL_ORDER = ["No", "Steady", "Up", "Down"]
CODE_ALLOWED = "ICD-9 code: up to 8 letters, digits or dots (for example 250.83, 428, V57)"
OPTIONAL_CODE_ALLOWED = CODE_ALLOWED + ", or leave blank"
MEDICATION_MEANING = ("Was {drug} prescribed, and was its dose changed? No = not prescribed, Steady = dose "
                      "unchanged, Up = dose increased, Down = dose decreased. Optional: leave the column out for No.")
ALLOWED_TEXT = {"not_recorded": "not_recorded (or ?)", "not_measured": "not_measured (or None)"}

GENERIC_OPTION_LABELS = {"not_recorded": "Not recorded", "not_measured": "Not measured"}
COLUMN_OPTION_LABELS = {"change": {"Ch": "Yes, changed", "No": "No change"}}
# Written the way the source dataset writes them; accepted in uploaded files as well.
VALUE_ALIASES = {"?": "not_recorded"}
LAB_ALIASES = {"None": "not_measured"}

ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
CODE_COLUMNS = ["diag_1", "diag_2", "diag_3"]
CODE_PATTERN = re.compile(r"^[A-Z0-9.]{1,8}$")
CHOICE_COLUMNS = [col for col in config.CATEGORICAL_COLUMNS
                  if col not in ID_COLUMNS + CODE_COLUMNS + config.MEDICATION_COLUMNS]

# A stay may have only one or two diagnoses, so the secondary and additional codes can be blank ("?" is the
# dataset's marker for a missing value, and the model was trained with it).
OPTIONAL_CODE_COLUMNS = ["diag_2", "diag_3"]
MISSING_MARKER = "?"
OPTIONAL_COLUMNS = OPTIONAL_CODE_COLUMNS + config.MEDICATION_COLUMNS
REQUIRED_COLUMNS = [col for col in config.FEATURE_COLUMNS if col not in OPTIONAL_COLUMNS]

# Offered in the form and accepted in files. Female and Male are what the model learned; only 3 of the 99,343
# eligible encounters had another value, so Non-binary, Other and Not declared reach the model's rare-category bucket.
GENDER_OPTIONS = ["Female", "Male", "Non-binary", "Other", "Not declared"]

# A fictitious encounter for the "fill with an example" button.
EXAMPLE_ENCOUNTER = {
    "time_in_hospital": 4, "num_lab_procedures": 43, "num_procedures": 1, "num_medications": 15,
    "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1, "number_diagnoses": 8,
    "race": "Caucasian", "gender": "Female", "age": "[70-80)", "weight": "not_recorded",
    "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    "payer_code": "MC", "medical_specialty": "InternalMedicine",
    "diag_1": "428", "diag_2": "401", "diag_3": "250",
    "max_glu_serum": "not_measured", "A1Cresult": ">7", "change": "Ch", "diabetesMed": "Yes",
    "medications": {"insulin": "Up", "metformin": "Steady"},
}


def _code_labels():
    return json.loads(resources.files("readmission").joinpath("web/code_labels.json").read_text(encoding="utf-8"))


def allowed_values(pipeline):
    """{categorical column: values the trained model knows}, read from its one-hot encoder."""
    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["cat_encoder"]
    return {col: [str(value) for value in categories]
            for col, categories in zip(config.CATEGORICAL_COLUMNS, encoder.categories_, strict=True)}


def _to_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return None


def _is_blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _option_sort_key(column, value):
    if value in ("not_recorded", "not_measured"):
        return (0, 0, "")
    numbers = re.findall(r"\d+", value)
    if column in ID_COLUMNS or column == "weight":
        return (1, int(numbers[0]) if numbers else 10**6, value)
    return (1, 0, value)


class InputSchema:
    def __init__(self, allowed):
        allowed = {**allowed, "gender": GENDER_OPTIONS}
        self.allowed = {col: set(values) for col, values in allowed.items()}
        self._ordered = allowed
        self._code_labels = _code_labels()

    @classmethod
    def from_pipeline(cls, pipeline):
        return cls(allowed_values(pipeline))

    # ---- validation -------------------------------------------------------------------------------------

    def clean(self, raw):
        """Validate one encounter given as {column: value}; returns (row for the model, list of errors).

        Medication columns may be left out (meaning "No"), and so may the secondary and additional diagnosis
        (meaning not recorded). The old source spelling `citoglipton` is accepted."""
        errors = []
        renamed = {config.COLUMN_RENAMES.get(key, key): value for key, value in raw.items()}
        for key in renamed:
            if key not in config.FEATURE_COLUMNS:
                errors.append(f"unknown field '{str(key)[:40]}'")
        row = {}
        for col in config.NUMERIC_COLUMNS:
            low, high = config.NUMERIC_LIMITS[col]
            number = None if _is_blank(renamed.get(col)) else _to_int(renamed.get(col))
            if number is None or not low <= number <= high:
                errors.append(f"{col}: enter a whole number from {low} to {high}")
            else:
                row[col] = number
        for col in config.CATEGORICAL_COLUMNS:
            value = renamed.get(col)
            if col in config.MEDICATION_COLUMNS and _is_blank(value):
                value = DEFAULT_MEDICATION_LEVEL
            if col in OPTIONAL_CODE_COLUMNS and (_is_blank(value) or str(value).strip() == MISSING_MARKER):
                row[col] = MISSING_MARKER
                continue
            if _is_blank(value):
                errors.append(f"{col}: required")
                continue
            cleaned = self._clean_categorical(col, value)
            if cleaned is None:
                errors.append(f"{col}: not an allowed value")
            else:
                row[col] = cleaned
        self._check_medications(row, errors)
        return row, errors

    @staticmethod
    def _check_medications(row, errors):
        """Combinations that never occur in the training data: the two yes/no questions must agree with the drugs."""
        if "diabetesMed" not in row or any(col not in row for col in config.MEDICATION_COLUMNS):
            return
        levels = [row[col] for col in config.MEDICATION_COLUMNS]
        prescribed = any(level != DEFAULT_MEDICATION_LEVEL for level in levels)
        if row["diabetesMed"] == "Yes" and not prescribed:
            errors.append("diabetesMed: Yes needs at least one medication that is not No.")
        elif row["diabetesMed"] == "No" and prescribed:
            errors.append("diabetesMed: No cannot be combined with a medication that is not No.")
        if row.get("change") not in (None, "Ch") and any(level in ("Up", "Down") for level in levels):
            errors.append("change: a medication marked Up or Down needs the answer Yes, changed.")

    def _clean_categorical(self, col, value):
        if col in CODE_COLUMNS:
            text = str(value).strip().upper()
            return text if CODE_PATTERN.match(text) else None
        if col in ID_COLUMNS:
            number = _to_int(value)
            return number if number is not None and str(number) in self.allowed[col] else None
        text = str(value).strip()
        text = VALUE_ALIASES.get(text, text)
        if col in config.NOT_MEASURED_REPLACEMENTS:
            text = LAB_ALIASES.get(text, text)
        return text if text in self.allowed[col] else None

    # ---- description for the web page ---------------------------------------------------------------------

    def _option(self, col, value):
        if col in ID_COLUMNS:
            return {"value": value, "label": f"{value} - {self._code_labels[col].get(value, 'Other')}"}
        label = COLUMN_OPTION_LABELS.get(col, {}).get(value) or GENERIC_OPTION_LABELS.get(value, value)
        return {"value": value, "label": label}

    def _options(self, col):
        values = self._ordered[col]
        if col != "gender":  # gender keeps its listed order
            values = sorted(values, key=lambda value: _option_sort_key(col, value))
        return [self._option(col, value) for value in values]

    def _allowed_text(self, col):
        if col in ID_COLUMNS:
            return ", ".join(option["label"] for option in self._options(col))
        parts = []
        for option in self._options(col):
            value, label = option["value"], COLUMN_OPTION_LABELS.get(col, {}).get(option["value"])
            parts.append(f"{value} ({label.lower()})" if label else ALLOWED_TEXT.get(value, value))
        return ", ".join(parts)

    def guide(self, group_medications=False):
        """One row per input column, for the upload directions and the workbook's Field guide sheet.

        With `group_medications` the drug columns that can vary share a single row."""
        example = {key: value for key, value in EXAMPLE_ENCOUNTER.items() if key != "medications"}
        example.update(EXAMPLE_ENCOUNTER["medications"])
        rows, medications_done = [], False
        for col in config.FEATURE_COLUMNS:
            if col in config.MEDICATION_COLUMNS:
                if group_medications:
                    if not medications_done:
                        names = [c for c in config.MEDICATION_COLUMNS if len(self._ordered[c]) > 1]
                        rows.append({"column": ", ".join(names), "required": False, "example": "No",
                                     "meaning": MEDICATION_MEANING.format(drug="each drug"),
                                     "allowed": ", ".join(LEVEL_ORDER) + " (some drugs allow only some of these)"})
                        medications_done = True
                    continue
                levels = sorted(self._ordered[col], key=LEVEL_ORDER.index)
                rows.append({"column": col, "required": False, "example": example.get(col, DEFAULT_MEDICATION_LEVEL),
                             "meaning": MEDICATION_MEANING.format(drug=col), "allowed": ", ".join(levels)})
                continue
            if col in config.NUMERIC_COLUMNS:
                low, high = config.NUMERIC_LIMITS[col]
                allowed = f"whole number from {low} to {high}"
            elif col in CODE_COLUMNS:
                allowed = OPTIONAL_CODE_ALLOWED if col in OPTIONAL_CODE_COLUMNS else CODE_ALLOWED
            else:
                allowed = self._allowed_text(col)
            rows.append({"column": col, "required": col in REQUIRED_COLUMNS, "meaning": HINTS[col],
                         "allowed": allowed, "example": example[col]})
        return rows

    def describe(self):
        """Everything the web page needs to draw the form and the upload directions."""
        medications = [{"name": col, "levels": sorted(self._ordered[col])}
                       for col in config.MEDICATION_COLUMNS if len(self._ordered[col]) > 1]
        return {
            "numeric": [{"name": col, "label": LABELS[col], "hint": HINTS[col], "min": config.NUMERIC_LIMITS[col][0],
                         "max": config.NUMERIC_LIMITS[col][1]} for col in config.NUMERIC_COLUMNS],
            "choices": [{"name": col, "label": LABELS[col], "hint": HINTS[col], "options": self._options(col)}
                        for col in ID_COLUMNS + CHOICE_COLUMNS],
            "codes": [{"name": col, "label": LABELS[col], "hint": HINTS[col], "optional": col in OPTIONAL_CODE_COLUMNS}
                      for col in CODE_COLUMNS],
            "medications": {"default": DEFAULT_MEDICATION_LEVEL, "hint": MEDICATION_HINT, "options": medications},
            "guide": self.guide(group_medications=True),
            "limits": {"max_rows": MAX_ROWS, "max_upload_bytes": MAX_UPLOAD_BYTES},
            "required_columns": REQUIRED_COLUMNS,
            "optional_columns": OPTIONAL_COLUMNS,
            "example": EXAMPLE_ENCOUNTER,
        }
