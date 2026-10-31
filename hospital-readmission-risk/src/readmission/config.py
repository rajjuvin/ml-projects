"""Project-wide constants: data schema, split settings and the eligibility filter."""

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

ID_COLUMNS = ["encounter_id", "patient_nbr"]
GROUP_COLUMN = "patient_nbr"
SOURCE_TARGET_COLUMN = "readmitted"
POSITIVE_LABEL = "<30"

# Discharge dispositions excluded from the cohort (expired or hospice): readmission is not a meaningful
# outcome and the follow-up workflow differs (Strack et al., 2014).
# 11 Expired; 13 Hospice/home; 14 Hospice/medical facility; 19-21 Expired (various).
EXCLUDED_DISCHARGE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]

# The source file misspells this medication; it is corrected on load.
COLUMN_RENAMES = {"citoglipton": "sitagliptin"}

NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "sitagliptin",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Columns where the literal string "None" means "test not performed" (not a missing value).
NOT_MEASURED_REPLACEMENTS = {
    "max_glu_serum": {"None": "not_measured"},
    "A1Cresult": {"None": "not_measured"},
}
