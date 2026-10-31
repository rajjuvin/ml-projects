"""Load the saved model and turn validated encounters into risk scores and plain-language results."""

import json
from pathlib import Path

import joblib
import pandas as pd

from readmission import config
from readmission.schema import InputSchema
from readmission.train import METADATA_FILE, MODEL_FILE

# Shown at the top of every page and returned with every API response.
DECLARATIONS = [
    "Educational prototype built on 1999-2008 hospital data. It is not a clinical tool and must not be used to "
    "make decisions about anyone's care.",
    "Do not enter real patient information. Use made-up values only.",
    "Entries are scored in memory and are not stored or logged.",
]
SCORE_NOTE = ("The risk score ranks encounters by estimated risk of readmission within 30 days. It is not a "
              "calibrated probability, so do not read it as an exact chance.")


class Predictor:
    def __init__(self, pipeline, metadata):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])
        self.schema = InputSchema.from_pipeline(pipeline)

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
        return cls(joblib.load(model_dir / MODEL_FILE), metadata)

    def score(self, rows):
        """Risk scores for validated rows (dicts holding every model column)."""
        frame = pd.DataFrame(rows, columns=config.FEATURE_COLUMNS)
        return [float(score) for score in self.pipeline.predict_proba(frame)[:, 1]]

    def result(self, row_number, score):
        flagged = score >= self.threshold
        verdict = "at or above" if flagged else "below"
        return {
            "row": row_number,
            "risk_score": round(score, 3),
            "flagged": flagged,
            "message": (f"{'Flagged for follow-up review' if flagged else 'Not flagged'}: the risk score is "
                        f"{verdict} the cutoff of {self.threshold:.2f}."),
        }

    def notes(self):
        notes = [SCORE_NOTE]
        test = self.metadata.get("test_metrics")
        if test:
            notes.append(f"On held-out historical data, about {test['Precision@Threshold']:.0%} of flagged "
                         f"encounters were readmitted within 30 days, and {test['Recall@Threshold']:.0%} of all "
                         "readmissions were flagged.")
        return notes

    def summary(self):
        return {"model": self.metadata["model_name"], "threshold": round(self.threshold, 4)}
