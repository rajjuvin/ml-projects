# Hospital Readmission Risk Prediction

Estimates the probability that a diabetic inpatient encounter is followed by a hospital readmission **in under 30 days**, using information assumed available at discharge. The demonstration simulates helping a care-coordination team decide which discharged patients to prioritize for follow-up review.

> **Retrospective engineering prototype, not a clinically validated tool.** It uses 1999-2008 data, does not recommend treatment, and must not be used to decide who receives care. See the [model card](docs/model_card.md).

## Results so far

Five models compared on the same patient-grouped, 5-fold cross-validation of the training data (out-of-fold predictions; the held-out test set played no part in this comparison). About 11.4% of encounters are positive, so the PR-AUC of a random guess is 0.114.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Precision in top 10% | Lift in top 10% |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.679** | **0.234** | 0.746 | 0.215 | 0.462 | **0.294** | **0.277** | **2.43** |
| Neural Network (MLP) | 0.671 | 0.222 | 0.731 | 0.205 | 0.470 | 0.286 | 0.261 | 2.28 |
| Logistic Regression | 0.666 | 0.218 | 0.754 | 0.212 | 0.421 | 0.282 | 0.264 | 2.31 |
| Random Forest | 0.667 | 0.214 | 0.736 | 0.204 | 0.452 | 0.282 | 0.261 | 2.29 |
| Decision Tree | 0.645 | 0.202 | 0.687 | 0.184 | 0.505 | 0.269 | 0.240 | 2.10 |

Note: PR-AUC is computed as average precision (scikit-learn’s `average_precision_score`), which differs slightly from the trapezoidal area under the precision-recall curve.

Accuracy, precision, recall and F1 are at each model's own decision threshold, chosen to maximize F1 on the out-of-fold predictions. Accuracy is shown for completeness but is not a useful yardstick here: predicting "not readmitted" for every encounter already scores about 0.886. The comparison relies on ranking metrics (ROC-AUC, PR-AUC) and on what a team with limited capacity would see: **flagging the 10% highest-risk encounters finds readmissions about 2.4 times as often as picking at random.**

Full table: [reports/oof_comparison.csv](reports/oof_comparison.csv).

### Held-out test results (XGBoost)

After the model, settings and threshold were fixed, XGBoost was scored once on the held-out test set: 19,802 encounters from patients not used in training.

| ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Flagged | Precision in top 10% | Lift in top 10% |
|---|---|---|---|---|---|---|---|---|
| 0.676 | 0.236 | 0.746 | 0.210 | 0.459 | 0.288 | 24.4% | 0.277 | 2.48 |

"Flagged" is the share of test encounters whose predicted probability is at or above the threshold (0.1405).

These agree closely with the cross-validated figures above (ROC-AUC 0.679, PR-AUC 0.234), so the results do not appear to be overfitted to the training data. At the chosen threshold the model flags about 24% of encounters; roughly 1 in 5 of those is readmitted within 30 days, and about 46% of all readmissions are caught. Full details are in the [model card](docs/model_card.md).

### What the numbers say
- XGBoost is the strongest model, but only moderately: the dataset carries limited predictive signal, and the gap to logistic regression is small (about 0.016 PR-AUC, roughly four times the fold-to-fold spread).
- A default 0.5 probability cutoff is useless with this class balance (the tuned Random Forest never exceeds a predicted probability of 0.3, giving a precision of 0), which is why thresholds are chosen from out-of-fold predictions.

## Project layout

```
src/readmission/    config, data loading and splitting, preprocessing, models, evaluation, tuning, training and comparison CLIs
tests/              unit tests on synthetic data (no dataset needed)
notebooks/          exploration, model tuning and comparison (hospreadmissionpred.ipynb)
scripts/            download_data.py
docs/               model card
reports/            saved comparison table and tuned hyperparameters (best_params.json)
data/, models/      downloaded data and trained artifacts (not committed)
```

## Quickstart

Requires Python 3.13. Dependency versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps

python scripts/download_data.py                   # fetches and verifies the UCI dataset into data/
pytest                                            # unit tests
python -m readmission.tune                        # optional: re-run the hyperparameter search, rewrites reports/best_params.json (about 20 minutes)
python -m readmission.compare                     # reproduce the comparison table (a few minutes)
python -m readmission.train --model xgboost       # trains, picks the threshold, saves models/readmission_model.joblib
```

`train` and `compare` read the tuned values from `reports/best_params.json`. To try one model without waiting for all five, use `python -m readmission.compare --models decision_tree` (prints the result and does not overwrite the saved table), or `python -m readmission.tune --models decision_tree` to re-tune just that model.

`python -m readmission.train --evaluate-test` also scores the held-out test set. Run it once, after the model and threshold are final.

## Method summary

- **Cohort:** inpatient encounters with a diabetes diagnosis, excluding death and hospice discharges. 101,766 encounters in the source data.
- **Split:** 80/20 by patient, so no patient appears in both train and test. Cross-validation is grouped by patient.
- **Preprocessing:** missing values (`?`) imputed; lab results "None" kept as an explicit `not_measured` category; one-hot encoding with rare categories pooled; numerics scaled only for logistic regression and the MLP.
- **Tuning:** exhaustive grid search (30 random candidates for XGBoost) with patient-grouped cross-validation, selecting on PR-AUC (average precision). `python -m readmission.tune` runs it and saves the winners, with their CV scores and the data fingerprint, to `reports/best_params.json`.
- **Threshold:** maximizes F1 on out-of-fold training predictions, never on the test set.

## Roadmap

1. ✅ Model comparison (logistic regression, decision tree, random forest, XGBoost, neural network)
2. ✅ Reproducible package, tests and CI
3. ✅ Reproducible hyperparameter tuning script, with tuned values stored in a JSON file
4. Prediction API (FastAPI) with input validation and tests
5. Docker packaging, live deployment, monitoring and rollback procedure
6. Demo page and final write-up

## Data and attribution

This project uses the *Diabetes 130-US Hospitals for Years 1999-2008* dataset (Clore et al., 2014), provided through the UCI Machine Learning Repository and licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The associated study is described by Strack et al. (2014). The data is not redistributed here; see [data/README.md](data/README.md) for the download date, scope, changes made and how to download it.

## References

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. BioMed Research International, 2014, Article 781670. https://doi.org/10.1155/2014/781670

## License

Code is released under the [MIT License](../LICENSE). The dataset is licensed separately under CC BY 4.0, as described above.

## Built with Claude Code

This project (the `hospital-readmission-risk/` folder and its CI workflow) was developed with [Claude Code](https://claude.com/claude-code), Anthropic's AI coding assistant, working with the repository owner. Each phase (model evaluation, packaging, tests, CI) was reviewed and approved by the owner, and the commits that add or change this project carry a `Co-Authored-By: Claude` trailer. Other projects in this repository are not covered by this note.
