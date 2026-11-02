# Hospital Readmission Risk Prediction

Estimates the probability that a diabetic inpatient encounter is followed by a hospital readmission **in under 30 days**, using information assumed available at discharge. The demonstration simulates helping a care-coordination team decide which discharged patients to prioritize for follow-up review.

> **Retrospective engineering prototype, not a clinically validated tool.** It uses 1999-2008 data, does not recommend treatment, and must not be used to decide who receives care. See the [model card](docs/model_card.md), and the [project report](docs/PROJECT_REPORT.md) for the whole story from question to live service.

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

These agree closely with the cross-validated figures above (ROC-AUC 0.679, PR-AUC 0.234), so the results do not appear to be overfitted to the training data. At the chosen threshold the model flags about 24% of encounters; roughly 1 in 5 of those is readmitted within 30 days, and about 46% of all readmissions are caught. With 95% intervals from resampling patients, ROC-AUC is 0.676 (0.660 to 0.690), precision 0.210 (0.194 to 0.223) and recall 0.459 (0.430 to 0.485). In each tenth of the test set the average score is close to the observed readmission rate, so the scores are approximately calibrated on this split. Performance does differ between groups, for example by age, and the model card lists them. `python -m readmission.analyze` reproduces all of this. Full details are in the [model card](docs/model_card.md).

### What the numbers say
- XGBoost is the strongest model, but the differences are small: its lead over logistic regression is about 0.016 PR-AUC. The available data carries limited predictive signal.
- A default 0.5 probability cutoff is useless with this class balance (the tuned Random Forest never exceeds a predicted probability of 0.3, giving a precision of 0), which is why thresholds are chosen from out-of-fold predictions.

## Project layout

```
src/readmission/    config, data loading and splitting, preprocessing, models, evaluation, tuning, training and comparison CLIs,
                    and the web app (schema, predict, app, samples, web/ page and sample files)
tests/              unit tests on synthetic data (no dataset needed)
notebooks/          exploration, model tuning and comparison (hospreadmissionpred.ipynb)
scripts/            download_data.py
docs/               model card, project report and its figures (docs/figures, drawn by scripts/make_figures.py)
reports/            saved comparison table, tuned hyperparameters (best_params.json) and the test-set analysis (test_analysis.json)
data/               the downloaded dataset (not committed)
models/             the trained model and its metadata (readmission_model.joblib, metadata.json), committed so a release is
                    one exact code-and-model pair; `readmission.train` recreates them
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
python -m readmission.analyze                     # test-set intervals, results by group and calibration, saved to reports/test_analysis.json (about 30 seconds)
```

`train` and `compare` read the tuned values from `reports/best_params.json`. To try one model without waiting for all five, use `python -m readmission.compare --models decision_tree` (prints the result and does not overwrite the saved table), or `python -m readmission.tune --models decision_tree` to re-tune just that model.

`python -m readmission.train --evaluate-test` also scores the held-out test set. Run it once, after the model and threshold are final.

## Web app

A small web page that scores encounters with the trained model. It needs the trained model files in `models/`, which are included in the repository (`readmission.train` recreates them). A `.joblib` file is a Python pickle, so only load model files you trained yourself or took from this repository.

```bash
python -m readmission.app        # then open http://127.0.0.1:8000   (or the readmission-serve command)
```

- **Enter values:** a form with one field per model input, each with a short question or description. Diabetes drugs start as "No"; add a medication row only for drugs that were prescribed. The secondary and additional diagnosis codes are optional. The medication answers must agree with the listed drugs (for example, "prescribed = Yes" needs at least one drug, and a drug marked Up or Down needs "changed = Yes"), because the model never saw contradictory combinations.
- **Upload a file:** a CSV with one row per encounter, at most 10 rows and 50 KB. The page links a sample as a plain CSV or as an Excel workbook (a "Sample data" sheet plus a "Field guide" sheet that explains every column), and lists every column with its meaning and allowed values.
- **Result:** for each row, only the row number, a risk score, whether it is flagged (at or above the model's cutoff) and a short explanation. Nothing that was entered is shown again, and the form and file picker are cleared after every submission.
- **Privacy by design:** requests are scored in memory. Nothing is written to disk and no request data is logged, and responses are not cacheable. Error messages name the problem field or row but do not repeat a value that was entered, with one exception: if text is entered where a column heading is expected (in an uploaded file), or where a field or medication name is expected (in a direct API request), and it is not recognized, it is shown back shortened to 40 characters so it can be corrected. This can include an actual value, for example when a file has no heading row. None of this is stored or logged. The page asks users not to enter real patient information. The tests check these properties.
- **API:** `POST /api/predict` (JSON for one encounter), `POST /api/predict-file` (CSV text), `GET /api/health`, `GET /api/schema` and `GET /api/stats`. The page presents the risk score as a ranking score, not an exact chance; on the held-out test set the scores were close to the observed rates (see the model card), but that was checked on one split only.
- **Usage totals:** `GET /api/stats` shows running totals since the instance started: encounters scored, how many were flagged, inputs rejected and unexpected errors. They are counts only (nothing about any single request is kept) and start again from zero on a restart.
- **Notice mode:** set the environment variable `READMISSION_NOTICE` to a sentence and the page shows it instead of the form, while scoring answers 503 with it. `GET /api/health` stays healthy and reports `"mode": "notice"`. Unset the variable to return to normal.
- **Logging:** an unexpected error is logged as one line naming only the kind of error (never its message, which could contain submitted values). Nothing else about a request is logged by the app. A hosting platform keeps its own request log; see the model card.

## Docker

The `Dockerfile` packages the web app together with the trained model (`models/readmission_model.joblib` and `models/metadata.json` are committed for this reason, so a release is one exact code-and-model pair). From this folder:

```bash
docker build -t readmission-app .
docker run --rm -p 8080:8080 readmission-app      # then open http://localhost:8080
```

- The image is Linux-based and about 0.7 GB. The server runs as an ordinary user (not root) and listens on the port named by the `PORT` variable (8080 by default).
- The libraries are installed at the pinned versions in `requirements.txt`, with `xgboost-cpu` (the same release without the large GPU libraries) in place of `xgboost`. To build with the standard package instead, add `--build-arg XGBOOST_PACKAGE=xgboost`; the app itself would also need a change to ask XGBoost for a GPU, since the model is set up for the CPU.
- `GET /api/health` returns `status`, the model name and cutoff, the app version, when the model was created and the mode (`normal` or `notice`). The image's health check calls it.
- `.dockerignore` is an allow-list: only the code, the pinned requirements and the two model files enter the image, never the dataset, notes or tests.

## Live demo

A copy of the web app runs on Google Cloud Run: <https://readmission-app-364492331058.us-central1.run.app>

- It is a learning demonstration, subject to every limitation in the [model card](docs/model_card.md). Do not enter real patient information.
- It runs on a small budget and a free trial, so it may be paused or taken down at any time. It scales to zero when nobody is using it, so the first visit after a quiet period (about 15 minutes) takes roughly 5 to 8 seconds while the container starts; later requests answer in a fraction of a second.
- The app stores nothing and logs nothing about what is submitted. The hosting platform keeps its own request log (time, the visitor's IP address, browser and result code), which the app cannot switch off.
- `GET /api/health` shows which release is live (`app_version`) and whether it is in notice mode; `GET /api/stats` shows running totals since the current instance started.

### Release, deploy and roll back

A release is a version tag. The image is published to `ghcr.io/rajjuvin/readmission-app:<version>`, and the owner deploys it by hand.

1. Change `__version__` in `src/readmission/__init__.py` (the only place it is written), then commit and push. Wait for the tests to pass.
2. Tag and push: `git tag -a vX.Y.Z -m "Release X.Y.Z"` then `git push origin vX.Y.Z`. The release workflow checks that the tag matches the version, builds the image, starts it, scores a sample file and only then publishes it.
3. Deploy the new image as a named revision, then check `/api/health`:

```bash
gcloud run deploy readmission-app --image ghcr.io/rajjuvin/readmission-app:X.Y.Z --region us-central1 --memory 512Mi --cpu 1 --min-instances 0 --max-instances 1 --timeout 30 --revision-suffix vX-Y-Z
```

Every deploy or setting change keeps the earlier revisions, so a rollback moves traffic rather than rebuilding anything:

```bash
gcloud run services update-traffic readmission-app --region us-central1 --to-revisions readmission-app-vA-B-C=100   # back to an earlier revision
gcloud run services update-traffic readmission-app --region us-central1 --to-latest                                   # forward again
```

To pause scoring with a message, set `READMISSION_NOTICE` (see Web app) and remove it afterwards; to take the app offline at once, remove the public-access permission:

```bash
gcloud run services update readmission-app --region us-central1 --update-env-vars "READMISSION_NOTICE=The demo is paused for maintenance." --revision-suffix vX-Y-Z-paused-1
gcloud run services update readmission-app --region us-central1 --remove-env-vars READMISSION_NOTICE --revision-suffix vX-Y-Z-live-2
gcloud run services remove-iam-policy-binding readmission-app --region us-central1 --member=allUsers --role=roles/run.invoker
```

The rollback and both off switches were each tried once on the live service.

## Method summary

- **Cohort:** inpatient encounters with a diabetes diagnosis, excluding death and hospice discharges. 101,766 encounters in the source data.
- **Split:** 80/20 by patient, so no patient appears in both train and test. Cross-validation is grouped by patient.
- **Preprocessing:** missing values (`?`) imputed; lab results "None" kept as an explicit `not_measured` category; one-hot encoding with rare categories pooled; numerics scaled only for logistic regression and the MLP.
- **Tuning:** exhaustive grid search (30 random candidates for XGBoost) with patient-grouped cross-validation, selecting on PR-AUC (average precision). `python -m readmission.tune` runs it and saves the winners, with their CV scores and the data fingerprint, to `reports/best_params.json`.
- **Threshold:** maximizes F1 on out-of-fold training predictions, never on the test set. For the current model it is 0.1405: an encounter scoring at or above it is flagged (about 24% of encounters). It is far below the usual 0.5 because only about 11% of encounters are readmitted.

## Roadmap

1. ✅ Model comparison (logistic regression, decision tree, random forest, XGBoost, neural network)
2. ✅ Reproducible package, tests and CI
3. ✅ Reproducible hyperparameter tuning script, with tuned values stored in a JSON file
4. ✅ Prediction API (FastAPI) and web page (form and CSV upload) with input validation and tests
5. ✅ Docker packaging, live deployment on Google Cloud Run, usage totals and a rollback procedure
6. ✅ Final write-up: the [project report](docs/PROJECT_REPORT.md), with intervals, results by group and calibration (`python -m readmission.analyze`)

## Data and attribution

This project uses the *Diabetes 130-US Hospitals for Years 1999-2008* dataset (Clore et al., 2014), provided through the UCI Machine Learning Repository and licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The associated study is described by Strack et al. (2014). The data is not redistributed here; see [data/README.md](data/README.md) for the download date, scope, changes made and how to download it. The descriptions of the admission and discharge codes shown in the web form (`src/readmission/web/code_labels.json`) are taken from the dataset's `IDS_mapping.csv`.

## References

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. BioMed Research International, 2014, Article 781670. https://doi.org/10.1155/2014/781670

## License

Code is released under the [MIT License](../LICENSE). The dataset is licensed separately under CC BY 4.0, as described above.

## Built with Claude Code

This project (the `hospital-readmission-risk/` folder and its CI workflow) was developed with [Claude Code](https://claude.com/claude-code), Anthropic's AI coding assistant, working with the repository owner. Each phase (model evaluation, packaging, tests, CI) was reviewed and approved by the owner, and the commits that add or change this project carry a `Co-Authored-By: Claude` trailer. Other projects in this repository are not covered by this note.
