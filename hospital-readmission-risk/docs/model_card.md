# Model card: 30-day readmission risk (diabetic inpatient encounters)

**Status:** retrospective, educational prototype. Not validated for clinical use.

## Intended use
- **Users:** care-management and discharge-planning teams (simulated), and healthcare leaders reviewing how such a tool would behave.
- **Decision supported (simulated):** prioritizing which discharged diabetes patients get further human review or follow-up planning, when follow-up capacity is limited.
- **Prediction:** the probability that an inpatient encounter is followed by a readmission in under 30 days, using information assumed available at discharge.

## Not intended for
- Recommending medications, doses or treatments, or estimating treatment effects.
- Deciding who receives or is denied care, or replacing clinical judgement.
- Any claim that follow-up prevents readmission.
- Use on populations, years or coding systems other than the training data (see Limitations).

## Data
- UCI "Diabetes 130-US Hospitals for Years 1999-2008" (Clore et al., 2014; CC BY 4.0): 101,766 inpatient encounters from 130 US hospitals where diabetes was a recorded diagnosis, with laboratory tests and medications during a stay of 1-14 days.
- **Label:** 1 if `readmitted == "<30"`, otherwise 0 (about 11.4% positive after filtering).
- **Exclusions:** encounters ending in death or hospice discharge (discharge disposition ids 11, 13, 14, 19, 20, 21), because readmission is not a meaningful outcome there and the follow-up workflow differs.
- **Split:** 80/20 by `patient_nbr`, so no patient appears in both training and test data. Cross-validation is also grouped by patient (5 stratified folds).
- **Features:** 8 numeric utilization measures and 39 categorical fields (demographics, admission/discharge codes, diagnoses, lab flags, medications). Identifiers are not used. The misspelled source column `citoglipton` is corrected to `sitagliptin`.

## Model
- Preprocessing: `"?"` treated as missing and imputed (median for numerics, an explicit `not_recorded` category for categoricals), lab "None" values kept as `not_measured`, one-hot encoding with rare categories pooled.
- Candidates: logistic regression, decision tree, random forest, XGBoost, MLP. Hyperparameters were tuned with patient-grouped cross-validation, selecting on PR-AUC. The search spaces and winning values can be reproduced with `python -m readmission.tune` and are recorded in `reports/best_params.json`.
- Decision threshold: chosen to maximize F1 on out-of-fold training predictions (never on the test set).

## Evaluation
Metrics are computed on out-of-fold predictions (training data) for model comparison, and on the held-out test set once for the selected model. See the README for the comparison table. Because only about 11% of encounters are positive, accuracy is reported but not used for selection: predicting "not readmitted" for everyone already scores about 0.886. PR-AUC is computed as average precision (scikit-learn’s `average_precision_score`), which differs slightly from the trapezoidal area under the precision-recall curve.

**Held-out test results (selected model: XGBoost).** Scored once, on 19,802 encounters from patients not used in training (base rate 11.2%), with the decision threshold of 0.1405 fixed beforehand from out-of-fold training predictions:

| Metric | Out-of-fold (training) | Held-out test |
|---|---|---|
| ROC-AUC | 0.679 | 0.676 |
| PR-AUC (random guess = base rate) | 0.234 (0.114) | 0.236 (0.112) |
| Accuracy | 0.746 | 0.746 |
| Precision | 0.215 | 0.210 |
| Recall | 0.462 | 0.459 |
| F1 | 0.294 | 0.288 |
| Share of encounters flagged | 24.5% | 24.4% |
| Precision in the top 10% | 0.277 | 0.277 |
| Lift in the top 10% | 2.42 | 2.48 |

The test results agree closely with the cross-validated estimates, so the comparison and the threshold do not appear to be overfitted to the training data. At this threshold about 1 in 5 flagged encounters is a true 30-day readmission, and about 46% of readmissions are caught.

**Uncertainty, groups and calibration.** `python -m readmission.analyze` describes the same test set further and saves aggregate counts and rates to `reports/test_analysis.json`. It does not change the model or the threshold.

- *95% intervals* (1,000 resamples of whole patients, since one patient can have several encounters): ROC-AUC 0.676 (0.660 to 0.690), PR-AUC 0.236 (0.213 to 0.258), precision 0.210 (0.194 to 0.223), recall 0.459 (0.430 to 0.485), lift in the top 10% 2.48 (2.27 to 2.68).
- *By group* (encounters, with enough cases to report; other groups are in the file with counts only): the model flags older patients more often (30.8% of ages 80 to 90 against 18.1% of ages 50 to 60), and its ROC-AUC falls with age, from about 0.71 at 50 to 60 to 0.64 at 80 to 90 and 0.54 at 90 and over. Women are flagged somewhat more often than men (25.9% against 22.7%) with recall of 48.1% against 43.3%. Caucasian and African American patients have similar recall (45.9% and 47.2%) and precision (21.3% and 19.6%). Encounters with the race not recorded show much lower recall (23.7%) and flagged share (12.7%), from only 38 readmissions. These group figures have no intervals of their own and small groups are noisy, so they point to where to look, not to settled differences.
- *Calibration:* in each tenth of the test set, ordered by score, the average score is close to the observed readmission rate (highest tenth 0.280 against 0.277; overall 0.114 against 0.112), and the Brier score is 0.094 against 0.099 for always predicting the base rate. The scores are therefore approximately calibrated on this test set. That was checked on one split and in tenths only.

## Demonstration web app
A small web page (`python -m readmission.app`) lets a visitor enter one made-up encounter or upload a CSV of at most 10 rows, and shows a risk score and whether the encounter is flagged. The score is a ranking score, not a calibrated probability, and the page says so. The app scores requests in memory, stores nothing, writes no log of what is submitted (it logs only the kind of an unexpected error and keeps in-memory totals of encounters scored, flagged and rejected, with nothing per visitor), and asks users not to enter real patient information. If the app is hosted, the hosting platform keeps its own request log (time, address of the visitor's connection, browser and result code, never the submitted values), which the app cannot switch off. It is a demonstration, subject to every limitation below. The form offers Female, Male, Non-binary, Other and Not declared for gender, which is broader than the training data: only 3 of the 99,343 eligible encounters carried a gender other than Female or Male (recorded as Unknown/Invalid), so the model has learned nothing about Non-binary, Other or Not declared and scores all three like that rare category, that is, like an unknown gender. The secondary and additional diagnosis codes are optional because a stay can have only one or two diagnoses; a blank is treated as not recorded, as in the training data.

## Limitations and risks
- **Modest predictive power.** Best ROC-AUC is about 0.68. The model ranks patients better than chance but misses many readmissions and flags many who are not readmitted.
- **Old data.** Encounters are from 1999-2008. Practice patterns, coding (ICD-9) and populations have changed.
- **Readmissions are only those recorded in the same data source.** Returns to other hospitals are not captured.
- **Fairness only partly assessed.** Race, gender and age are model inputs. Performance and flag rates by group were measured once on the test set (see above) and differ: for example by age, and for encounters with no recorded race. The differences have no intervals, no cause has been investigated, and no adjustment was made. This is not a fairness audit, and it should be repeated and understood before any real-world consideration.
- **Scores are approximately calibrated, but the page does not promise it.** On the held-out test set the average score matches the observed rate in each tenth. This was checked on one split, so the page still presents the score as a ranking score, not an exact chance, and it may be less accurate for other groups, years or hospitals.
- **Association, not causation.** Feature patterns do not imply that changing a feature changes the outcome.

## References
- Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, Article 781670. https://doi.org/10.1155/2014/781670
