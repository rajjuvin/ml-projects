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
- Candidates: logistic regression, decision tree, random forest, XGBoost, MLP. Hyperparameters were tuned with grouped cross-validation, refitting on PR-AUC.
- Decision threshold: chosen to maximize F1 on out-of-fold training predictions (never on the test set).

## Evaluation
Metrics are computed on out-of-fold predictions (training data) for model comparison, and on the held-out test set once for the selected model. See the README for the comparison table. Because only about 11% of encounters are positive, accuracy is reported but not used for selection: predicting "not readmitted" for everyone already scores about 0.886.

**Held-out test results:** pending (to be run once, after Phase 2 review).

## Limitations and risks
- **Modest predictive power.** Best ROC-AUC is about 0.68. The model ranks patients better than chance but misses many readmissions and flags many who are not readmitted.
- **Old data.** Encounters are from 1999-2008. Practice patterns, coding (ICD-9) and populations have changed.
- **Readmissions are only those recorded in the same data source.** Returns to other hospitals are not captured.
- **Fairness not yet assessed.** Race, gender and age are model inputs. Performance by subgroup has not been evaluated and should be before any real-world consideration.
- **Probabilities are uncalibrated.** Scores should be read as a risk ranking, not an exact probability, unless calibration is added.
- **Association, not causation.** Feature patterns do not imply that changing a feature changes the outcome.

## References
- Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, Article 781670. https://doi.org/10.1155/2014/781670
