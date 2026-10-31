# Data

The dataset is not stored in this repository. Download it with:

```bash
python scripts/download_data.py
```

This places `diabetic_data.csv` and `IDS_mapping.csv` here and verifies their SHA-256 checksums.

## Data source and attribution

This project uses the *Diabetes 130-US Hospitals for Years 1999-2008* dataset (Clore et al., 2014), provided through the UCI Machine Learning Repository. The associated study is described by Strack et al. (2014).

- **Dataset:** https://doi.org/10.24432/C5230J
- **Downloaded:** October 22, 2026
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), which allows sharing and adaptation for any purpose provided appropriate credit is given.
- **Scope:** 101,766 inpatient encounters from 130 US hospitals (1999-2008) where diabetes was one of the recorded diagnoses, the patient underwent laboratory tests and medications during the encounter, and the stay was 1-14 days.
- **Changes made:** the original files are used unchanged. Cleaning, exclusions and derived variables are applied in code (`src/readmission/data.py`, `src/readmission/features.py`) and documented in the notebook. The original study excluded encounters resulting in death or hospice discharge (Strack et al., 2014), and this project does the same. The misspelled column `citoglipton` is corrected to `sitagliptin`.

## References

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. BioMed Research International, 2014, Article 781670. https://doi.org/10.1155/2014/781670
