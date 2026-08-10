# SQL-Based Healthcare Analytics & AI Pipeline

Querying a local SQLite hospital database with SQL, exploring patient data, identifying clinical trends, and preparing a leak-free scikit-learn pipeline for classification documented end-to-end in a Jupyter notebook.

## Project Files

| File | Description |
|---|---|
| `hospital.db` | SQLite database  `Patients`, `Doctors`, `Admissions` tables, 51,208 rows each, joined via `PatientID` / `DoctorID` |
| `hospital_analytics.ipynb` | Full notebook: SQL extraction → EDA → clinical trend analysis → feature engineering → ML pipeline → baseline models |

## Environment

- Local: DB Browser for SQLite + Jupyter Notebook in VS Code
- Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

## Data Source & Cleaning

Built from three raw CSV exports (`Patients.csv`, `Doctors.csv`, `Admissions.csv`, 55,500 rows each). Data quality issues found and resolved before loading into SQLite:

- Only the first 1,000 rows of each file had `PatientID` / `DoctorID` / `AdmissionID` populated; the remaining ~50,208 real records had blank ID columns, and ~4,300 rows were fully blank padding.
- Verified all three files are row-aligned (same record = same row position across files), then reconstructed clean sequential surrogate keys for all 51,208 real records.
- Result: 0 nulls, 0 duplicate keys, full referential integrity between `Patients`, `Doctors`, and `Admissions`.

## Pipeline

1. **Connect & query (SQL)** `sqlite3` connection, schema inspection (`PRAGMA table_info`), core join query across all three tables, aggregate query for condition prevalence.
2. **Load into pandas** `pd.read_sql_query` into a working dataframe.
3. **Exploratory data analysis** structural checks, derived `LengthOfStay` from admission/discharge dates, numeric distributions and outlier detection (`Age`, `BillingAmount`, `LengthOfStay`), categorical frequency counts, target class balance.
4. **Clinical trend analysis** correlation matrix on numeric features, bivariate breakdowns (billing by admission type/condition, length of stay by admission type, age by test result), crosstabs across categorical fields, chi-square independence tests.
5. **Feature engineering** dropped identifier and high-cardinality columns (`PatientID`, `AdmissionID`, `DoctorName`, `Hospital` [37,146 unique values], `Room_Number`, raw date columns superseded by `LengthOfStay`); split remaining features into numeric vs. categorical.
6. **ML-ready pipeline** `LabelEncoder` on target (`TestResults`), stratified `train_test_split`, `ColumnTransformer` (`StandardScaler` on numeric, `OneHotEncoder` on categorical, fit on train only to prevent leakage).
7. **Baseline modeling** `LogisticRegression` and `RandomForestClassifier` trained on the processed features.

## Key Findings

- **No demographic/clinical association**: `MedicalCondition` and `TestResults` are statistically independent of `Gender`, `Blood_Type`, `AdmissionType`, and `InsuranceProvider` near-uniform distribution across all categories, confirmed with chi-square tests.
- **Baseline models confirm no signal**: Logistic Regression reached 34% accuracy predicting `TestResults` (3-class, balanced — chance level is ~33.3%). Random Forest was run as a non-linear check against the same result.
- **Conclusion**: This is a synthetic dataset with independently/randomly assigned categorical fields. Near-chance model performance is the correct, expected outcome it confirms the pipeline is leak-free and working correctly, not that the model is broken. The deliverable demonstrates correct SQL extraction, EDA methodology, and scikit-learn preprocessing practice on realistic (if clinically flat) healthcare data.

## How to Run

1. Open `hospital.db` in DB Browser for SQLite to explore tables directly, or
2. Open `hospital_analytics.ipynb` in VS Code / Jupyter, ensure `hospital.db` is in the same folder, and run cells top to bottom.