
# HOUSE_PRICE_PROJECT
🏠 California Housing Price Prediction

Predicting median district house values from noisy, real-world-style housing data

`Python` `scikit-learn` `LightGBM` `Pandas` `Jupyter` `License: MIT`

**Best Model: LightGBM Regressor | Validation RMSE: 0.4791 | Validation R²: —**

---

## 📌 Overview

This project predicts `TargetPrice` — the median house value (in units of $100,000) for a California district property cluster — using demographic and structural features. It's framed as a regression problem: given income, room counts, occupancy, and location data for a district, predict its median home value.

The dataset mirrors real-world PropTech data challenges on purpose: it injects Gaussian noise, introduces missing values in property age records, and obfuscates feature names. The pipeline combines deep exploratory analysis, ratio/density feature engineering, and a tuned LightGBM regressor, finishing with a bounded, submission-ready prediction file.

---

## 🗂️ Repository Structure

```
HOUSE_PRICE_PROJECT/
├── Models/               # preprocessing_and_model_pipeline
├── notebook/             # Main EDA & modeling notebook
├── reports/              # Generated project report (PDF)
├── outputs/              # submission.csv and saved plots
└── README.md
```

---

## 📊 Dataset

The project uses district-level housing data, split into training and test sets:

| File | Description |
|---|---|
| `estate_train.csv` | Training features + actual `TargetPrice` |
| `estate_test.csv` | Test features only (for leaderboard predictions) |
| `estate_sample_submission.csv` | Submission format template (`PropertyID`, `TargetPrice`) |

**Columns:** `IncomeLevel`, `PropertyAge`, `TotalRooms`, `TotalBedrooms`, `NeighborhoodPop`, `AvgOccupancy`, `RoomsPerHousehold`, `BedroomsRatio`, `Latitude`, `Longitude` → `TargetPrice`

---

## 🔧 Pipeline

| Step | Description |
|---|---|
| 1. Data Loading | Load train/test sets; check shapes, dtypes, duplicates, and missing values. |
| 2. EDA | Target distribution (raw vs. log), missingness pattern analysis, correlation heatmap, outlier boxplots, feature-vs-target scatter plots. |
| 3. Feature Engineering | 5 engineered features — missingness flag, population/occupancy ratios, income-per-room, log-transformed population. |
| 4. Preprocessing | Median imputation for `PropertyAge`, IQR-based outlier capping, RobustScaler, log-transformed target. |
| 5. Model Training | LightGBM Regressor trained on the processed feature set. |
| 6. Cross-Validation | 3-fold CV, RMSE scored on the original (non-log) price scale. |
| 7. Model Interpretation | Feature importance ranking and residual analysis. |
| 8. Prediction | Final predictions inverse-transformed, clipped to a realistic non-negative range. |
| 9. Submission | Final predictions written to `submission.csv`. |

---

## 🧠 Model & Results

| Model | Notes |
|---|---|
| LightGBM Regressor ⭐ | 300 estimators, max depth 6, learning rate 0.05 — best performer |

**Best model:** LightGBM Regressor

| Metric | Score |
|---|---|
| Mean CV RMSE | 0.4791 (± 0.0069) |
| Fold 1 / 2 / 3 RMSE | 0.4798 / 0.4703 / 0.4872 |
| Residual Mean / Std | 0.0315 / 0.3886 |

---

## ✨ Key Features Engineered

- **Missingness signal:** `PropertyAge_missing` binary flag
- **Density ratios:** `PopPerHousehold`, `RoomsPerBedroom`
- **Interaction feature:** `IncomePerRoom` (income normalized by room count)
- **Skew correction:** `LogPop` (log-transformed neighborhood population)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/HOUSE_PRICE_PROJECT.git
cd HOUSE_PRICE_PROJECT

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm

# Launch the notebook
jupyter notebook notebook/
```

---

## 🛠️ Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · LightGBM

---

## 📁 Outputs

- `eda_plots.png` — exploratory data analysis visualizations
- `feature_importance.png` — LightGBM feature importance chart
- `submission.csv` — final `TargetPrice` predictions in the required format
- `California_Housing_Price_Prediction_Report.pdf` — full written project report

---

## 📈 Future Improvements

- Hyperparameter tuning via Bayesian search / Optuna
- Ensemble/stacking across LightGBM, XGBoost, and CatBoost
- Explicit geographic clustering from `Latitude`/`Longitude` as a categorical feature
- Iterative or KNN-based imputation for `PropertyAge` as an alternative to median imputation

---

## 📄 License

This project is available under the MIT License.

---

*Made for the Estate Price Prediction Challenge* 🏠
