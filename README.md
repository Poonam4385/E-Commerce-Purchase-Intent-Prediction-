# E-Commerce Purchase Intent Prediction using Explainable AI

An explainable machine-learning project that predicts whether an e-commerce visitor is likely to make a purchase based on browsing behavior.

## Features

- Exploratory data analysis
- Feature engineering
- Six-model comparison
- CatBoost final model
- Threshold optimization
- Probability calibration
- SHAP and LIME explanations
- PageValues ablation analysis
- FastAPI prediction service
- Streamlit dashboard

## Dataset

The dataset contains about 12,000 browsing sessions with 17 input features and one target:

```text
Converted
0 = No Purchase
1 = Purchase
```

The conversion rate is about 15.7%, so PR-AUC, recall, F1-score, and ROC-AUC are emphasized over accuracy alone.

## Final Model

CatBoost was selected using an independent validation set.

Final untouched-test results:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.970 |
| PR-AUC | 0.863 |
| Accuracy | 91.8% |
| Purchase Precision | 69.7% |
| Purchase Recall | 85.2% |
| Purchase F1-score | 76.6% |
| Brier Score | 0.051 |

The calibrated decision threshold is approximately:

```text
0.35
```

## Explainability

SHAP and LIME identified `PageValues` as the strongest purchase-intent feature.

The PageValues ablation test showed:

```text
With PageValues:
ROC-AUC ≈ 0.972
PR-AUC  ≈ 0.862

Without PageValues:
ROC-AUC ≈ 0.848
PR-AUC  ≈ 0.560
```

This indicates that PageValues provides strong predictive information but may be more useful for late-stage purchase prediction than early intervention.

## Saved Artifacts

```text
artifacts/
├── purchase_intent_model_improved.joblib
├── probability_calibrator.joblib
└── decision_threshold_improved.joblib
```

## Run Streamlit

```powershell
python -m streamlit run dashboard\app.py
```

Default address:

```text
http://localhost:8501
```

## Run FastAPI

```powershell
python -m uvicorn src.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Limitations and Improvements

The model mainly uses single-session behavior and does not include complete customer history, cart history, campaign exposure, or product preferences.

Future improvements include:

- Early-intent prediction without PageValues
- Customer-history features
- Temporal validation
- Model-drift monitoring
- Periodic retraining

## Conclusion

The final CatBoost model provides strong purchase-intent prediction with high ROC-AUC and PR-AUC while remaining interpretable through SHAP and LIME. The improved train-validation-test design and probability calibration make the final reported performance more reliable.