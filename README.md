# 📉 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

> **Which customers are about to leave the telecom operator — and why?**
> A binary classification project on 7,043 customers: data cleaning, EDA, a baseline, and a comparison of Logistic Regression and CatBoost, with a focus on the business trade-off between catching churners and false alarms.

---

## 🎯 Business problem

Acquiring a new customer costs much more than keeping an existing one.
If the company knows in advance **who is likely to churn**, it can offer them a discount or a better plan *before* they leave.

**Goal:** predict `Churn` (Yes / No) from customer data — contract type, services, tenure, and charges.

---

## 📊 Results

| Model | ROC-AUC | Recall | Accuracy |
|---|:---:|:---:|:---:|
| Baseline: "nobody churns" | 0.500 | 0.000 | 0.735 |
| Logistic Regression | 0.842 | 0.567 | 0.807 |
| CatBoost | 0.845 | 0.505 | 0.803 |
| **CatBoost (Balanced)** ✅ | **0.845** | **0.778** | 0.757 |

*Test set: 1,409 customers (20%, stratified). 5-fold cross-validation ROC-AUC: **0.845 ± 0.011**.*

### What these numbers mean

- 🪤 **Accuracy is misleading here.** Only ~26.5% of customers churn, so the useless "nobody churns" model already gets 73.5% accuracy while catching **zero** churners.
- ⚖️ **A complex model is not automatically better.** CatBoost beats Logistic Regression by only 0.003 ROC-AUC, which is smaller than the cross-validation spread (± 0.011), so the difference is noise. Further gains have to come from **better features, not a fancier model**.
- 🎯 **Recall matters most for the business.** With `auto_class_weights="Balanced"`, CatBoost finds **78% of churners instead of 51%**: it misses **83 customers instead of 185** (out of 374). The price is lower accuracy (0.80 → 0.76), meaning more false alarms. Losing a customer costs more than an unnecessary discount, so this trade-off is worth it.

---

## 🔍 Key insights from EDA

- 📄 **Contract type is the strongest signal.** Month-to-month customers churn far more often than customers on one- or two-year contracts.
- ⏳ **New customers are the risk group.** Most churn happens in the first months of tenure.
- 🧹 **Data quirk:** `TotalCharges` is stored as text. 11 customers have an empty value, all with `tenure = 0`: they had just joined and had not been billed yet, so their value was set to 0.

---

## 🛠️ Approach

```
Load data → Clean → EDA → Stratified train/test split
        → Baseline → Logistic Regression → CatBoost → CatBoost (Balanced)
        → Cross-validation → Feature importance → Error analysis
```

1. **Cleaning:** convert `TotalCharges` to numbers, encode `Churn` as 1/0, drop `customerID`.
2. **Baseline:** `DummyClassifier`, which always predicts "no churn", as the reference point.
3. **Logistic Regression:** one-hot encoding + `StandardScaler`.
4. **CatBoost:** works with categorical features directly via `cat_features`.
5. **Evaluation:** ROC-AUC as the main metric, recall for the business, and 5-fold cross-validation to check stability.
6. **Error analysis:** inspect the churners the model missed to get ideas for new features.

---

## 📁 Project structure

```
telco-churn-prediction/
├── churn_baseline.ipynb   # main notebook: from raw data to results
├── requirements.txt       # dependencies
├── README.md
└── data/                  # not in the repo — download from Kaggle (see below)
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🚀 How to run

```bash
git clone https://github.com/WizardCode428/telco-churn-prediction.git
cd telco-churn-prediction

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Then download the dataset from Kaggle, put the CSV into `data/`, open `churn_baseline.ipynb`, and choose **Run All**.

---

## 🗺️ Next steps

- [ ] Feature engineering: number of active services, "new customer" flag (`tenure ≤ 6`)
- [ ] Tune the decision threshold to match a business cost of a missed churner
- [ ] Explain individual predictions with SHAP
- [ ] Simple Streamlit demo: enter customer data → get churn probability

---

## 📚 Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, published on Kaggle by *blastchar*): 7,043 customers, 21 columns.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

![stonks](stonks.jpg)
