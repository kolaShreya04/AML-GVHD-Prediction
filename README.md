# AML-GVHD Prediction Model

Machine learning models to predict Graft-versus-Host Disease (GVHD) and survival outcomes in high-risk Acute Myeloid Leukemia (AML) patients post-hematopoietic stem cell transplantation (HSCT).

---

## 📋 Project Overview

This project uses clinical data from 20 HR-AML patients who underwent HSCT to predict:
- **Acute GVHD (aGVHD)** occurrence
- **Patient survival** status

The dataset includes 84 features covering:
- Demographics (age, gender)
- Clinical history (diagnosis, chemotherapy)
- Transplant details (donor match, transplant type)
- Post-transplant outcomes (lymphocyte counts, infections, GVHD)

---

## 🧠 Models Implemented

| Model | Type | Library |
|-------|------|---------|
| Random Forest | Ensemble ML | scikit-learn |
| Linear Regression | Regression | scikit-learn |
| SVM | Classification | scikit-learn |
| SGD Classifier | Linear ML | scikit-learn |
| XGBoost | Gradient Boosting | xgboost |
| Feedforward Neural Network | Deep Learning | PyTorch |

---

## 🛠️ Data Cleaning Pipeline

The `Data_Cleaning.py` file handles:

- **Row/column filtering** — drops unnecessary identifiers
- **Date processing** — converts dates and calculates days between diagnosis and transplant
- **Missing value handling** — replaces `/` with `'none'` or `0`
- **One-hot encoding** — for categorical features
- **MinMax scaling** — normalizes all numerical features

---

## 📊 Model Performance

### Random Forest (for aGVHD prediction)
- **Accuracy:** 0.95
- **MAE:** 0.05
- **Top predictive features:**
  - `Mutations/fusion genes_none`
  - `AML Classification_m4`
  - `pre-transplant status_cr`

### Linear Regression
- **MAE:** 0.056

---

## 🚀 How to Run

### Clone the repository
```bash
git clone https://github.com/yourusername/AML-GVHD-Prediction.git
cd AML-GVHD-Prediction

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the ML models
python py_ml.py

# Run the PyTorch models
python model423.py
# or
python skele5.py
```

## Results output

```bash
Randforest mae and accuracy
0.05 0.95
0.95
              precision    recall  f1-score   support

           0       1.00      0.75      0.86         4
           1       0.94      1.00      0.97        16

    accuracy                           0.95        20
   macro avg       0.97      0.88      0.91        20
weighted avg       0.95      0.95      0.95        20

[[ 3  1]
 [ 0 16]]

Linear Regression mae
0.056247954537702

Top 10 positive feature importances:
 Mutations/fusion genes_none , Score: 0.05452
 AML Classification_m4 , Score: 0.03693
 pre-transplant status_cr , Score: 0.03683
 End ( NK cells (K/ml) ) , Score: 0.03382
 C4D1 ( CD8+ T cells (K/ml) ) , Score: 0.03325
 C4D1 ( B cells (K/ml) ) , Score: 0.03184
 C5D1 ( NK cells (K/ml) ) , Score: 0.03062
 C1D1 ( Treg cells (K/ml) ) , Score: 0.03049
 WBC ( Minimum blood count for the fourth course of treatment ) , Score: 0.02866
 C1D1 ( NK cells (K/ml) ) , Score: 0.02771

Top 10 negative feature importances:
 HGB ( Minimum blood count for the fourth course of treatment ) , Score: -0.00064
 PLT ( Minimum blood count for the first course of treatment ) , Score: -0.00207
 C5D1 ( CD8+ T cells (K/ml) ) , Score: -0.00238
 Number of pre-transplant chemotherapy treatments , Score: -0.00262
 Mutations/fusion genes_mds isomerization , Score: -0.00265
 C3D1 ( NK cells (K/ml) ) , Score: -0.00272
 C2D1 ( CD8+ T cells (K/ml) ) , Score: -0.00297
 Intial Diagnosis Risk levl_medium , Score: -0.00405
 PLT ( Minimum blood count for the second course of treatment ) , Score: -0.00406
 HGB ( The fifth course of decitabine treatment ) , Score: -0.00631
```
