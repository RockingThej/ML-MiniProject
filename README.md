# Predicting Myocardial Infarction (MI) Risk using Supervised and Unsupervised Machine Learning

**Course:** UE24CS352A Machine Learning, Mini-Project

**Team**
| Name | SRN |
|---|---|
| Tharunyah Sudhakar | PES2UG24CS561 |
| Thejaswi S | PES2UG24CS562 |

---

## 1. Problem Statement
Classify a patient as **high or low MI risk** from 13 routine clinical measurements, and use unsupervised
learning (PCA, K-Means, DBSCAN) to check whether risk-related patient groups emerge without using the labels.

## 2. Dataset
Kaggle: *Heart Attack Analysis & Prediction Dataset* (`heart.csv`).

- 303 patients, 13 features + binary target `output` (1 = high risk).
- 302 patients remain after removing 1 duplicate row. About 54% are labelled high risk, about 68% are male. No missing values.
- Download `heart.csv` and place it at `data/heart.csv` (the file is not included in the repo).
- Some copies use the original UCI column names; `src/common.py` renames them automatically.

| Feature | Meaning |
|---|---|
| age, sex | Age in years; sex (0 = female, 1 = male) |
| cp | Chest pain type (0-3) |
| trtbps, chol | Resting blood pressure (mmHg); serum cholesterol (mg/dL) |
| fbs | Fasting blood sugar flag |
| restecg | Resting ECG result (0-2) |
| thalachh | Maximum heart rate achieved |
| exng | Exercise-induced angina (0/1) |
| oldpeak | ST depression (continuous) |
| slp | Slope of peak exercise ST segment (0-2) |
| caa | Number of major vessels flagged (documented 0-3) |
| thall | Thallium stress test result (documented 1-3) |

**Data quirks we found:** `caa` contains a value 4 and `thall` contains a value 0 (outside the documented ranges).
We kept them; `thall` is one-hot encoded so 0 becomes its own category.

## 3. Repository Structure
```
.
├── data/                 # put heart.csv here (not committed)
├── results/              # generated figures and CSV tables
├── models/               # saved best model (generated, not committed)
├── src/
│   ├── common.py         # config, data loading, preprocessing pipeline
│   ├── eda.py            # exploratory data analysis
│   ├── clustering.py     # PCA, K-Means, DBSCAN
│   └── train.py          # tuning, model selection, evaluation, plots
├── requirements.txt
└── README.md
```

## 4. Setup
Requires Python 3.9+.
```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Then place `heart.csv` in `data/`.

## 5. How to Run
Run from the repository root, in this order:
```bash
python src/eda.py          # EDA figures + summary table          -> results/
python src/clustering.py   # PCA / K-Means / DBSCAN analysis       -> results/
python src/train.py        # tune 9 models, evaluate, save best    -> results/, models/
```
`train.py` is the slowest step (grid search with repeated cross-validation over nine models).
All outputs are written to `results/`:

| File | Content |
|---|---|
| `eda_summary.csv`, `eda_correlation.png`, `eda_continuous.png`, `eda_categorical.png` | Data exploration |
| `clustering_results.csv`, `clustering.png`, `kmeans_cluster_profiles.csv` | Unsupervised analysis |
| `model_comparison.csv` | CV and test metrics for all 9 models, best hyperparameters |
| `roc_pr_all_models.png`, `confusion_matrix_best.png`, `feature_importance.png` | Supervised evaluation |

## 6. Methodology
**Preprocessing.** Continuous features (age, trtbps, chol, thalachh, oldpeak) are standardised; cp, restecg, slp and thall are
one-hot encoded; binary features and caa pass through. Everything is inside scikit-learn `Pipeline`s, so scaling and encoding are
fitted on training folds only (no data leakage).

**Unsupervised analysis.** On scaled data: PCA, K-Means with k = 2 (tuning PCA components 2-10) and DBSCAN (grid over eps and min_samples).
Labels are used only afterwards, to measure agreement with Adjusted Rand Index (ARI) and silhouette score.

**Supervised learning.**
- Stratified 80/20 split: 242 train, 61 test (`random_state = 42`).
- Nine models: Logistic Regression, Elastic Net, Gradient Boosting, Bagging, Random Forest, SVM (linear, RBF, polynomial), MLP.
- Hyperparameters tuned with `GridSearchCV` using 5-fold x 2-repeat stratified cross-validation on the training set only.
- The final model is chosen by **cross-validated F1**; the test set is used once, for reporting.
- Metrics: F1, AUROC, precision, recall (sensitivity), accuracy, ROC/PR curves, confusion matrix, permutation importance.

## 7. Results

**Selected model: Elastic Net** (C = 0.1, L1 ratio = 0.5)

| Model | CV F1 | Test AUROC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|
| **Elastic Net (selected)** | 0.871 | 0.885 | 0.909 | 0.789 | 0.845 | 0.820 |
| Random Forest | 0.871 | 0.890 | 0.848 | 0.737 | 0.789 | 0.754 |
| SVM (RBF) | 0.870 | 0.893 | 0.909 | 0.789 | 0.845 | 0.820 |
| Logistic Regression | 0.869 | 0.886 | 0.909 | 0.811 | 0.857 | 0.836 |
| Gradient Boosting | 0.868 | 0.871 | 0.879 | 0.784 | 0.829 | 0.803 |
| SVM (linear) | 0.866 | 0.899 | 0.909 | 0.789 | 0.845 | 0.820 |
| SVM (polynomial) | 0.852 | 0.904 | 0.939 | 0.705 | 0.805 | 0.754 |
| Bagging | 0.833 | 0.858 | 0.879 | 0.784 | 0.829 | 0.803 |
| Neural Net (MLP) | 0.804 | 0.812 | 0.818 | 0.771 | 0.794 | 0.770 |

- Test confusion matrix (Elastic Net): 30 true positives, 20 true negatives, 8 false positives, 3 false negatives. Majority-class accuracy is about 0.54.
- Most important features (permutation importance): `caa`, `thall`, `thalachh`, `oldpeak`, `cp`.
- **Clustering:** K-Means ARI 0.30 (3 PCA components; 0.285 with 2), silhouette 0.33-0.41. DBSCAN ARI 0.16 (many points labelled noise).
  The two K-Means groups differ clearly in age, peak heart rate, exercise angina and vessel findings.

## 8. Limitations
1. **Small test set (61 patients).** One patient changes accuracy by about 1.6 points, so differences between the top models are within noise. The top four models are tied on CV F1 (0.868-0.871).
   Elastic Net was chosen by the pre-set rule and because it is the most interpretable.
2. **Label direction.** In this copy of the dataset, exercise angina, ST depression, vessel count and male sex are associated with the *lower* value of `output`,
   opposite to clinical expectation. The label coding may be inverted. We kept the coding as given; this does not change predictive metrics but affects clinical interpretation.
3. **Single small cohort, no external validation.** This is a teaching exercise, not a clinical tool.
4. **Future work:** bootstrap confidence intervals, nested cross-validation, external datasets, probability calibration.

## 9. Individual Contributions

**Thejaswi S (PES2UG24CS562)**
- Repository setup and documentation (GitHub repo, README, project structure)
- Dataset acquisition, cleaning and data-quality checks (duplicate removal, `caa` / `thall` coding anomalies)
- Exploratory data analysis (`src/eda.py`): correlation matrix, distributions, categorical risk plots
- Unsupervised analysis (`src/clustering.py`): PCA, K-Means, DBSCAN, ARI / silhouette evaluation, cluster profiling

**Tharunyah Sudhakar (PES2UG24CS561)**
- Preprocessing pipeline and train/test strategy (`src/common.py`)
- Supervised modelling (`src/train.py`): tuning with repeated stratified cross-validation for 9 models, model selection
- Evaluation: test metrics, ROC/PR curves, confusion matrix, permutation importance
- Interpretation of model results and live demonstration of the training pipeline

**Both**
- Problem definition and literature review
- Interpretation of results and limitations (including the label-coding issue)
- Write-up and presentation slides

