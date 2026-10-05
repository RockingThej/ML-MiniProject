"""Supervised models: tune with repeated stratified CV on the training set,
select by CV F1, then evaluate every tuned model once on a held-out test set.

Scaling/encoding lives inside each Pipeline, so it is fitted only on training
folds (no leakage). With only ~300 patients, CV is more reliable than a
single small validation split.
"""
import json
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"   # also applies to joblib worker processes
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import MODELS, RANDOM_STATE, RESULTS, load_data, make_preprocessor, train_test

R = RANDOM_STATE


def model_zoo():
    """name -> (estimator, param_grid). Grid keys are for the 'clf' step."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=5000),
                                {"C": [0.01, 0.1, 1, 10, 100]}),
        "Elastic Net": (LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000,
                                           random_state=R),
                        {"C": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=R),
                              {"learning_rate": [0.05, 0.1, 0.25],
                               "n_estimators": [50, 100], "max_depth": [1, 2]}),
        "Bagging": (BaggingClassifier(DecisionTreeClassifier(), random_state=R),
                    {"n_estimators": [10, 50, 100]}),
        "Random Forest": (RandomForestClassifier(random_state=R),
                          {"n_estimators": [200, 500], "max_depth": [None, 5],
                           "max_features": ["sqrt"], "min_samples_leaf": [1, 3]}),
        "SVM (linear)": (SVC(kernel="linear", probability=True, random_state=R),
                         {"C": [0.01, 0.1, 1, 10]}),
        "SVM (rbf)": (SVC(kernel="rbf", probability=True, random_state=R),
                      {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]}),
        "SVM (poly)": (SVC(kernel="poly", probability=True, random_state=R),
                       {"C": [0.1, 1, 10], "degree": [2, 3, 4]}),
        "Neural Net (MLP)": (MLPClassifier(max_iter=1000, random_state=R),
                             {"hidden_layer_sizes": [(16, 8), (18, 15, 10, 8)],
                              "alpha": [1e-3, 1e-1], "learning_rate_init": [1e-2]}),
    }


def scores(model, X):
    return model.predict_proba(X)[:, 1]


def main():
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test(df)
    print(f"Train {X_tr.shape}, Test {X_te.shape}; "
          f"high-risk share train={y_tr.mean():.2f} test={y_te.mean():.2f}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=R)
    rows, fitted = [], {}
    for name, (est, grid) in model_zoo().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", est)])
        grid = {f"clf__{k}": v for k, v in grid.items()}
        gs = GridSearchCV(pipe, grid, scoring={"f1": "f1", "roc_auc": "roc_auc"},
                          refit="f1", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        fitted[name] = best
        i = gs.best_index_
        pred, prob = best.predict(X_te), scores(best, X_te)
        rows.append({
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_f1": gs.cv_results_["mean_test_f1"][i],
            "cv_f1_std": gs.cv_results_["std_test_f1"][i],
            "cv_auroc": gs.cv_results_["mean_test_roc_auc"][i],
            "test_accuracy": accuracy_score(y_te, pred),
            "test_auroc": roc_auc_score(y_te, prob),
            "test_precision": precision_score(y_te, pred),
            "test_recall": recall_score(y_te, pred),
            "test_f1": f1_score(y_te, pred),
        })
        print(f"{name:22s} CV-F1={rows[-1]['cv_f1']:.3f}  Test AUROC={rows[-1]['test_auroc']:.3f} "
              f"Recall={rows[-1]['test_recall']:.3f}")

    res = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    res.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("\n", res.drop(columns="best_params").round(3).to_string(index=False))

    # Majority-class baseline for context
    maj = max(y_te.mean(), 1 - y_te.mean())
    print(f"\nMajority-class baseline accuracy on test: {maj:.3f}")

    # Selected model = best CV F1 (selection never looks at the test set)
    best_name = res.iloc[0]["model"]
    best_model = fitted[best_name]
    print("Selected model:", best_name)
    joblib.dump(best_model, MODELS / "best_model.joblib")
    (MODELS / "best_model.json").write_text(json.dumps(
        {"name": best_name, "params": {k: str(v) for k, v in res.iloc[0]["best_params"].items()}},
        indent=2))

    # ROC / PR curves for all models on the test set
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in fitted.items():
        RocCurveDisplay.from_predictions(y_te, scores(m, X_te), name=name, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y_te, scores(m, X_te), name=name, ax=ax[1])
    ax[0].set_title("ROC (test set)"); ax[1].set_title("Precision-Recall (test set)")
    plt.tight_layout(); plt.savefig(RESULTS / "roc_pr_all_models.png", dpi=150); plt.close()

    ConfusionMatrixDisplay.from_estimator(best_model, X_te, y_te, cmap="Blues")
    plt.title(f"Confusion matrix: {best_name}")
    plt.savefig(RESULTS / "confusion_matrix_best.png", dpi=150); plt.close()

    # Which features matter? (permutation importance on the test set)
    pi = permutation_importance(best_model, X_te, y_te, scoring="roc_auc",
                                n_repeats=30, random_state=R)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    imp.plot.barh(figsize=(7, 5), title=f"Permutation importance (AUROC drop): {best_name}")
    plt.tight_layout(); plt.savefig(RESULTS / "feature_importance.png", dpi=150); plt.close()
    print("Saved model and figures.")


if __name__ == "__main__":
    main()
