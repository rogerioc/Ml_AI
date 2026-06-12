# ML Classifier Benchmark — Breast Cancer Diagnosis

A clean side-by-side comparison of **seven classic machine-learning classifiers** on the **Breast Cancer Wisconsin** dataset, using scikit-learn. For each model it prints the confusion matrix and accuracy on a held-out test set, so you can compare them on the same split.

> Portfolio project covering supervised-learning fundamentals: preprocessing, train/test split, feature scaling, and model evaluation.

## The task

Binary classification of tumors as **benign (2)** or **malignant (4)** from 9 cytological features (clump thickness, cell-size uniformity, bare nuclei, mitoses, etc.) — the classic UCI Breast Cancer Wisconsin dataset (`Data.csv`).

## Models compared

| Model | scikit-learn estimator |
|---|---|
| Decision Tree | `DecisionTreeClassifier(criterion="entropy")` |
| Logistic Regression | `LogisticRegression` |
| K-Nearest Neighbors | `KNeighborsClassifier(n_neighbors=5, metric="minkowski")` |
| Support Vector Machine (linear) | `SVC(kernel="linear")` |
| Kernel SVM (RBF) | `SVC(kernel="rbf")` |
| Naive Bayes | `GaussianNB` |
| Random Forest | `RandomForestClassifier(n_estimators=10, criterion="entropy")` |

## Pipeline

1. Load `Data.csv`, split features (`X`) and label (`y`).
2. **Train/test split** (75/25, fixed `random_state` for reproducibility).
3. **Feature scaling** with `StandardScaler` (fit on train, transform both).
4. Train each classifier and evaluate with **confusion matrix + accuracy** on the test set.

## Running

```bash
pip install numpy pandas scikit-learn matplotlib
python RunAll.py
```

Output: for each model, its confusion matrix and accuracy score — a quick, honest benchmark on a shared split.

## What it demonstrates

- Comfort across the **core supervised-learning toolbox** (linear, tree-based, kernel, instance-based, probabilistic models).
- Correct **evaluation hygiene**: scaling fit only on training data, reproducible splits, comparison on identical test data.
- Reading results critically rather than reporting a single accuracy number.

## Tech stack

`Python` · `scikit-learn` · `pandas` · `NumPy`

---

*Built by [Rogério Celestino](https://rogerioc.github.io/about/) — senior software engineer focused on Applied AI.*
