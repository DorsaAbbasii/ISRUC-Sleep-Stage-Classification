from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight


FEATURE_FILE = Path(r"C:\Users\Dorsa Abbasi\Downloads\isruc_features_clean_4ch.csv")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    5: "REM",
}

print("Loading features...")
df = pd.read_csv(FEATURE_FILE)

exclude_columns = {
    "group",
    "subject",
    "subject_group",
    "epoch",
    "label",
    "label_name",
}

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
]

X = df[feature_columns]
y = df["label"]
groups = df["subject_group"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

param_grid = {
    "learning_rate": [0.03, 0.05, 0.08],
    "max_iter": [200, 300],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 0.1],
}

results = []
best_model = None
best_score = -1
best_params = None
best_pred = None

print("Starting GBM tuning...")

for i, params in enumerate(ParameterGrid(param_grid), start=1):
    print(f"\nExperiment {i}")
    print(params)

    model = HistGradientBoostingClassifier(
        **params,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    row = {
        **params,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    results.append(row)

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = model
        best_params = params
        best_pred = y_pred

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="macro_f1", ascending=False)

results_df.to_csv(
    RESULTS_DIR / "isruc_gbm_tuning_results.csv",
    index=False,
    encoding="utf-8-sig",
)

labels_order = [0, 1, 2, 3, 5]
target_names = [LABEL_NAMES[x] for x in labels_order]

report = classification_report(
    y_test,
    best_pred,
    labels=labels_order,
    target_names=target_names,
)

cm = confusion_matrix(y_test, best_pred, labels=labels_order)

with open(RESULTS_DIR / "isruc_gbm_tuned_4ch_report.txt", "w", encoding="utf-8") as f:
    f.write("ISRUC Sleep Staging Tuned GBM 4-Channel EEG Baseline\n")
    f.write("====================================================\n\n")
    f.write("Channel families: C3, C4, O1, O2\n")
    f.write("Model: Tuned HistGradientBoostingClassifier\n")
    f.write("Split: subject-wise train/test split\n")
    f.write(f"Feature file: {FEATURE_FILE}\n")
    f.write(f"Total epochs: {len(df)}\n")
    f.write(f"Train epochs: {len(train_idx)}\n")
    f.write(f"Test epochs: {len(test_idx)}\n\n")

    f.write("Best parameters:\n")
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")

    f.write("\n")
    f.write(f"Best Accuracy: {accuracy_score(y_test, best_pred):.4f}\n")
    f.write(f"Best Macro F1: {f1_score(y_test, best_pred, average='macro'):.4f}\n\n")

    f.write(report)
    f.write("\nConfusion matrix labels: Wake, N1, N2, N3, REM\n")
    f.write(str(cm))

print("\nDone.")
print("Best parameters:")
print(best_params)
print("Best Accuracy:", round(accuracy_score(y_test, best_pred), 4))
print("Best Macro F1:", round(f1_score(y_test, best_pred, average="macro"), 4))

print("\nSaved:")
print("results/isruc_gbm_tuning_results.csv")
print("results/isruc_gbm_tuned_4ch_report.txt")