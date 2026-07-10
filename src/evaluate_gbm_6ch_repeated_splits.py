from pathlib import Path
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


FEATURE_FILE = Path("isruc_features_clean_6ch.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BEST_PARAMS = {
    "l2_regularization": 0.1,
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
}

LABELS_ORDER = [0, 1, 2, 3, 5]
LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        "Feature file not found: isruc_features_clean_6ch.csv\n"
        "Please run src/train_isruc_baseline_clean_6ch.py first."
    )

print("Loading 6-channel features...")
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

print("Total epochs:", len(df))
print("Subjects:", df["subject_group"].nunique())
print("Features:", len(feature_columns))

splitter = GroupShuffleSplit(
    n_splits=5,
    test_size=0.2,
    random_state=42,
)

rows = []
reports = []

for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
    print(f"\nSplit {split_id}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = HistGradientBoostingClassifier(
        **BEST_PARAMS,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    row = {
        "split": split_id,
        "train_epochs": len(train_idx),
        "test_epochs": len(test_idx),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    for label, name in zip(LABELS_ORDER, LABEL_NAMES):
        row[f"{name}_f1"] = f1_score(y_test, y_pred, labels=[label], average="macro")

    rows.append(row)

    reports.append(
        f"Split {split_id}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n"
        + classification_report(
            y_test,
            y_pred,
            labels=LABELS_ORDER,
            target_names=LABEL_NAMES,
        )
        + "\n\n"
    )

results_df = pd.DataFrame(rows)

summary = results_df[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])

results_path = RESULTS_DIR / "isruc_gbm_6ch_repeated_splits.csv"
report_path = RESULTS_DIR / "isruc_gbm_6ch_repeated_splits_report.txt"

results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("ISRUC 6-Channel Tuned GBM Repeated Subject-wise Splits\n")
    f.write("======================================================\n\n")
    f.write("Model: Tuned HistGradientBoostingClassifier\n")
    f.write("Channels: F3, C3, O1, F4, C4, O2\n")
    f.write("Evaluation: 5 repeated subject-wise train/test splits\n")
    f.write("Feature file: isruc_features_clean_6ch.csv\n")
    f.write(f"Total epochs: {len(df)}\n")
    f.write(f"Subjects: {df['subject_group'].nunique()}\n")
    f.write(f"Number of features: {len(feature_columns)}\n\n")

    f.write("Summary:\n")
    f.write(summary.to_string())
    f.write("\n\nDetailed split results:\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\nClassification reports:\n\n")
    for report in reports:
        f.write(report)

print("\nDone.")
print("\nSummary:")
print(summary)

print("\nSaved:")
print(results_path)
print(report_path)       