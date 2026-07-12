from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight


FEATURE_FILE = Path("isruc_features_enhanced_6ch.csv")
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
        "Feature file not found: isruc_features_enhanced_6ch.csv\n"
        "Please run src/create_enhanced_6ch_features.py first."
    )

print("Loading enhanced 6-channel features...")
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

normalized_matrices = []
count_matrices = []
metric_rows = []
split_rows = []

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

    split_rows.append({
        "split": split_id,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    cm_counts = confusion_matrix(
        y_test,
        y_pred,
        labels=LABELS_ORDER,
    )

    cm_normalized = confusion_matrix(
        y_test,
        y_pred,
        labels=LABELS_ORDER,
        normalize="true",
    )

    count_matrices.append(cm_counts)
    normalized_matrices.append(cm_normalized)

    report = classification_report(
        y_test,
        y_pred,
        labels=LABELS_ORDER,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    for class_name in LABEL_NAMES:
        metric_rows.append({
            "split": split_id,
            "class": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1": report[class_name]["f1-score"],
            "support": report[class_name]["support"],
        })

mean_cm_counts = np.mean(count_matrices, axis=0)
mean_cm_normalized = np.mean(normalized_matrices, axis=0)

cm_counts_df = pd.DataFrame(
    mean_cm_counts,
    index=[f"true_{name}" for name in LABEL_NAMES],
    columns=[f"pred_{name}" for name in LABEL_NAMES],
)

cm_normalized_df = pd.DataFrame(
    mean_cm_normalized,
    index=[f"true_{name}" for name in LABEL_NAMES],
    columns=[f"pred_{name}" for name in LABEL_NAMES],
)

metrics_df = pd.DataFrame(metric_rows)
split_df = pd.DataFrame(split_rows)

class_summary = (
    metrics_df
    .groupby("class", as_index=False)
    .agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support_mean=("support", "mean"),
    )
)

confusion_pairs = []

for true_idx, true_name in enumerate(LABEL_NAMES):
    for pred_idx, pred_name in enumerate(LABEL_NAMES):
        if true_idx != pred_idx:
            confusion_pairs.append({
                "true_class": true_name,
                "predicted_class": pred_name,
                "mean_confusion_rate": mean_cm_normalized[true_idx, pred_idx],
            })

confusion_pairs_df = (
    pd.DataFrame(confusion_pairs)
    .sort_values(by="mean_confusion_rate", ascending=False)
)

counts_path = RESULTS_DIR / "isruc_enhanced_6ch_confusion_matrix_counts_mean.csv"
normalized_path = RESULTS_DIR / "isruc_enhanced_6ch_confusion_matrix_normalized_mean.csv"
class_metrics_path = RESULTS_DIR / "isruc_enhanced_6ch_class_metrics_repeated_splits.csv"
split_metrics_path = RESULTS_DIR / "isruc_enhanced_6ch_split_metrics.csv"
confusion_pairs_path = RESULTS_DIR / "isruc_enhanced_6ch_top_confusions.csv"
report_path = RESULTS_DIR / "isruc_enhanced_6ch_confusion_analysis_report.txt"
plot_path = RESULTS_DIR / "isruc_enhanced_6ch_confusion_matrix_normalized.png"

cm_counts_df.to_csv(counts_path, encoding="utf-8-sig")
cm_normalized_df.to_csv(normalized_path, encoding="utf-8-sig")
class_summary.to_csv(class_metrics_path, index=False, encoding="utf-8-sig")
split_df.to_csv(split_metrics_path, index=False, encoding="utf-8-sig")
confusion_pairs_df.to_csv(confusion_pairs_path, index=False, encoding="utf-8-sig")

old_n1_f1 = 0.3895
old_macro_f1 = 0.6702
old_accuracy = 0.6992

enhanced_n1_f1 = class_summary[class_summary["class"] == "N1"]["f1_mean"].iloc[0]
enhanced_macro_f1 = split_df["macro_f1"].mean()
enhanced_accuracy = split_df["accuracy"].mean()

with open(report_path, "w", encoding="utf-8") as f:
    f.write("ISRUC Enhanced 6-Channel Tuned GBM Confusion Matrix and Error Analysis\n")
    f.write("======================================================================\n\n")
    f.write("Model: Tuned HistGradientBoostingClassifier\n")
    f.write("Channels: F3, C3, O1, F4, C4, O2\n")
    f.write("Feature file: isruc_features_enhanced_6ch.csv\n")
    f.write("Evaluation: 5 repeated subject-wise train/test splits\n")
    f.write("Confusion matrix: mean normalized by true class\n\n")

    f.write("Overall comparison with original 6-channel GBM:\n")
    f.write(f"Original Accuracy Mean: {old_accuracy:.4f}\n")
    f.write(f"Enhanced Accuracy Mean: {enhanced_accuracy:.4f}\n")
    f.write(f"Accuracy Improvement: {enhanced_accuracy - old_accuracy:.4f}\n\n")

    f.write(f"Original Macro F1 Mean: {old_macro_f1:.4f}\n")
    f.write(f"Enhanced Macro F1 Mean: {enhanced_macro_f1:.4f}\n")
    f.write(f"Macro F1 Improvement: {enhanced_macro_f1 - old_macro_f1:.4f}\n\n")

    f.write(f"Original N1 F1 Mean: {old_n1_f1:.4f}\n")
    f.write(f"Enhanced N1 F1 Mean: {enhanced_n1_f1:.4f}\n")
    f.write(f"N1 F1 Improvement: {enhanced_n1_f1 - old_n1_f1:.4f}\n\n")

    f.write("Mean normalized confusion matrix:\n")
    f.write(cm_normalized_df.to_string())
    f.write("\n\nClass metrics across repeated splits:\n")
    f.write(class_summary.to_string(index=False))
    f.write("\n\nTop confusion pairs:\n")
    f.write(confusion_pairs_df.head(10).to_string(index=False))

    f.write("\n\nMain observation:\n")
    weakest_class = class_summary.sort_values(by="f1_mean").iloc[0]
    f.write(
        f"The weakest class is still {weakest_class['class']} "
        f"with mean F1 = {weakest_class['f1_mean']:.4f}.\n"
    )

    n1_confusions = confusion_pairs_df[confusion_pairs_df["true_class"] == "N1"].head(3)
    f.write("\nMost important N1 confusions:\n")
    f.write(n1_confusions.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.imshow(mean_cm_normalized)
plt.title("Mean Normalized Confusion Matrix - Enhanced 6ch Tuned GBM")
plt.xticks(range(len(LABEL_NAMES)), LABEL_NAMES)
plt.yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar(label="Mean rate")

for i in range(len(LABEL_NAMES)):
    for j in range(len(LABEL_NAMES)):
        plt.text(
            j,
            i,
            f"{mean_cm_normalized[i, j]:.2f}",
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print("\nSaved:")
print(counts_path)
print(normalized_path)
print(class_metrics_path)
print(split_metrics_path)
print(confusion_pairs_path)
print(report_path)
print(plot_path)

print("\nMean normalized confusion matrix:")
print(cm_normalized_df)

print("\nClass metrics:")
print(class_summary)

print("\nTop confusion pairs:")
print(confusion_pairs_df.head(10).to_string(index=False))

print("\nComparison:")
print("Original Accuracy:", old_accuracy)
print("Enhanced Accuracy:", round(enhanced_accuracy, 4))
print("Original Macro F1:", old_macro_f1)
print("Enhanced Macro F1:", round(enhanced_macro_f1, 4))
print("Original N1 F1:", old_n1_f1)
print("Enhanced N1 F1:", round(enhanced_n1_f1, 4))