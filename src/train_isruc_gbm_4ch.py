from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight


LABEL_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    5: "REM",
}

LABELS_ORDER = [0, 1, 2, 3, 5]

CHANNEL_FAMILIES = ["C3", "C4", "O1", "O2"]

FEATURE_COLUMNS = []
for family in CHANNEL_FAMILIES:
    FEATURE_COLUMNS.extend([
        f"{family}_mean",
        f"{family}_std",
        f"{family}_min",
        f"{family}_max",
        f"{family}_median",
        f"{family}_ptp",
        f"{family}_energy",
        f"{family}_delta_power",
        f"{family}_theta_power",
        f"{family}_alpha_power",
        f"{family}_beta_power",
    ])


def find_feature_file():
    possible_paths = [
        Path("isruc_features_clean_4ch.csv"),
        Path("..") / "isruc_features_clean_4ch.csv",
        Path(r"C:\Users\Dorsa Abbasi\Downloads\isruc_features_clean_4ch.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find isruc_features_clean_4ch.csv. "
        "Make sure the feature file exists in Downloads."
    )


def main():
    feature_path = find_feature_file()
    print("Loading features from:", feature_path)

    df = pd.read_csv(feature_path)

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["subject_group"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )

    print("\nTraining GBM / Histogram Gradient Boosting 4-channel EEG baseline...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    target_names = [LABEL_NAMES[x] for x in LABELS_ORDER]

    report = classification_report(
        y_test,
        y_pred,
        labels=LABELS_ORDER,
        target_names=target_names,
    )

    cm = confusion_matrix(y_test, y_pred, labels=LABELS_ORDER)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    report_path = results_dir / "isruc_gbm_4ch_baseline_report.txt"
    comparison_path = results_dir / "isruc_model_comparison.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ISRUC Sleep Staging GBM 4-Channel EEG Baseline\n")
        f.write("=============================================\n\n")
        f.write("Channel families: C3, C4, O1, O2\n")
        f.write("Model: HistGradientBoostingClassifier\n")
        f.write("Split: subject-wise train/test split\n")
        f.write(f"Feature file: {feature_path}\n")
        f.write(f"Total epochs: {len(df)}\n")
        f.write(f"Train epochs: {len(train_idx)}\n")
        f.write(f"Test epochs: {len(test_idx)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix labels: Wake, N1, N2, N3, REM\n")
        f.write(str(cm))

    comparison_df = pd.DataFrame([
        {
            "experiment": "Clean 4-channel EEG baseline",
            "model": "Random Forest",
            "accuracy": 0.6660,
            "macro_f1": 0.6109,
        },
        {
            "experiment": "Clean 4-channel EEG baseline",
            "model": "GBM / HistGradientBoosting",
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
        },
    ])

    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("\nClassification report:")
    print(report)

    print("\nSaved:")
    print(report_path)
    print(comparison_path)


if __name__ == "__main__":
    main()