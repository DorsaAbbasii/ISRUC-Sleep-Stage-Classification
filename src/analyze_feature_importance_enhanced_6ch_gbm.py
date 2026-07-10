from pathlib import Path
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance


FEATURE_FILE = Path("isruc_features_enhanced_6ch.csv")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHANNELS = ["F3", "C3", "O1", "F4", "C4", "O2"]

BEST_PARAMS = {
    "l2_regularization": 0.1,
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
}

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        "Feature file not found: isruc_features_enhanced_6ch.csv\n"
        "Please run src/create_enhanced_6ch_features.py first."
    )

print("Loading enhanced 6-channel feature file...")
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

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

print("Training enhanced 6-channel tuned GBM...")
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

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nRunning permutation importance...")
print("This may take several minutes.")

scorer = make_scorer(f1_score, average="macro")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring=scorer,
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
})

importance_df = importance_df.sort_values(by="importance_mean", ascending=False)


def remove_prev_prefix(feature_name):
    if feature_name.startswith("prev_"):
        return feature_name[len("prev_"):]
    return feature_name


def get_channel(feature_name):
    clean_name = remove_prev_prefix(feature_name)

    for ch in CHANNELS:
        if clean_name.startswith(ch + "_"):
            return ch

    for left in CHANNELS:
        for right in CHANNELS:
            prefix = f"{left}_{right}_"
            if clean_name.startswith(prefix):
                return f"{left}-{right}"

    return "unknown"


def get_feature_group(feature_name):
    if feature_name.startswith("prev_"):
        return "previous_epoch_context"

    if "_rel_" in feature_name:
        return "relative_band_power"

    if "ratio" in feature_name:
        return "band_power_ratio"

    if feature_name.endswith("_diff") or feature_name.endswith("_absdiff"):
        return "left_right_difference"

    if any(feature_name.endswith("_" + band) for band in ["delta_power", "theta_power", "alpha_power", "beta_power"]):
        return "original_band_power"

    return "original_statistical"


def get_feature_type(feature_name):
    clean_name = remove_prev_prefix(feature_name)

    if "_rel_" in clean_name:
        return clean_name.split("_rel_")[-1]

    for ch in CHANNELS:
        prefix = ch + "_"
        if clean_name.startswith(prefix):
            return clean_name[len(prefix):]

    return clean_name


importance_df["channel"] = importance_df["feature"].apply(get_channel)
importance_df["feature_group"] = importance_df["feature"].apply(get_feature_group)
importance_df["feature_type"] = importance_df["feature"].apply(get_feature_type)
importance_df["importance_positive"] = importance_df["importance_mean"].clip(lower=0)

by_channel = (
    importance_df
    .groupby("channel", as_index=False)["importance_positive"]
    .sum()
    .sort_values(by="importance_positive", ascending=False)
)

by_group = (
    importance_df
    .groupby("feature_group", as_index=False)["importance_positive"]
    .sum()
    .sort_values(by="importance_positive", ascending=False)
)

by_type = (
    importance_df
    .groupby("feature_type", as_index=False)["importance_positive"]
    .sum()
    .sort_values(by="importance_positive", ascending=False)
)

importance_path = RESULTS_DIR / "isruc_enhanced_6ch_gbm_feature_importance.csv"
channel_path = RESULTS_DIR / "isruc_enhanced_6ch_gbm_importance_by_channel.csv"
group_path = RESULTS_DIR / "isruc_enhanced_6ch_gbm_importance_by_feature_group.csv"
type_path = RESULTS_DIR / "isruc_enhanced_6ch_gbm_importance_by_feature_type.csv"
report_path = RESULTS_DIR / "isruc_enhanced_6ch_gbm_feature_importance_report.txt"

importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
by_channel.to_csv(channel_path, index=False, encoding="utf-8-sig")
by_group.to_csv(group_path, index=False, encoding="utf-8-sig")
by_type.to_csv(type_path, index=False, encoding="utf-8-sig")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("ISRUC Enhanced 6-Channel Tuned GBM Feature Importance Analysis\n")
    f.write("==============================================================\n\n")
    f.write("Model: Tuned HistGradientBoostingClassifier\n")
    f.write("Channels: F3, C3, O1, F4, C4, O2\n")
    f.write("Feature file: isruc_features_enhanced_6ch.csv\n")
    f.write("Method: permutation importance on the test set\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")

    f.write("Top 30 features:\n")
    f.write(importance_df.head(30).to_string(index=False))
    f.write("\n\nImportance by feature group:\n")
    f.write(by_group.to_string(index=False))
    f.write("\n\nImportance by channel:\n")
    f.write(by_channel.to_string(index=False))
    f.write("\n\nImportance by feature type:\n")
    f.write(by_type.head(30).to_string(index=False))

print("\nSaved:")
print(importance_path)
print(channel_path)
print(group_path)
print(type_path)
print(report_path)

print("\nTop 20 features:")
print(importance_df.head(20)[["feature", "importance_mean", "importance_std", "feature_group"]].to_string(index=False))

print("\nImportance by feature group:")
print(by_group.to_string(index=False))

print("\nImportance by channel:")
print(by_channel.to_string(index=False))