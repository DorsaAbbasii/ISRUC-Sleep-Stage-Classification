from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


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

print("Loading 4-channel feature file...")
df = pd.read_csv(FEATURE_FILE)

print("Original epochs:", len(df))

# Remove the same subject skipped in the strict 6-channel experiment
df = df[~((df["group"] == "non_normal") & (df["subject"] == 8))].copy()

print("Filtered epochs:", len(df))
print("Subjects:", df["subject_group"].nunique())

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

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

print("Training Random Forest 4-channel same-109 baseline...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

labels_order = [0, 1, 2, 3, 5]
target_names = [LABEL_NAMES[x] for x in labels_order]

report = classification_report(
    y_test,
    y_pred,
    labels=labels_order,
    target_names=target_names,
)

cm = confusion_matrix(y_test, y_pred, labels=labels_order)

output_path = RESULTS_DIR / "isruc_clean_4ch_same109_baseline_report.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("ISRUC Sleep Staging Clean 4-Channel EEG Baseline Same-109 Subjects\n")
    f.write("==================================================================\n\n")
    f.write("Channel families: C3, C4, O1, O2\n")
    f.write("Model: Random Forest\n")
    f.write("Split: subject-wise train/test split\n")
    f.write("Excluded subject: non_normal subject 8\n")
    f.write(f"Total epochs: {len(df)}\n")
    f.write(f"Train epochs: {len(train_idx)}\n")
    f.write(f"Test epochs: {len(test_idx)}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix labels: Wake, N1, N2, N3, REM\n")
    f.write(str(cm))

print("\nDone.")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("\nSaved:")
print(output_path)