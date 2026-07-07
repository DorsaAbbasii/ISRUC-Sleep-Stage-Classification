from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import mne

from scipy.signal import welch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


DATASETS = {
    "non_normal": Path(r"C:\Users\Dorsa Abbasi\Downloads\non-normal  sleep-disorder patients"),
    "healthy": Path(r"C:\Users\Dorsa Abbasi\Downloads\healthy subjects"),
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHANNEL_FAMILIES = {
    "F3": ["F3-M2", "F3-A2", "F3"],
    "C3": ["C3-M2", "C3-A2", "C3"],
    "O1": ["O1-M2", "O1-A2", "O1"],
    "F4": ["F4-M1", "F4-A1", "F4"],
    "C4": ["C4-M1", "C4-A1", "C4"],
    "O2": ["O2-M1", "O2-A1", "O2"],
}

EPOCH_SECONDS = 30

LABEL_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    5: "REM",
}


def read_labels(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        labels = [line.strip() for line in f if line.strip()]
    return [int(x) for x in labels]


def read_raw_rec_as_edf(rec_path: Path):
    """ISRUC files have .rec extension, but MNE reads EDF.
    A temporary .edf hard link/copy is used only during reading.
    """
    with tempfile.TemporaryDirectory(dir=str(rec_path.parent)) as tmpdir:
        temp_edf = Path(tmpdir) / f"{rec_path.stem}.edf"
        try:
            os.link(rec_path, temp_edf)
        except Exception:
            import shutil
            shutil.copy2(rec_path, temp_edf)
        raw = mne.io.read_raw_edf(temp_edf, preload=True, verbose=False)
        return raw


def select_channel_for_family(ch_names, aliases):
    for alias in aliases:
        if alias in ch_names:
            return alias
    return None


def read_rec_selected_channels(rec_path: Path):
    raw = read_raw_rec_as_edf(rec_path)
    ch_names = raw.ch_names

    selected = {}
    missing = []
    for family, aliases in CHANNEL_FAMILIES.items():
        selected_channel = select_channel_for_family(ch_names, aliases)
        if selected_channel is None:
            missing.append(family)
        else:
            selected[family] = selected_channel

    if missing:
        raise ValueError(
            f"Missing required channel family/families {missing}. Available channels: {ch_names}"
        )

    raw.pick(list(selected.values()))
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    return data, sfreq, selected


def bandpower_features(epochs, sfreq, prefix):
    """epochs shape: n_epochs x samples_per_epoch"""
    freqs, psd = welch(epochs, fs=sfreq, nperseg=512, axis=1)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    features = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        features[f"{prefix}_{band_name}_power"] = psd[:, mask].mean(axis=1)

    return features


def add_channel_features(rows, channel_epochs, sfreq, family_name):
    rows[f"{family_name}_mean"] = channel_epochs.mean(axis=1)
    rows[f"{family_name}_std"] = channel_epochs.std(axis=1)
    rows[f"{family_name}_min"] = channel_epochs.min(axis=1)
    rows[f"{family_name}_max"] = channel_epochs.max(axis=1)
    rows[f"{family_name}_median"] = np.median(channel_epochs, axis=1)
    rows[f"{family_name}_ptp"] = np.ptp(channel_epochs, axis=1)
    rows[f"{family_name}_energy"] = np.mean(channel_epochs ** 2, axis=1)
    rows.update(bandpower_features(channel_epochs, sfreq, family_name))


def extract_features_for_subject(group_name, subject_id, rec_path, ann_path):
    labels = read_labels(ann_path)
    data, sfreq, selected_channels = read_rec_selected_channels(rec_path)

    samples_per_epoch = int(sfreq * EPOCH_SECONDS)
    expected_epochs = data.shape[1] // samples_per_epoch
    n_epochs = min(expected_epochs, len(labels))

    if len(labels) != expected_epochs:
        print(
            f"Warning: {group_name} subject {subject_id}: "
            f"signal epochs={expected_epochs}, labels={len(labels)}, using={n_epochs}"
        )

    data = data[:, : n_epochs * samples_per_epoch]
    labels = labels[:n_epochs]

    rows = {
        "group": [group_name] * n_epochs,
        "subject": [subject_id] * n_epochs,
        "subject_group": [f"{group_name}_{subject_id}"] * n_epochs,
        "epoch": list(range(n_epochs)),
        "label": labels,
    }

    for idx, family_name in enumerate(CHANNEL_FAMILIES.keys()):
        channel_epochs = data[idx].reshape(n_epochs, samples_per_epoch)
        add_channel_features(rows, channel_epochs, sfreq, family_name)

    return pd.DataFrame(rows), selected_channels, n_epochs


all_subject_dfs = []
channel_rows = []
skipped_rows = []

for group_name, root in DATASETS.items():
    print(f"\nProcessing group: {group_name}")

    if group_name == "healthy":
        subject_items = [(subject_id, root / f"{subject_id}.rec", root / f"{subject_id}_1.txt") for subject_id in range(1, 11)]
    else:
        subject_folders = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        subject_folders = sorted(subject_folders, key=lambda p: int(p.name))
        subject_items = [
            (int(folder.name), folder / f"{folder.name}.rec", folder / f"{folder.name}_1.txt")
            for folder in subject_folders
        ]

    for subject_id, rec_path, ann_path in subject_items:
        print(f"Subject {subject_id}")
        try:
            df_subject, selected, n_epochs = extract_features_for_subject(group_name, subject_id, rec_path, ann_path)
        except ValueError as exc:
            print(f"  Skipped: {exc}")
            skipped_rows.append({
                "group": group_name,
                "subject": subject_id,
                "reason": str(exc),
            })
            continue

        all_subject_dfs.append(df_subject)
        row = {"group": group_name, "subject": subject_id, "epochs": n_epochs}
        for family, selected_channel in selected.items():
            row[f"{family}_channel_used"] = selected_channel
        channel_rows.append(row)

if not all_subject_dfs:
    raise RuntimeError("No subjects were processed. Check dataset paths and channels.")

df = pd.concat(all_subject_dfs, ignore_index=True)
df["label_name"] = df["label"].map(LABEL_NAMES)

feature_file = "isruc_features_clean_6ch.csv"
df.to_csv(feature_file, index=False, encoding="utf-8-sig")

channels_df = pd.DataFrame(channel_rows)
channels_df.to_csv(RESULTS_DIR / "isruc_clean_6ch_channels_used.csv", index=False, encoding="utf-8-sig")

skipped_df = pd.DataFrame(skipped_rows)
skipped_df.to_csv(RESULTS_DIR / "isruc_clean_6ch_skipped_subjects.csv", index=False, encoding="utf-8-sig")

print("\nFeature dataset saved:")
print(feature_file)
print("Total epochs:", len(df))
print("Processed subjects:", len(channel_rows))
print("Skipped subjects:", len(skipped_rows))

print("\nLabel counts:")
print(df["label_name"].value_counts())

print("\nChannels used per family:")
for family in CHANNEL_FAMILIES.keys():
    col = f"{family}_channel_used"
    print(f"{family}:")
    print(channels_df[col].value_counts().to_string())

if skipped_rows:
    print("\nSkipped subjects:")
    print(skipped_df[["group", "subject"]].to_string(index=False))

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

print("\nTraining Random Forest clean 6-channel EEG baseline...")
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

with open(RESULTS_DIR / "isruc_clean_6ch_baseline_report.txt", "w", encoding="utf-8") as f:
    f.write("ISRUC Sleep Staging Clean 6-Channel EEG Baseline\n")
    f.write("===============================================\n\n")
    f.write("Channel families: F3, C3, O1, F4, C4, O2\n")
    f.write(f"Accepted aliases: {CHANNEL_FAMILIES}\n")
    f.write("Model: Random Forest\n")
    f.write("Split: subject-wise train/test split\n")
    f.write(f"Total epochs: {len(df)}\n")
    f.write(f"Processed subjects: {len(channel_rows)}\n")
    f.write(f"Skipped subjects: {len(skipped_rows)}\n")
    f.write(f"Train epochs: {len(train_idx)}\n")
    f.write(f"Test epochs: {len(test_idx)}\n\n")

    if skipped_rows:
        f.write("Skipped subjects because at least one required 6-channel EEG family was missing:\n")
        f.write(skipped_df.to_string(index=False))
        f.write("\n\n")

    f.write("Channels used per subject:\n")
    f.write(channels_df.to_string(index=False))
    f.write("\n\n")

    f.write("Label counts:\n")
    f.write(df["label_name"].value_counts().to_string())
    f.write("\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix labels: Wake, N1, N2, N3, REM\n")
    f.write(str(cm))

print("\nDone.")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("\nClassification report:")
print(report)

print("\nSaved:")
print(feature_file)
print("results/isruc_clean_6ch_channels_used.csv")
print("results/isruc_clean_6ch_skipped_subjects.csv")
print("results/isruc_clean_6ch_baseline_report.txt")
