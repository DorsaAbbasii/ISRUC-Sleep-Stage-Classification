from pathlib import Path
import os
import tempfile
import math
from collections import Counter, defaultdict

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

# Clean 4-channel EEG baseline. These four channel families are available for all 110 subjects
# in your downloaded dataset, but with different naming conventions.
CHANNEL_FAMILIES = {
    "C3": ["C3-M2", "C3-A2", "C3"],
    "C4": ["C4-M1", "C4-A1", "C4"],
    "O1": ["O1-M2", "O1-A2", "O1"],
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
LABELS_ORDER = [0, 1, 2, 3, 5]


def read_labels(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        labels = [line.strip() for line in f if line.strip()]
    return [int(x) for x in labels]


def read_rec_selected_channels(rec_path: Path):
    """Read .rec as EDF by creating a temporary .edf hard link/copy."""
    with tempfile.TemporaryDirectory(dir=str(rec_path.parent)) as tmpdir:
        temp_edf = Path(tmpdir) / f"{rec_path.stem}.edf"
        try:
            os.link(rec_path, temp_edf)
        except Exception:
            import shutil
            shutil.copy2(rec_path, temp_edf)

        raw = mne.io.read_raw_edf(temp_edf, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        ch_names = raw.ch_names

        selected = {}
        for family, aliases in CHANNEL_FAMILIES.items():
            chosen = None
            for alias in aliases:
                if alias in ch_names:
                    chosen = alias
                    break
            if chosen is None:
                raise ValueError(
                    f"No channel found for family {family} in {rec_path}. Available channels: {ch_names}"
                )
            selected[family] = chosen

        data = {}
        for family, ch_name in selected.items():
            data[family] = raw.get_data(picks=[ch_name])[0]

        return data, sfreq, selected


def bandpower_features(epochs, sfreq, prefix):
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


def basic_features(epochs, prefix):
    return {
        f"{prefix}_mean": epochs.mean(axis=1),
        f"{prefix}_std": epochs.std(axis=1),
        f"{prefix}_min": epochs.min(axis=1),
        f"{prefix}_max": epochs.max(axis=1),
        f"{prefix}_median": np.median(epochs, axis=1),
        f"{prefix}_ptp": np.ptp(epochs, axis=1),
        f"{prefix}_energy": np.mean(epochs ** 2, axis=1),
    }


def extract_features_for_subject(group_name, subject_id, rec_path, ann_path):
    labels = read_labels(ann_path)
    channel_data, sfreq, selected_channels = read_rec_selected_channels(rec_path)

    samples_per_epoch = int(sfreq * EPOCH_SECONDS)
    expected_epochs = min(len(sig) // samples_per_epoch for sig in channel_data.values())
    n_epochs = min(expected_epochs, len(labels))

    if len(labels) != expected_epochs:
        print(
            f"Warning: {group_name} subject {subject_id}: "
            f"signal epochs={expected_epochs}, labels={len(labels)}, using={n_epochs}"
        )

    rows = {
        "group": [group_name] * n_epochs,
        "subject": [subject_id] * n_epochs,
        "subject_group": [f"{group_name}_{subject_id}"] * n_epochs,
        "epoch": list(range(n_epochs)),
        "label": labels[:n_epochs],
    }

    for family, signal in channel_data.items():
        signal = signal[: n_epochs * samples_per_epoch]
        epochs = signal.reshape(n_epochs, samples_per_epoch)
        rows.update(basic_features(epochs, family))
        rows.update(bandpower_features(epochs, sfreq, family))
        rows[f"{family}_channel_used"] = [selected_channels[family]] * n_epochs

    return pd.DataFrame(rows), selected_channels, n_epochs


all_subject_dfs = []
channel_usage_rows = []
channel_alias_counter = defaultdict(Counter)

for group_name, root in DATASETS.items():
    print(f"\nProcessing group: {group_name}")

    if group_name == "healthy":
        subject_ids = range(1, 11)
        for subject_id in subject_ids:
            print(f"Subject {subject_id}")
            rec_path = root / f"{subject_id}.rec"
            ann_path = root / f"{subject_id}_1.txt"
            df_subject, selected, n_epochs = extract_features_for_subject(group_name, subject_id, rec_path, ann_path)
            all_subject_dfs.append(df_subject)
            row = {"group": group_name, "subject": subject_id, "epochs": n_epochs}
            for fam, ch in selected.items():
                row[f"{fam}_channel_used"] = ch
                channel_alias_counter[fam][ch] += 1
            channel_usage_rows.append(row)
    else:
        subject_folders = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        subject_folders = sorted(subject_folders, key=lambda p: int(p.name))
        for subject_folder in subject_folders:
            subject_id = int(subject_folder.name)
            print(f"Subject {subject_id}")
            rec_path = subject_folder / f"{subject_id}.rec"
            ann_path = subject_folder / f"{subject_id}_1.txt"
            df_subject, selected, n_epochs = extract_features_for_subject(group_name, subject_id, rec_path, ann_path)
            all_subject_dfs.append(df_subject)
            row = {"group": group_name, "subject": subject_id, "epochs": n_epochs}
            for fam, ch in selected.items():
                row[f"{fam}_channel_used"] = ch
                channel_alias_counter[fam][ch] += 1
            channel_usage_rows.append(row)


df = pd.concat(all_subject_dfs, ignore_index=True)
df["label_name"] = df["label"].map(LABEL_NAMES)
channel_usage_df = pd.DataFrame(channel_usage_rows)

feature_columns = []
for family in CHANNEL_FAMILIES:
    feature_columns.extend([
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

# Save features. This file can be large, but it is useful for reproducibility.
df.to_csv("isruc_features_clean_4ch.csv", index=False, encoding="utf-8-sig")
channel_usage_df.to_csv("isruc_clean_4ch_channels_used.csv", index=False, encoding="utf-8-sig")

print("\nFeature dataset saved:")
print("isruc_features_clean_4ch.csv")
print("Total epochs:", len(df))

print("\nLabel counts:")
print(df["label_name"].value_counts())

print("\nChannels used per family:")
for fam in CHANNEL_FAMILIES:
    print(f"{fam}:")
    for ch, count in channel_alias_counter[fam].most_common():
        print(f"  {ch}: {count}")

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
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

print("\nTraining Random Forest clean 4-channel EEG baseline...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

target_names = [LABEL_NAMES[x] for x in LABELS_ORDER]
report = classification_report(y_test, y_pred, labels=LABELS_ORDER, target_names=target_names)
cm = confusion_matrix(y_test, y_pred, labels=LABELS_ORDER)

with open("isruc_clean_4ch_baseline_report.txt", "w", encoding="utf-8") as f:
    f.write("ISRUC Sleep Staging Clean 4-Channel EEG Baseline\n")
    f.write("===============================================\n\n")
    f.write("Channel families: C3, C4, O1, O2\n")
    f.write(f"Accepted aliases: {CHANNEL_FAMILIES}\n")
    f.write("Model: Random Forest\n")
    f.write("Split: subject-wise train/test split\n")
    f.write(f"Total epochs: {len(df)}\n")
    f.write(f"Train epochs: {len(train_idx)}\n")
    f.write(f"Test epochs: {len(test_idx)}\n\n")
    f.write("Channels used per subject:\n")
    f.write(channel_usage_df.to_string(index=False))
    f.write("\n\nChannel alias counts:\n")
    for fam in CHANNEL_FAMILIES:
        f.write(f"\n{fam}:\n")
        for ch, count in channel_alias_counter[fam].most_common():
            f.write(f"  {ch}: {count}\n")
    f.write("\nLabel counts:\n")
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
print("isruc_features_clean_4ch.csv")
print("isruc_clean_4ch_channels_used.csv")
print("isruc_clean_4ch_baseline_report.txt")
