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

# Clean central EEG channel selection.
# These are the same channel family with different naming conventions in the dataset.
CHANNEL_ALIASES = ["C3-M2", "C3-A2", "C3"]
CANONICAL_CHANNEL_NAME = "C3" 
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


def read_raw_rec(rec_path: Path):
    """ISRUC recordings use .rec extension, but MNE reads them as EDF if extension is .edf."""
    with tempfile.TemporaryDirectory(dir=str(rec_path.parent)) as tmpdir:
        temp_edf = Path(tmpdir) / f"{rec_path.stem}.edf"
        try:
            os.link(rec_path, temp_edf)
        except Exception:
            import shutil
            shutil.copy2(rec_path, temp_edf)

        raw = mne.io.read_raw_edf(temp_edf, preload=True, verbose=False)
        return raw


def pick_clean_c3_channel(raw, rec_path: Path):
    for ch in CHANNEL_ALIASES:
        if ch in raw.ch_names:
            raw.pick([ch])
            return raw, ch
    raise ValueError(
        f"No clean C3 channel alias found in {rec_path}. Available channels: {raw.ch_names}"
    )


def load_rec_clean_c3(rec_path: Path):
    raw = read_raw_rec(rec_path)
    raw, channel_used = pick_clean_c3_channel(raw, rec_path)
    data = raw.get_data()[0]
    sfreq = raw.info["sfreq"]
    return data, sfreq, channel_used


def bandpower_features(epochs, sfreq):
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
        features[f"{band_name}_power"] = psd[:, mask].mean(axis=1)

    return features


def extract_features_for_subject(group_name, subject_id, rec_path, ann_path):
    labels = read_labels(ann_path)
    signal, sfreq, channel_used = load_rec_clean_c3(rec_path)

    samples_per_epoch = int(sfreq * EPOCH_SECONDS)
    expected_epochs = len(signal) // samples_per_epoch
    n_epochs = min(expected_epochs, len(labels))

    if len(labels) != expected_epochs:
        print(
            f"Warning: {group_name} subject {subject_id}: "
            f"signal epochs={expected_epochs}, labels={len(labels)}, using={n_epochs}"
        )

    signal = signal[: n_epochs * samples_per_epoch]
    labels = labels[:n_epochs]
    epochs = signal.reshape(n_epochs, samples_per_epoch)

    rows = {
        "group": [group_name] * n_epochs,
        "subject": [subject_id] * n_epochs,
        "subject_group": [f"{group_name}_{subject_id}"] * n_epochs,
        "epoch": list(range(n_epochs)),
        "channel_used": [channel_used] * n_epochs,
        "canonical_channel": [CANONICAL_CHANNEL_NAME] * n_epochs,
        "label": labels,
    }

    rows["mean"] = epochs.mean(axis=1)
    rows["std"] = epochs.std(axis=1)
    rows["min"] = epochs.min(axis=1)
    rows["max"] = epochs.max(axis=1)
    rows["median"] = np.median(epochs, axis=1)
    rows["ptp"] = np.ptp(epochs, axis=1)
    rows["energy"] = np.mean(epochs ** 2, axis=1)

    rows.update(bandpower_features(epochs, sfreq))
    return pd.DataFrame(rows)


all_subject_dfs = []
channel_rows = []

for group_name, root in DATASETS.items():
    print(f"\nProcessing group: {group_name}")

    if group_name == "healthy":
        subject_ids = range(1, 11)
        for subject_id in subject_ids:
            print(f"Subject {subject_id}")
            rec_path = root / f"{subject_id}.rec"
            ann_path = root / f"{subject_id}_1.txt"
            df_subject = extract_features_for_subject(group_name, subject_id, rec_path, ann_path)
            channel_rows.append({
                "group": group_name,
                "subject": subject_id,
                "channel_used": df_subject["channel_used"].iloc[0],
                "epochs": len(df_subject),
            })
            all_subject_dfs.append(df_subject)

    else:
        subject_folders = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        subject_folders = sorted(subject_folders, key=lambda p: int(p.name))
        for subject_folder in subject_folders:
            subject_id = int(subject_folder.name)
            print(f"Subject {subject_id}")
            rec_path = subject_folder / f"{subject_id}.rec"
            ann_path = subject_folder / f"{subject_id}_1.txt"
            df_subject = extract_features_for_subject(group_name, subject_id, rec_path, ann_path)
            channel_rows.append({
                "group": group_name,
                "subject": subject_id,
                "channel_used": df_subject["channel_used"].iloc[0],
                "epochs": len(df_subject),
            })
            all_subject_dfs.append(df_subject)


df = pd.concat(all_subject_dfs, ignore_index=True)
df["label_name"] = df["label"].map(LABEL_NAMES)
channel_df = pd.DataFrame(channel_rows)

features_path = "isruc_features_clean_C3.csv"
channels_path = "isruc_clean_C3_channels_used.csv"
report_path = "isruc_clean_C3_baseline_report.txt"

df.to_csv(features_path, index=False, encoding="utf-8-sig")
channel_df.to_csv(channels_path, index=False, encoding="utf-8-sig")

print("\nFeature dataset saved:")
print(features_path)
print("Total epochs:", len(df))

print("\nLabel counts:")
print(df["label_name"].value_counts())

print("\nChannels used per subject:")
print(channel_df["channel_used"].value_counts())

feature_columns = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "ptp",
    "energy",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
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

print("\nTraining Random Forest clean C3 baseline...")
model.fit(X_train, y_train)
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

with open(report_path, "w", encoding="utf-8") as f:
    f.write("ISRUC Sleep Staging Clean C3 Baseline\n")
    f.write("======================================\n\n")
    f.write("Channel family: C3 central EEG\n")
    f.write(f"Accepted aliases: {', '.join(CHANNEL_ALIASES)}\n")
    f.write("Model: Random Forest\n")
    f.write("Split: subject-wise train/test split\n")
    f.write(f"Total epochs: {len(df)}\n")
    f.write(f"Train epochs: {len(train_idx)}\n")
    f.write(f"Test epochs: {len(test_idx)}\n\n")
    f.write("Channels used per subject:\n")
    f.write(channel_df.to_string(index=False))
    f.write("\n\nChannel counts:\n")
    f.write(channel_df["channel_used"].value_counts().to_string())
    f.write("\n\nLabel counts:\n")
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
print(features_path)
print(channels_path)
print(report_path)
