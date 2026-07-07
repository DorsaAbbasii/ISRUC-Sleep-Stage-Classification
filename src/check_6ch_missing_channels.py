from pathlib import Path
import os
import tempfile
import pandas as pd
import mne

DATASETS = {
    "non_normal": Path(r"C:\Users\Dorsa Abbasi\Downloads\non-normal  sleep-disorder patients"),
    "healthy": Path(r"C:\Users\Dorsa Abbasi\Downloads\healthy subjects"),
}

CHANNEL_FAMILIES = {
    "F3": ["F3-M2", "F3-A2", "F3"],
    "C3": ["C3-M2", "C3-A2", "C3"],
    "O1": ["O1-M2", "O1-A2", "O1"],
    "F4": ["F4-M1", "F4-A1", "F4"],
    "C4": ["C4-M1", "C4-A1", "C4"],
    "O2": ["O2-M1", "O2-A1", "O2"],
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def read_rec_channels(rec_path: Path):
    with tempfile.TemporaryDirectory(dir=str(rec_path.parent)) as tmpdir:
        temp_edf = Path(tmpdir) / f"{rec_path.stem}.edf"

        try:
            os.link(rec_path, temp_edf)
        except Exception:
            import shutil
            shutil.copy2(rec_path, temp_edf)

        raw = mne.io.read_raw_edf(temp_edf, preload=False, verbose=False)
        return raw.ch_names


def find_channel(ch_names, aliases):
    for alias in aliases:
        if alias in ch_names:
            return alias
    return None


rows = []

for group_name, root in DATASETS.items():
    print(f"Checking group: {group_name}")

    if group_name == "healthy":
        subject_items = [(subject_id, root / f"{subject_id}.rec") for subject_id in range(1, 11)]
    else:
        folders = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        folders = sorted(folders, key=lambda p: int(p.name))
        subject_items = [(int(folder.name), folder / f"{folder.name}.rec") for folder in folders]

    for subject_id, rec_path in subject_items:
        ch_names = read_rec_channels(rec_path)

        row = {
            "group": group_name,
            "subject": subject_id,
            "rec_path": str(rec_path),
        }

        missing = []

        for family, aliases in CHANNEL_FAMILIES.items():
            selected = find_channel(ch_names, aliases)
            row[f"{family}_selected"] = selected if selected is not None else "MISSING"

            if selected is None:
                missing.append(family)

        row["missing_families"] = ", ".join(missing)
        row["has_all_6_channels"] = len(missing) == 0
        row["available_channels"] = ", ".join(ch_names)

        rows.append(row)


df = pd.DataFrame(rows)

df.to_csv(RESULTS_DIR / "isruc_6ch_channel_availability.csv", index=False, encoding="utf-8-sig")

missing_df = df[df["has_all_6_channels"] == False]
missing_df.to_csv(RESULTS_DIR / "isruc_6ch_missing_channels.csv", index=False, encoding="utf-8-sig")

print("\nSaved:")
print("results/isruc_6ch_channel_availability.csv")
print("results/isruc_6ch_missing_channels.csv")

print("\nSummary:")
print("Total subjects:", len(df))
print("Subjects with all 6 channels:", df["has_all_6_channels"].sum())
print("Subjects missing at least one channel:", len(missing_df))

print("\nMissing subjects:")
if len(missing_df) == 0:
    print("None")
else:
    print(missing_df[["group", "subject", "missing_families"]].to_string(index=False))