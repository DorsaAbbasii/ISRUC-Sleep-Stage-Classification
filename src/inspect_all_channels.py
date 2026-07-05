from pathlib import Path
import os
import tempfile
from collections import Counter
import pandas as pd
import mne

DATASETS = {
    "non_normal": Path(r"C:\Users\Dorsa Abbasi\Downloads\non-normal  sleep-disorder patients"),
    "healthy": Path(r"C:\Users\Dorsa Abbasi\Downloads\healthy subjects"),
}

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

rows = []
channel_counter = Counter()

for group_name, root in DATASETS.items():
    print(f"Checking {group_name}")

    if group_name == "healthy":
        subject_ids = range(1, 11)

        for subject_id in subject_ids:
            rec_path = root / f"{subject_id}.rec"
            channels = read_rec_channels(rec_path)

            for ch in channels:
                channel_counter[ch] += 1

            rows.append({
                "group": group_name,
                "subject": subject_id,
                "channels": ", ".join(channels),
            })

    else:
        subject_folders = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        subject_folders = sorted(subject_folders, key=lambda p: int(p.name))

        for subject_folder in subject_folders:
            subject_id = int(subject_folder.name)
            rec_path = subject_folder / f"{subject_id}.rec"
            channels = read_rec_channels(rec_path)

            for ch in channels:
                channel_counter[ch] += 1

            rows.append({
                "group": group_name,
                "subject": subject_id,
                "channels": ", ".join(channels),
            })

df = pd.DataFrame(rows)
df.to_csv("isruc_all_channels.csv", index=False, encoding="utf-8-sig")
df.to_excel("isruc_all_channels.xlsx", index=False)

print("\nSaved:")
print("isruc_all_channels.csv")
print("isruc_all_channels.xlsx")

print("\nChannel frequency:")
for ch, count in channel_counter.most_common():
    print(ch, count)