from pathlib import Path
from collections import Counter
import pandas as pd

DATASETS = {
    "non_normal": Path(r"C:\Users\Dorsa Abbasi\Downloads\non-normal  sleep-disorder patients"),
    "healthy": Path(r"C:\Users\Dorsa Abbasi\Downloads\healthy subjects"),
}

def read_labels(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]

rows = []
total_counter = Counter()

for group_name, root in DATASETS.items():
    if group_name == "healthy":
        subject_ids = range(1, 11)
        for subject_id in subject_ids:
            ann_path = root / f"{subject_id}_1.txt"
            labels = read_labels(ann_path)
            counter = Counter(labels)
            total_counter.update(counter)

            row = {"group": group_name, "subject": subject_id, "total_labels": len(labels)}
            for label, count in sorted(counter.items()):
                row[f"label_{label}"] = count
            rows.append(row)

    else:
        subject_folders = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        subject_folders = sorted(subject_folders, key=lambda p: int(p.name))

        for subject_folder in subject_folders:
            subject_id = int(subject_folder.name)
            ann_path = subject_folder / f"{subject_id}_1.txt"
            labels = read_labels(ann_path)
            counter = Counter(labels)
            total_counter.update(counter)

            row = {"group": group_name, "subject": subject_id, "total_labels": len(labels)}
            for label, count in sorted(counter.items()):
                row[f"label_{label}"] = count
            rows.append(row)

df = pd.DataFrame(rows).fillna(0)
df.to_csv("isruc_label_counts.csv", index=False, encoding="utf-8-sig")
df.to_excel("isruc_label_counts.xlsx", index=False)

print("Saved:")
print("isruc_label_counts.csv")
print("isruc_label_counts.xlsx")

print("\nTotal label counts:")
for label, count in sorted(total_counter.items()):
    print(label, count)