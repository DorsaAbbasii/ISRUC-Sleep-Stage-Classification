\# Progress Log



\## Week 1 — Dataset Preparation and Baselines



\### Planned

\- Find a suitable open dataset for sleep stage classification.

\- Check whether the dataset can be used for EEG-based experiments.

\- Build the first baseline models.



\### Completed

\- Selected the ISRUC-SLEEP dataset.

\- Checked dataset structure and readability using Python.

\- Confirmed that all recordings and annotation files are available.

\- Analyzed sleep-stage label distribution.

\- Inspected channel names and standardized EEG channel aliases.

\- Built a clean C3 baseline.

\- Built a clean 4-channel EEG baseline using C3, C4, O1, and O2.

\- Uploaded the current project to GitHub.



\### Results

| Experiment | Model | Accuracy | Macro F1 |

|---|---|---:|---:|

| Clean C3 baseline | Random Forest | 0.6451 | 0.5945 |

| Clean 4-channel EEG baseline | Random Forest | 0.6660 | 0.6109 |



\### Problems / Limitations

\- The same EEG channels have different names in different recordings.

\- This was handled by creating channel families such as C3-M2 / C3-A2 / C3.

\- N1 is still the weakest class.

\- Current models use handcrafted statistical and frequency-domain features.



\### Next Step

\- Train a GBM / Gradient Boosting model on the same 4-channel feature set.

\- Compare GBM with the current Random Forest baseline.

# Progress Log



\## Week 1 — Dataset Preparation and Baselines



\### Planned

\- Find a suitable open dataset for sleep stage classification.

\- Check whether the dataset can be used for EEG-based experiments.

\- Build the first baseline models.



\### Completed

\- Selected the ISRUC-SLEEP dataset.

\- Checked dataset structure and readability using Python.

\- Confirmed that all recordings and annotation files are available.

\- Analyzed sleep-stage label distribution.

\- Inspected channel names and standardized EEG channel aliases.

\- Built a clean C3 baseline.

\- Built a clean 4-channel EEG baseline using C3, C4, O1, and O2.

\- Uploaded the current project to GitHub.



\### Results

| Experiment | Model | Accuracy | Macro F1 |

|---|---|---:|---:|

| Clean C3 baseline | Random Forest | 0.6451 | 0.5945 |

| Clean 4-channel EEG baseline | Random Forest | 0.6660 | 0.6109 |



\### Problems / Limitations

\- The same EEG channels have different names in different recordings.

\- This was handled by creating channel families such as C3-M2 / C3-A2 / C3.

\- N1 is still the weakest class.

\- Current models use handcrafted statistical and frequency-domain features.



\### Next Step

\- Train a GBM / Gradient Boosting model on the same 4-channel feature set.

\- Compare GBM with the current Random Forest baseline.

## Week 2 — GBM Baseline



\### Planned

\- Train a GBM / Gradient Boosting model using the same clean 4-channel EEG feature set.

\- Compare the GBM result with the Random Forest baseline.



\### Completed

\- Implemented a GBM baseline using HistGradientBoostingClassifier.

\- Used the same 4-channel EEG features and the same subject-wise train/test split.

\- Compared GBM with the previous Random Forest baseline.



\### Results



| Experiment | Model | Accuracy | Macro F1 |

|---|---|---:|---:|

| Clean 4-channel EEG baseline | Random Forest | 0.6660 | 0.6109 |

| Clean 4-channel EEG baseline | GBM / HistGradientBoosting | 0.6581 | 0.6355 |



\### Conclusion

GBM achieved slightly lower accuracy than Random Forest, but improved Macro F1. This means the model became better balanced across classes. The improvement is especially visible for the difficult N1 class.



\### Next Step

\- Tune GBM hyperparameters.

\- Try the same approach with a 6-channel EEG feature set.

