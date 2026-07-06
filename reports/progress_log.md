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

