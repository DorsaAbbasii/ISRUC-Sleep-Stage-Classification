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

## Week 2 — Tuned GBM Baseline



\### Planned

\- Tune the GBM model using the same clean 4-channel EEG feature set.

\- Compare the tuned GBM result with the previous Random Forest and GBM baselines.



\### Completed

\- Implemented hyperparameter tuning for HistGradientBoostingClassifier.

\- Tested different learning rates, iteration numbers, leaf nodes, and regularization values.

\- Saved tuning results and the best tuned GBM report.



\### Results



| Experiment | Model | Accuracy | Macro F1 |

|---|---|---:|---:|

| Clean 4-channel EEG baseline | Random Forest | 0.6660 | 0.6109 |

| Clean 4-channel EEG baseline | GBM / HistGradientBoosting | 0.6581 | 0.6355 |

| Clean 4-channel EEG baseline | Tuned GBM / HistGradientBoosting | 0.6614 | 0.6395 |



\### Conclusion

Tuned GBM achieved the best Macro F1 score so far. Random Forest still has slightly higher accuracy, but tuned GBM gives better balanced performance across sleep stages.



\### Next Step

\- Try a clean 6-channel EEG feature set.

\- Compare 4-channel and 6-channel results.

\## Week 3 — Fair 6-Channel Tuned GBM Comparison



\### Planned

\- Check whether adding frontal EEG channels improves the model.

\- Compare 4-channel and 6-channel models fairly on the same subject subset.



\### Completed

\- Checked availability of the 6 EEG channel families: F3, C3, O1, F4, C4, and O2.

\- Found that 109 out of 110 subjects have all required 6 channels.

\- Excluded non\_normal subject 8 because F3 and F4 were missing.

\- Trained a tuned GBM model on the 6-channel feature set.

\- Trained a tuned GBM model on the 4-channel feature set using the same 109 subjects.

\- Compared both models fairly.



\### Results



| Experiment | Model | Subjects | Accuracy | Macro F1 |

|---|---|---:|---:|---:|

| 4-channel same-109 | Tuned GBM | 109 | 0.6598 | 0.6373 |

| 6-channel same-109 | Tuned GBM | 109 | 0.6850 | 0.6603 |



\### Conclusion

Adding F3 and F4 improved the tuned GBM model. The 6-channel tuned GBM achieved the best result so far, with Accuracy = 0.6850 and Macro F1 = 0.6603.



\### Next Step

\- Analyze feature importance or permutation importance.

\- Identify which EEG channel families and feature types contribute most to the classification.

## Week 3 — Feature Importance Analysis

### Completed
- Performed permutation importance analysis for the best tuned 6-channel GBM model.
- Analyzed importance by individual feature, EEG channel, and feature type.

### Results
The most important individual features were:
- C3_delta_power
- C4_delta_power
- F3_beta_power
- O1_alpha_power
- O2_delta_power

The most important channels were:
- C3
- F3
- O2
- F4
- O1
- C4

The most important feature types were:
- delta_power
- alpha_power
- theta_power
- beta_power

### Conclusion
The best model mainly relies on frequency-domain EEG features. The added frontal channels, especially F3 and F4, contributed useful information, which supports the improvement observed in the 6-channel tuned GBM experiment.

### Next Step
- Run repeated subject-wise splits or GroupKFold cross-validation to check whether the best result is stable.
## Week 3 — Repeated Subject-wise Split Evaluation

### Completed
- Evaluated the best tuned 6-channel GBM model using 5 repeated subject-wise train/test splits.
- Used the same 6 EEG channel families: F3, C3, O1, F4, C4, and O2.
- Reported mean and standard deviation for Accuracy, Macro F1, and Weighted F1.

### Results

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.6992 | 0.0190 |
| Macro F1 | 0.6702 | 0.0158 |
| Weighted F1 | 0.7010 | 0.0175 |

### Conclusion
The tuned 6-channel GBM model showed stable performance across repeated subject-wise splits. This supports the reliability of the previous best result. The weakest class remains N1, which should be analyzed further in the next step.

### Next Step
- Perform error analysis using a confusion matrix.
- Investigate why N1 is harder to classify.

## Week 3 — Confusion Matrix and Error Analysis

### Completed
- Performed confusion matrix analysis for the tuned 6-channel GBM model.
- Used 5 repeated subject-wise train/test splits.
- Computed mean normalized confusion matrix and class-level metrics.

### Results

| Class | Mean F1 |
|---|---:|
| Wake | 0.8286 |
| N1 | 0.3895 |
| N2 | 0.6941 |
| N3 | 0.8220 |
| REM | 0.6170 |

The weakest class was N1.  
True N1 epochs were most often confused with:
- N2: 24.7%
- REM: 18.9%
- Wake: 15.1%

### Conclusion
The model performs well on Wake and N3, reasonably on N2, and moderately on REM. The main limitation is N1 classification. This is likely because N1 is a transitional sleep stage and can be difficult to separate from Wake, N2, and REM using simple epoch-level features.

### Next Step
- Improve the feature set to help the model distinguish N1 from neighboring stages.
- Add temporal/context features and additional EEG frequency features.

## Week 3 — Enhanced 6-Channel Feature Experiment

### Completed
- Created an enhanced 6-channel feature set based on the clean 6-channel EEG features.
- Added relative band power, band power ratios, left-right channel differences, and previous-epoch context features.
- Evaluated the enhanced feature set using the tuned GBM model and 5 repeated subject-wise train/test splits.

### Results

| Experiment | Accuracy Mean | Accuracy Std | Macro F1 Mean | Macro F1 Std |
|---|---:|---:|---:|---:|
| Original 6-channel tuned GBM | 0.6992 | 0.0190 | 0.6702 | 0.0158 |
| Enhanced 6-channel tuned GBM | 0.7131 | 0.0187 | 0.6865 | 0.0146 |

### Class-level result

| Class | Enhanced Mean F1 |
|---|---:|
| Wake | 0.8353 |
| N1 | 0.4193 |
| N2 | 0.7009 |
| N3 | 0.8295 |
| REM | 0.6478 |

### Conclusion
The enhanced feature set improved the tuned 6-channel GBM model. Accuracy increased from 0.6992 to 0.7131, and Macro F1 increased from 0.6702 to 0.6865. The N1 class also improved from about 0.3895 to about 0.4193, although N1 remains the weakest class.

### Next Step
- Analyze feature importance for the enhanced feature set.
- Check whether the new temporal/context features or ratio features contributed most to the improvement.S

## Week 3 — Enhanced Feature Importance Analysis

### Completed
- Performed permutation importance analysis for the enhanced 6-channel tuned GBM model.
- Analyzed importance by individual feature, feature group, channel, and feature type.

### Results

The most important feature groups were:

| Feature Group | Importance |
|---|---:|
| Previous-epoch context | 0.1588 |
| Original band power | 0.1226 |
| Relative band power | 0.0763 |
| Original statistical features | 0.0593 |
| Left-right channel differences | 0.0491 |
| Band power ratios | 0.0232 |

The most important channels were:

| Channel | Importance |
|---|---:|
| C3 | 0.1013 |
| O1 | 0.0879 |
| O2 | 0.0841 |
| F3 | 0.0783 |
| F4 | 0.0719 |
| C4 | 0.0658 |

### Conclusion
The enhanced model benefited most from previous-epoch context features. This suggests that temporal continuity between neighboring sleep epochs is useful for sleep-stage classification. Original band power and relative band power features also remained important. This supports the idea that the improvement of the enhanced model came mainly from temporal context and frequency-domain EEG information.

### Next Step
- Run confusion matrix analysis for the enhanced model.
- Compare whether N1 confusion decreased compared with the original 6-channel GBM model.

## Week 3 — Enhanced Confusion Matrix and Error Analysis

### Completed
- Performed confusion matrix analysis for the enhanced 6-channel tuned GBM model.
- Used 5 repeated subject-wise train/test splits.
- Compared the enhanced model with the original 6-channel GBM model.

### Results

| Metric | Original 6-channel GBM | Enhanced 6-channel GBM | Improvement |
|---|---:|---:|---:|
| Accuracy | 0.6992 | 0.7131 | +0.0139 |
| Macro F1 | 0.6702 | 0.6865 | +0.0163 |
| N1 F1 | 0.3895 | 0.4193 | +0.0298 |

### Class-level Results

| Class | Mean F1 |
|---|---:|
| Wake | 0.8353 |
| N1 | 0.4193 |
| N2 | 0.7009 |
| N3 | 0.8295 |
| REM | 0.6478 |

### Main Error Pattern
The weakest class is still N1.

True N1 epochs were most often confused with:
- N2: 24.2%
- REM: 17.3%
- Wake: 15.5%

### Conclusion
The enhanced feature set improved the overall model performance and increased N1 F1 from 0.3895 to 0.4193. However, N1 remains the most difficult class. This suggests that temporal/context features help, but additional methods may still be needed to better separate N1 from N2, REM, and Wake.

### Next Step
- Stop adding new experiments temporarily.
- Update README with the final result table.
- Prepare a short final project summary for supervisor/practice report.