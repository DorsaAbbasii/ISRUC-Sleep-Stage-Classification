# Final Project Summary — ISRUC Sleep Stage Classification

## Project Goal

The goal of this project is to build a reproducible machine learning pipeline for automatic sleep-stage classification from EEG signals using the ISRUC-SLEEP dataset.

The task is to classify each 30-second EEG epoch into one of five sleep stages:

- Wake
- N1
- N2
- N3
- REM

## Dataset

The ISRUC-SLEEP dataset was used. The project includes healthy subjects and sleep-disorder subjects.

A dataset completeness check was performed before modeling. The recordings were readable, annotations were available, and the dataset was suitable for baseline experiments.

For the strict 6-channel EEG setup, 109 out of 110 subjects had all required EEG channel families:

- F3
- C3
- O1
- F4
- C4
- O2

One subject was excluded from the strict 6-channel experiments because F3 and F4 were missing.

## Methods

The project followed this pipeline:

1. Dataset quality checking
2. EEG feature extraction
3. Random Forest baseline models
4. GBM / HistGradientBoosting models
5. Hyperparameter tuning
6. Fair 4-channel vs 6-channel comparison
7. Repeated subject-wise split evaluation
8. Feature importance analysis
9. Confusion matrix and error analysis
10. Enhanced feature engineering

## Main Experiments

| Experiment | Subjects | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Clean C3 Random Forest baseline | 110 | 0.6451 | 0.5945 |
| Clean 4-channel Random Forest baseline | 110 | 0.6660 | 0.6109 |
| Clean 4-channel GBM baseline | 110 | 0.6581 | 0.6355 |
| Clean 4-channel tuned GBM | 110 | 0.6614 | 0.6395 |
| Clean 4-channel tuned GBM, same-109 | 109 | 0.6598 | 0.6373 |
| Clean 6-channel tuned GBM, same-109 | 109 | 0.6850 | 0.6603 |
| Clean 6-channel tuned GBM, repeated splits | 109 | 0.6992 ± 0.0190 | 0.6702 ± 0.0158 |
| Enhanced 6-channel tuned GBM, repeated splits | 109 | 0.7131 ± 0.0187 | 0.6865 ± 0.0146 |

## Best Result

The best result was achieved by the enhanced 6-channel tuned GBM model.

Final repeated-split performance:

- Accuracy: 0.7131 ± 0.0187
- Macro F1: 0.6865 ± 0.0146
- Weighted F1: 0.7138 ± 0.0180

## Enhanced Features

The enhanced feature set added:

- Relative band power
- Band power ratios
- Left-right channel differences
- Previous-epoch context features

The enhanced model improved over the original 6-channel GBM:

| Metric | Original 6-channel GBM | Enhanced 6-channel GBM | Improvement |
|---|---:|---:|---:|
| Accuracy | 0.6992 | 0.7131 | +0.0139 |
| Macro F1 | 0.6702 | 0.6865 | +0.0163 |
| N1 F1 | 0.3895 | 0.4193 | +0.0298 |

## Feature Importance Findings

Feature importance analysis showed that the enhanced model benefited most from:

1. Previous-epoch context features
2. Original band power features
3. Relative band power features
4. Original statistical features
5. Left-right channel differences
6. Band power ratios

This suggests that temporal continuity between neighboring sleep epochs is useful for sleep-stage classification.

## Error Analysis

The main weakness of the model is the N1 class.

For the enhanced 6-channel GBM model:

| Class | Mean F1 |
|---|---:|
| Wake | 0.8353 |
| N1 | 0.4193 |
| N2 | 0.7009 |
| N3 | 0.8295 |
| REM | 0.6478 |

N1 was most often confused with:

- N2: 24.2%
- REM: 17.3%
- Wake: 15.5%

This shows that N1 remains the most difficult class, probably because it is a transitional sleep stage.

## Conclusion

This project built a reproducible EEG-based sleep-stage classification pipeline using the ISRUC-SLEEP dataset.

The experiments showed that:

- Multi-channel EEG improves performance compared with a single-channel baseline.
- Tuned GBM performs better than the initial Random Forest baseline.
- Repeated subject-wise evaluation confirms that model performance is stable.
- Enhanced temporal and frequency-domain features improve the model.
- N1 remains the most difficult sleep stage to classify.

The final enhanced 6-channel tuned GBM model achieved the best performance and provides a strong baseline for future thesis work.

## Future Work

Possible next steps:

- Add stronger temporal modeling.
- Test sequence models or deep learning models.
- Add more EEG features such as spectral entropy and sigma power.
- Compare with another sleep dataset.
- Improve N1 classification.
