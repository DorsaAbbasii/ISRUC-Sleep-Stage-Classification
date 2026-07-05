\# ISRUC Sleep Stage Classification



This project focuses on automatic sleep stage classification using EEG signals from the ISRUC-SLEEP dataset.



The current work includes:



\- dataset quality checking

\- label distribution analysis

\- channel availability inspection

\- clean single-channel EEG baseline

\- clean multi-channel EEG baseline

\- Random Forest classification with subject-wise train/test split



\## Dataset



The project uses the ISRUC-SLEEP dataset.



Raw dataset files are not included in this repository because they are large biomedical data files.



The experiments use 30-second sleep epochs with five sleep-stage labels:



| Label | Sleep Stage |

|---:|---|

| 0 | Wake |

| 1 | N1 |

| 2 | N2 |

| 3 | N3 |

| 5 | REM |



\## Dataset Source and Citation



This project uses the ISRUC-SLEEP Dataset, a public polysomnography dataset for sleep research.



Official dataset website:

https://sleeptight.isr.uc.pt/



The raw dataset files are not included in this repository.



If you use this dataset, please cite the original ISRUC-Sleep publication:



Khalighi, S., Sousa, T., Santos, J. M., \& Nunes, U. (2016).  

ISRUC-Sleep: A comprehensive public dataset for sleep researchers.  

Computer Methods and Programs in Biomedicine, 124, 180–192.  

DOI: 10.1016/j.cmpb.2015.10.013

\## Current Results



| Experiment | Accuracy | Macro F1 |

|---|---:|---:|

| Clean C3 baseline | 0.6451 | 0.5945 |

| Clean 4-channel EEG baseline | 0.6660 | 0.6109 |



The 4-channel EEG baseline improved the result compared with the single-channel C3 baseline.



\## Method



The current baseline uses:



\- 30-second EEG epochs

\- statistical features

\- frequency-domain features

\- Random Forest classifier

\- subject-wise train/test split



Subject-wise splitting is used to avoid data leakage between training and testing.



\## Channel Standardization



The dataset uses different names for the same EEG channels.  

For example, the C3 channel appears as:



\- C3-M2

\- C3-A2

\- C3



For the 4-channel baseline, the following channel families were used:



\- C3

\- C4

\- O1

\- O2



\## Repository Structure



```text

ISRUC-Sleep-Stage-Classification/

│

├── src/

│   ├── count\_isruc\_labels.py

│   ├── inspect\_all\_channels.py

│   ├── train\_isruc\_baseline\_clean\_c3.py

│   └── train\_isruc\_baseline\_clean\_4ch.py

│

├── results/

│   ├── isruc\_clean\_C3\_baseline\_report.txt

│   ├── isruc\_clean\_4ch\_baseline\_report.txt

│   ├── isruc\_clean\_C3\_channels\_used.csv

│   ├── isruc\_clean\_4ch\_channels\_used.csv

│   ├── isruc\_label\_counts.csv

│   └── isruc\_all\_channels.csv

│

├── reports/

├── README.md

├── requirements.txt

└── .gitignore

