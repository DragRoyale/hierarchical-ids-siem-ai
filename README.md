# Hierarchical Intrusion Detection under Class Imbalance
### A Two-Stage Framework for Scalable and Robust SIEM Analytics

## 📌 Overview

This repository contains the implementation of a **two-stage hierarchical intrusion detection system (IDS)** designed to address:

- Severe class imbalance in multi-class cybersecurity data  
- Degraded macro-F1 performance for minority attack classes  
- Computational scalability in high-volume SIEM environments  

The proposed framework decomposes intrusion detection into:

1. **Stage 1 (Binary Classification):**
   - Distinguishes **Benign vs Attack**
2. **Stage 2 (Multi-Class Classification):**
   - Classifies only detected attacks into specific categories

This design enables **conditional execution**, improving both:
- classification performance
- inference efficiency

---

## 🧠 Key Contributions
- ✔ Two-stage hierarchical IDS architecture  
- ✔ Improved macro-F1 under imbalanced conditions  
- ✔ Statistical validation (paired t-test + Cohen’s d)  
- ✔ Runtime benchmarking (up to **2.38× throughput improvement**)  
- ✔ Evaluation across multiple datasets:
  - UNSW-NB15  
  - CICIDS2017  
  - SIEM-like dataset  

---

## 📊 Results Summary
| Dataset | Stage | Metric | Result |
|--------|------|--------|--------|
| UNSW-NB15 | Cross-validation | Macro-F1 | ↑ Improvement |
| SIEM-like | 25 runs | Macro-F1 | 0.8488 → **0.8752** |
| CICIDS2017 | Stage 1 | Macro-F1 | **0.9887** |
| CICIDS2017 | Stage 2 | Macro-F1 | **0.8163** |
| Runtime | Throughput | Speed | up to **2.38× faster** |

---

## ⚙️ Project Structure
siem-ai-q1/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── extract_unsw_per_class.py
│   │   ├── check.py
│   │
│   ├── training/
│   │   ├── train_catboost_cv.py
│   │   ├── train_cicids_gpu.py
│   │   ├── train_cicids_stage2.py
│   │   ├── train_siem_single_stage.py
│   │   ├── train_siem_repeated.py
│   │
│   ├── evaluation/
│   │   ├── stat_test_unsw.py
│   │   ├── stat_test_by_stage.py
│   │   ├── stat_siem_significance.py
│   │   ├── per_class_analysis.py
│   │
│   ├── visualization/
│   │   ├── plot_unsw_cv_mean.py
│   │   ├── plot_unsw_cv_learning_curve.py
│   │
│   ├── utils/
│   │   ├── cpu.py
│
├── results/
│   ├── figures/
│   │   ├── binary_class_confusion.png
│   │   ├── roc_curve.png
│   │   ├── confusion_matrix_stage2.png
│   │   ├── loss.png
│   │   ├── by_epoch.png
│   │
│   ├── tables/
│   │   ├── unsw_cv_results.csv
│   │   ├── runtime_benchmark.csv
│   │
│   ├── arrays/
│       ├── *.npy
│
├── models/   ❗ (NEW FOLDER)
│   ├── *.joblib
│
├── notebooks/ (optional)
│
└── paper/
    ├── figures/ (copy final figures used in paper)

▶️ How to Run
1. Train Stage 1 (Binary)
python src/train_stage1.py
2. Train Stage 2 (Multi-class)
python src/train_stage2.py
📂 Datasets

This project uses the following datasets:

UNSW-NB15
CICIDS2017
SIEM-like dataset (private)

⚠️ Due to size and privacy constraints, datasets are not included.
Download links:
UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html

📄 Paper
This repository accompanies the research paper:
Hierarchical Intrusion Detection under Class Imbalance: A Two-Stage Framework for Scalable and Robust SIEM Analytics
