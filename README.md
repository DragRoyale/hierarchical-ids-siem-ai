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
| Dataset | Setting | Model | Macro-F1 |
|--------|--------|------|--------|
| UNSW-NB15 | Cross-validation | Single-stage | 0.5958 |
| UNSW-NB15 | Cross-validation | Two-stage | **0.6476** |
| UNSW-NB15 | Official split | Single-stage | **0.2397** |
| UNSW-NB15 | Official split | Two-stage | 0.1581 |
| SIEM-like | 25 runs | Single-stage | 0.8488 |
| SIEM-like | 25 runs | Two-stage | **0.8752** |
| CICIDS2017 | Stage 1 (Binary) | — | **0.9887** |
| CICIDS2017 | Stage 2 (Multi-class) | — | **0.8163** |
| Runtime | Throughput | Two-stage vs Single | **2.38× faster** |

---

## ⚙️ Project Structure
src/ → core implementation

results/ → figures and outputs used in paper

models/ → trained models

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
