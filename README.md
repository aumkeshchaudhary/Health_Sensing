 # 🫁 Health Sensing: Sleep Analysis & Breathing Disorder Detection

<div align="center">

**Deep Learning for Sleep Health Analytics**

*Automated detection of breathing disorders and sleep stage classification using physiological signals*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#-overview) • [Features](#-key-features) • [Dataset](#-dataset) • [Models](#-models) • [Results](#-results) • [Quick Start](#-quick-start)

</div>

---

## 🎯 Overview

This project builds an end-to-end system for analyzing multi-modal physiological signals collected during overnight sleep sessions.
It includes:

- High-quality signal visualization

- Noise cleaning & filtering

- Dataset creation (windowing + label assignment)

- Model training for:

  - Breathing Disorder Detection (Hypopnea / Obstructive Apnea / Normal)

  - Sleep Stage Classification (Wake, REM, N1, N2, N3)

- Evaluation using Leave-One-Participant-Out (LOSO) cross-validation

- Export of results, metrics, and confusion matrices

The project is fully modular, script-based, and matches the requirements given in the task.
---

## ✨ Key Features

### 🩺 Signal Processing
- Reads raw nasal airflow, thoracic movement, and SpO₂ signals
- Aligns them using timestamps
- Applies band-pass filtering to remove high-frequency noise
- Handles missing values, timestamp mismatches, and interpolation

### 📊 Visualization
- Plots 8-hour recordings per participant
- Overlays flow-event annotations (e.g., Apnea, Hypopnea)
- Automatically saves PDFs inside Visualizations/
- Script: vis.py
- 
### 🧪 Dataset Creation
- Splits signals into 30-second windows with 50% overlap
- Matches windows to labeled events
- Assigns Normal, Hypopnea, or Obstructive Apnea
- Saves windowed dataset in .npz and .csv format
- Script: create_dataset.py

### 🧠 Modeling

Models implemented:

| Task                         | CNN | Conv-LSTM | Transformer |
| ---------------------------- | --- | --------- | ----------- |
| Breathing Disorder Detection | ✅   | ✅         | —          |
| Sleep Stage Classification   | ✅   | ✅         | ✅         |

- All trained using LOSO Cross-Validation
- Automatically generates:
  - Confusion matrices
  - Metrics CSVs
  - Per-fold and aggregated results

- Scripts:
 - train_breathing_model.py
 - train_sleep_model.py

---

## 🏗️ Project Structure

            HealthSensingProject/
            │
            ├── Data/                          # Raw signals (not tracked in GitHub; large files)
            │   ├── AP01/
            │   ├── AP02/
            │   ├── AP03/
            │   ├── AP04/
            │   └── AP05/
            │
            ├── Dataset/                       # Generated datasets (ignored in GitHub)
            │   ├── breathing_windows.npz
            │   ├── breathing_labels.csv
            │   ├── sleep_windows.npz
            │   └── sleep_labels.csv
            │
            ├── scripts/                       # All executable Python scripts
            │   ├── vis.py                     # Visualization script for 8-hour plots
            │   ├── create_breathing_dataset.py
            │   ├── create_sleep_dataset.py
            │   ├── train_breathing_model.py
            │   └── train_sleep_model.py
            │
            ├── models/                        # Deep learning architectures
            │   ├── cnn_model.py               # 1D CNN
            │   ├── conv_lstm_model.py         # Conv-LSTM
            │   └── transformer_model.py       # Transformer for sleep staging
            │
            ├── utils/                         # Helper utilities
            │   ├── filtering.py               # Signal cleaning filters
            │   ├── metrics.py                 # Per-class metrics calculation
            │   └── helpers.py                 # Common functions (if any)
            │
            ├── Visualizations/                # PDF plots for each participant
            │   ├── AP01_visualization.pdf
            │   ├── AP02_visualization.pdf
            │   └── ...
            │
            ├── breathing_results/             # LOSO results for breathing task
            │   ├── results_cnn_metrics.csv
            │   ├── results_conv_lstm_metrics.csv
            │   └── confusion matrices (if saved)
            │
            ├── sleep_results/                 # LOSO results for sleep stage task
            │   ├── results_cnn_sleep_metrics.csv
            │   ├── results_conv_lstm_sleep_metrics.csv
            │   ├── results_transformer_sleep_metrics.csv
            │   └── confusion matrices (PNG)
            │
            ├── .gitignore                     # Ignores Data/ and Dataset/ folders
            ├── .gitattributes                 # Git LFS configuration
            ├── requirements.txt               # Python dependencies
            └── README.md                      # Full project documentation

---

## 🧠 Models & Performance

### Task 1 — Breathing Disorder Detection

Classes:

- Normal

- Hypopnea

- Obstructive Apnea

#### Key Result:
The dataset is extremely imbalanced → models learn to predict Normal.

| Model     | Avg Accuracy | Normal Recall | Hypopnea Recall | OA Recall |
| --------- | ------------ | ------------- | --------------- | --------- |
| CNN       | ~91.7%       | 100%          | 0%              | 0%        |
| Conv-LSTM | ~91.5%       | 100%          | 0%              | 0%        |

   ✔ High accuracy
   ✘ Poor clinical performance for minority classes
   ✔ Needs re-sampling or class-weighted loss

---
### Task 2 — Sleep Stage Classification

Classes: 
- Wake
- REM
- N1 
- N2
- N3

#### Performance (LOSO):

| Model       | Avg Accuracy | Notes                                     |
| ----------- | ------------ | ----------------------------------------- |
| CNN         | ~44–45%      | Strong N2 recall, poor N3                 |
| Conv-LSTM   | ~52%         | Best overall, captures temporal structure |
| Transformer | ~42%         | Requires tuning                           |

Sleep staging is much harder due to class imbalance + low inter-class separability.

---
### 🧪 Evaluation Strategy

LOSO (Leave-One-Participant-Out):

- Train on 4 participants
- Test on the remaining one
- Repeat for all 5
- Avoids **data leakage** when dealing with personalized physiological data
- More realistic than random splits

Metrics computed per class:

- Accuracy
- Precision
- Recall
- Sensitivity
- Specificity
- Confusion Matrix


---
## 🚀 Quick Start

### Install Dependencies

    pip install -r requirements.txt


### Create Dataset

    python scripts/create_breathing_dataset.py -in_dir Data -out_dir Dataset
    python scripts/create_sleep_dataset.py -in_dir Data -out_dir Dataset
    
### Visualize Participant

    python vis.py -name Data/AP01


### Train Models (LOSO Cross-Validation)
   
   Breathing: 
    
    python scripts/train_breathing_model.py -model cnn

   Sleep:

    python scripts/train_sleep_model.py -model conv_lstm
    
---

## 📊 Outputs

This project automatically generates:

- Visualizations/*.pdf
- breathing_results/*.csv
- sleep_results/*.csv
- confusion matrices
- Aggregate metrics across folds

Everything is stored cleanly and consistently.

---
## 🔍 Limitations & Future Work

- Extreme class imbalance limits detection of apnea/hypopnea
- Sleep stage classification requires more data
- Transformer needs hyperparameter tuning
- No data augmentation yet
- Could integrate frequency-domain features


## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---


<div align="center">

</div>
