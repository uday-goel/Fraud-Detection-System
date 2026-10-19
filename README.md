# Financial Fraud Detection

**Goal:** Detect fraudulent transactions in financial data.

**Guidelines followed**
- Anomaly detection / classification: **Isolation Forest** and **Autoencoder**
- Class imbalance handled with **SMOTE**
- Performance measured with **F1-score** and **AUC-ROC** (plus accuracy, precision, recall, PR-AUC, MAE, MSE, RMSE, R²)

## Dataset
Dataset Link = "https://www.kaggle.com/datasets/ealaxi/paysim1"
PaySim synthetic mobile-money transactions (`data/PS_20174392719_1491204439457_log.csv`):
6,362,620 transactions, 8,213 frauds (0.13%). Fraud occurs **only** in `TRANSFER` and `CASH_OUT`
transactions, so the pipeline keeps those two types (2,770,409 rows, 0.296% fraud).

## Pipeline (`fraud_detection.py`)
1. Load data
2. Preprocessing and feature engineering (balance-consistency errors, hour of day, flags; ID columns and the leaky `isFlaggedFraud` dropped)
3. EDA (7 figures saved to `outputs/figures/`)
4. Stratified split: 70% train / 10% validation / 20% test
5. Scaling (signed log1p + StandardScaler, fitted on train only)
6. **Isolation Forest** - unsupervised anomaly detection
7. **Autoencoder** (Keras) - trained on legitimate transactions only; reconstruction error = anomaly score
8. **SMOTE + Random Forest** - SMOTE-NC applied to the training split only
9. Evaluation on the untouched test set, comparison table and plots

Decision thresholds for the anomaly scores are tuned on the **validation** set (max F1); the test set is used once for final reporting.

## Results (test set, 554,082 transactions, 1,643 frauds)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | PR-AUC | MAE | MSE | R² |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.9985 | 0.9784 | 0.4967 | 0.6589 | 0.9576 | 0.5183 | 0.001525 | 0.001525 | 0.4842 |
| Autoencoder | 0.9990 | 0.7826 | 0.9288 | 0.8494 | 0.9994 | 0.9238 | 0.000976 | 0.000976 | 0.6697 |
| SMOTE + Random Forest | 0.99998 | 1.0000 | 0.9921 | 0.9960 | 0.9990 | 0.9981 | 0.000023 | 0.000023 | 0.9921 |

Accuracy, MAE, MSE and R² are computed on the 0/1 predictions. With a 0.3% fraud rate they look great
even for weak models, so judge the models on F1, precision/recall, AUC-ROC and PR-AUC.
Full log: `outputs/run_log.txt`, table: `outputs/model_comparison.csv`.

## Run
```bash
pip install -r requirements.txt
python fraud_detection.py                     # full dataset (~3 min on a laptop CPU)
python fraud_detection.py --sample-frac 0.1   # quick run on a 10% stratified sample
```
Options: `--data-path`, `--sample-frac`, `--epochs`, `--output-dir`.

## Pushing to GitHub (the dataset is > 100 MB)
```bash
git lfs install
git init && git add . && git commit -m "Fraud detection project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
`.gitattributes` already tracks `data/*.csv` with Git LFS. Without LFS, GitHub will reject the push.
