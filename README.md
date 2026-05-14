# Stroke prediction and analysis

This repository contains a machine learning workflow for stroke-related data: exploratory views, several classifiers for comparison, and a **Streamlit** web app for interactive prediction using a tuned **XGBoost** model.

This project is for **education and demonstration only**. It is **not** medical advice, diagnosis, or a substitute for a qualified clinician.

## Features

- **Prediction** — Enter patient-style fields (demographics, vitals, lifestyle). On **Submit**, the app loads the XGBoost model and shows a risk-style prediction and probability.
- **Model Details** — Load `train.csv`, show charts (Plotly), encoding, SMOTE class balance, correlation views, and optional re-training of comparison models inside the app.

The serialized model lives at `Models/XGBoostTunedModel.pkl`. If it is missing, the app can recreate it using `train_save_model.py` (or you can run that script manually).

## Requirements

- Python **3.9+** recommended (3.10 or 3.11 work well with current `scikit-learn` / `xgboost` wheels).
- Windows, macOS, or Linux.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/rajendrasarswat/Stroke_Prediction2.git
cd Stroke_Prediction2
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `scikit-learn` fails to resolve, install it explicitly:

```bash
pip install scikit-learn
```

### 4. (Optional) Regenerate the saved model

Only needed if you delete `Models/XGBoostTunedModel.pkl` or want to retrain from `train.csv`:

```bash
python train_save_model.py
```

### 5. Run the Streamlit app

Always run from the project directory so data and model paths resolve correctly:

```bash
streamlit run web_app.py
```

Then open the URL shown in the terminal (by default `http://localhost:8501`).

## Repository layout

| Path | Description |
|------|-------------|
| `web_app.py` | Streamlit entrypoint |
| `train_save_model.py` | Trains XGBoost and writes `Models/XGBoostTunedModel.pkl` |
| `Models/XGBoostTunedModel.pkl` | Serialized prediction model |
| `train.csv` / `test.csv` | Dataset files used by the notebook and app |
| `Stroke Prediction.ipynb` | Original analysis and training notebook |
| `requirements.txt` | Python dependencies |

## License and attribution

Add a license file if you intend open-source redistribution. Until then, assume **all rights reserved** unless you state otherwise in the repository settings.

## Contributing

Issues and pull requests are welcome. Please keep changes focused and document any new environment variables or data files the app expects.
