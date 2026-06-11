# Support Integrity Auditor (SIA)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-98.75%25-green?style=flat-square)
![F1](https://img.shields.io/badge/F1_Score-0.987-green?style=flat-square)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat-square)](https://si-auditor.streamlit.app/)
Automatically detects priority mismatches in CRM support tickets and generates evidence-backed reports.

---

## The Problem

Human agents assign priority labels (Low / Medium / High / Critical) to support tickets every day. They make mistakes:

| Problem | What Happens | Impact |
|:---|:---|:---|
| Agent Fatigue | Tired agent labels everything Low | Critical issues get ignored |
| Customer Favoritism | VIP customer gets Critical for minor issue | Real emergencies stuck in queue |
| Keyword Anchoring | Sees "urgent" and labels Critical without reading | Engineers waste time on trivial issues |
| Volume Overload | Too many tickets to read carefully | Random labeling errors |

Two types of mismatch occur:

- **Hidden Crisis** — ticket is more serious than labeled (dangerous)
- **False Alarm** — ticket is less serious than labeled (wastes resources)

---

## How It Works

SIA runs a 3-stage self-supervised pipeline. It requires zero pre-labeled mismatch data — it creates its own training labels from raw ticket data.

### Stage 1 — Pseudo-Label Generation

Two independent signals are fused to infer the true severity of each ticket, completely ignoring the human label.

**Signal A: NLP Keyword Scoring (60% weight)**

| Severity | Keywords |
|:---|:---|
| Critical | `production down` `all users` `breach` `emergency` `revenue loss` `immediately` |
| High | `billing error` `degraded` `timeout` `incorrect data` `integration broken` |
| Medium | `minor` `cosmetic` `display` `ui` `misaligned` |
| Low | `typo` `no rush` `spelling` `feature request` `nice to have` |

**Signal B: Resolution Time (40% weight)**

| Time | Implies |
|:---|:---|
| 0 – 8 hours | Team treated it as urgent |
| 8 – 48 hours | Moderate urgency |
| 48 – 120 hours | Normal processing |
| 120+ hours | Low urgency |

**Fusion:**

```
Fused Score = 0.60 x NLP_score + 0.40 x RT_score

Score >= 3.2  ->  Critical
Score >= 2.4  ->  High
Score >= 1.6  ->  Medium
Score <  1.6  ->  Low

Mismatch Label = 1 if Inferred != Assigned, else 0
```

### Stage 2 — Classifier Training

| Component | Details |
|:---|:---|
| Text Features | TF-IDF — 5,000 features, 1–3 grams |
| Structured Features | Resolution time, domain tier, channel, ticket type |
| Interaction Features | `nlp_score - assigned_priority` |
| Class Imbalance | SMOTE oversampling |
| Model | Logistic Regression (65%) + Random Forest (35%) |
| Threshold | Tuned on macro-F1 — optimal at 0.40 |

### Stage 3 — Evidence Dossier

For every flagged ticket, a structured JSON report is generated containing keywords found, resolution time interpretation, and a 2–3 sentence grounded explanation. Every evidence item is traceable to the input ticket. No hallucination.

---

## Results

| Metric | Result | Threshold | Status |
|:---|:---:|:---:|:---:|
| Accuracy | **98.75%** | >= 83% | PASS |
| Macro F1 | **0.9874** | >= 0.82 | PASS |
| Recall — Consistent | **0.9747** | >= 0.78 | PASS |
| Recall — Mismatch | **0.9988** | >= 0.78 | PASS |

**Confusion Matrix (test set, n = 1,600)**

| | Predicted Consistent | Predicted Mismatch |
|:---|:---:|:---:|
| True Consistent | 733 | 19 |
| True Mismatch | 1 | 847 |

Only 1 missed mismatch and 19 false positives out of 1,600 test tickets.

**Batch results on 20,000 real tickets:**

| Category | Count | % |
|:---|:---:|:---:|
| Consistent | 6,749 | 33.7% |
| Hidden Crisis | 8,852 | 44.3% |
| False Alarm | 4,029 | 20.2% |
| **Total Flagged** | **13,251** | **66.3%** |

---

## Ablation Study

| Signal | Mismatch Rate | Notes |
|:---|:---:|:---|
| NLP Only | 54.9% | Strong but misses behavioral signals |
| Resolution Time Only | 41.5% | Weaker — text is more informative |
| **Fused (NLP 60% + RT 40%)** | **53.0%** | Most balanced and reliable |
| Signal Agreement | 0.513 | Confirms signals are independent |

The 0.513 pairwise agreement confirms the two signals are independent — combining them gives a more reliable result than either alone. NLP is weighted higher (60%) because text is the most direct severity indicator.

---

## Quick Start

### Requirements

- Python 3.9+
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOURUSERNAME/support-integrity-auditor.git
cd support-integrity-auditor
```

### Step 2 — Create and activate virtual environment

```bash
python -m venv sia_env

# Windows
sia_env\Scripts\activate

# Mac / Linux
source sia_env/bin/activate
```

Your terminal will show `(sia_env)` once activated. Keep it active for all commands below.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the dataset

1. Download from [kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset)
2. Rename the file to `tickets.csv`
3. Place it inside the project folder

### Step 5 — Train the model

```bash
python train_pipeline.py
```

Expected output at the end:

```
PASS  accuracy >= 83%
PASS  macro_F1 >= 0.82
PASS  recall_both >= 0.78
```

This creates two folders automatically — `sia_model/` and `outputs/`.

### Step 6 — Launch the web app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

### Step 7 — Run inference on a CSV (optional)

```bash
python predict.py --input tickets.csv --output ./results
```

> Always run `train_pipeline.py` before `app.py` or `predict.py`.

---

## Web App

**Single Ticket Mode** — fill in ticket details, click Analyze Ticket, get instant verdict with a full evidence dossier.

**Batch CSV Mode** — upload `tickets.csv` and get:
- Mismatch type distribution chart
- Flagged tickets by channel
- Severity delta heatmap (assigned vs inferred)
- Downloadable evidence dossiers for all flagged tickets

---

## Evidence Dossier Schema

```json
{
  "ticket_id": "TKT-12345",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": "+3",
  "feature_evidence": [
    {
      "signal": "keyword",
      "value": "production down",
      "weight": "critical_signal"
    },
    {
      "signal": "resolution_time",
      "value": "2.1h",
      "interpretation": "Fast resolution supports Critical severity"
    }
  ],
  "constraint_analysis": "Text contains critical-severity signals conflicting with assigned label Low. Resolution time 2.1h supports inferred severity Critical. Recommend re-triage to Critical.",
  "confidence": "99.2%"
}
```

Hard Rule: Every `feature_evidence` item is traceable to a specific field in the input ticket. No fabricated claims.

---

## Project Structure

```
support-integrity-auditor/
├── train_pipeline.py       <- Run this first. Trains all 3 stages.
├── predict.py              <- Inference. Input CSV -> predictions + dossiers.
├── app.py                  <- Streamlit dashboard.
├── notebook.ipynb          <- Full pipeline in notebook format.
├── requirements.txt        <- All dependencies.
├── README.md
├── sia_model/              <- Created after training.
│   ├── clf_lr.pkl
│   ├── clf_rf.pkl
│   ├── tfidf.pkl
│   ├── scaler.pkl
│   └── config.pkl
└── outputs/                <- Created after training.
    ├── metrics.json
    ├── ablation.json
    ├── dossiers.json
    └── test_predictions.csv
```

---

## Dataset Columns

| Original Name | Renamed To | Role |
|:---|:---|:---|
| `Ticket_Subject` | `Ticket Subject` | Short summary of issue |
| `Ticket_Description` | `Ticket Description` | Full problem text |
| `Priority_Level` | `Ticket Priority` | Human-assigned label |
| `Ticket_Channel` | `Ticket Channel` | Email / Chat / Phone / Web Form |
| `Resolution_Time_Hours` | `Resolution Time` | Hours to resolve |
| `Issue_Category` | `Ticket Type` | Technical / Billing / Account / etc. |
| `Customer_Email` | `Customer Email` | Domain tier proxy |

---

## Tech Stack

| Technology | Purpose |
|:---|:---|
| Python 3.9+ | Core development |
| scikit-learn | LR, Random Forest, TF-IDF, metrics |
| imbalanced-learn | SMOTE oversampling |
| pandas / numpy | Data processing |
| scipy | Sparse matrix operations |
| joblib | Save and load model files |
| Streamlit | Web dashboard |

---

## Limitations

- Keywords are context-blind — "spelling error" triggers high-signal due to the word "error"
- Resolution time adds noise for tickets genuinely fixed fast despite low urgency
- English only — non-English tickets will score incorrectly
- Static keyword lists may miss new terminology

**Planned improvements:**
- [ ] Sentence embeddings instead of keyword scoring (all-MiniLM-L6-v2)
- [ ] LLM-based zero-shot severity scoring (Mistral-7B)
- [ ] Active learning loop on human corrections
- [ ] Multi-language support

---

## License

MIT License
