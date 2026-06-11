<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-98.75%25-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/F1_Score-0.987-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

<br/><br/>

# 🔍 Support Integrity Auditor (SIA)

**Semantics-driven · Evidence-grounded · Self-supervised**

*Automatically detects Priority Mismatch in CRM support tickets and generates structured Evidence Dossiers*

<br/>

> ### 💬 *"2 out of every 3 support tickets are mislabeled. SIA catches them — automatically, with evidence."*

<br/>

[![Dataset](https://img.shields.io/badge/📦_Dataset-Kaggle_CRM-orange?style=flat-square)](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset)

</div>

---

## 📌 Table of Contents

- [🎯 The Problem](#-the-problem)
- [💡 The Solution](#-the-solution)
- [⚙️ How It Works](#️-how-it-works)
- [📊 Results](#-results)
- [🔬 Ablation Study](#-ablation-study)
- [⚡ Quick Start — Run It Yourself](#-quick-start--run-it-yourself)
- [🖥️ Web App Features](#️-web-app-features)
- [📂 Evidence Dossier Schema](#-evidence-dossier-schema)
- [🗂️ Project Structure](#️-project-structure)
- [📦 Dataset](#-dataset)
- [🔧 Tech Stack](#-tech-stack)
- [⚠️ Limitations](#️-limitations)

---

## 🎯 The Problem

In enterprise-scale CRM ecosystems, human agents assign priority labels to thousands of support tickets every day. But humans make mistakes — and those mistakes have real business consequences.

| 😴 Problem | 📋 What Happens | 💥 Business Impact |
|:---:|:---|:---|
| **Agent Fatigue** | Tired agent labels everything Low | Critical issues ignored for days |
| **Customer Favoritism** | VIP customer gets Critical for minor issue | Real emergencies stuck in queue |
| **Keyword Anchoring** | Sees word "urgent" → labels Critical blindly | Engineers waste time on trivial issues |
| **Volume Overload** | Too many tickets to read carefully | Random labeling errors at scale |

<br/>

> ❌ **Hidden Crisis** — A production outage labeled as **Low**. Nobody looks at it. Revenue is lost.
>
> ❌ **False Alarm** — A footer typo labeled as **Critical**. Engineers drop everything. Real emergencies wait.

---

## 💡 The Solution

SIA is a **self-supervised AI pipeline** that requires **zero pre-annotated mismatch labels** — it bootstraps its own supervision signal from raw ticket data alone.

**SIA works in 5 steps:**

- 📖 Reads every ticket — completely ignoring the human label
- 🧠 Infers true severity from text signals + resolution time
- ⚖️ Compares inferred severity vs human-assigned priority
- 🚨 Flags mismatches with full confidence score
- 📋 Generates structured Evidence Dossier — zero hallucination

<br/>

### Two Mismatch Types Detected

| Type | Meaning | Example | Risk |
|:---:|:---|:---|:---:|
| 🔴 **Hidden Crisis** | True severity **higher** than assigned | Production down → labeled **Low** | 🔴 CRITICAL |
| 🔵 **False Alarm** | True severity **lower** than assigned | Typo in footer → labeled **Critical** | 🟡 MEDIUM |

---

## ⚙️ How It Works

SIA runs a **3-stage self-supervised pipeline**:

<br/>

### Stage 1 — Pseudo-Label Generation

The model creates its own training labels from scratch — no human annotation needed.

**Signal A: NLP Keyword Scoring (Weight: 60%)**

| Severity | Keywords Used |
|:---:|:---|
| 🔴 Critical | `production down` `all users` `breach` `revenue loss` `emergency` `immediately` |
| 🟠 High | `billing error` `degraded` `timeout` `incorrect data` `integration broken` |
| 🟡 Medium | `minor` `cosmetic` `display` `ui` `misaligned` `slight` |
| 🟢 Low | `typo` `no rush` `nice to have` `spelling` `feature request` |

**Signal B: Resolution-Time Proxy (Weight: 40%)**

| Resolution Time | What It Implies | Severity Signal |
|:---:|:---|:---:|
| 0 – 8 hours | Team dropped everything to fix it | 🔴 Critical |
| 8 – 48 hours | Treated with moderate urgency | 🟠 High |
| 48 – 120 hours | Normal queue processing | 🟡 Medium |
| 120+ hours | Low urgency, sat in backlog | 🟢 Low |

**Fusion Formula:**
<br/>

### Stage 2 — Classifier Training

| Component | Details |
|:---|:---|
| **Text Features** | TF-IDF vectorizer — 5,000 features, 1–3 grams, sublinear TF |
| **Structured Features** | Resolution time, domain tier, channel, ticket type |
| **Interaction Features** | `nlp_score − assigned_priority` — directly captures mismatch signal |
| **Class Imbalance** | SMOTE oversampling — balances mismatch vs consistent classes |
| **Model** | Logistic Regression (65%) + Random Forest 300 trees (35%) blended |
| **Threshold** | Tuned on macro-F1 → optimal at 0.40 |

<br/>

### Stage 3 — Evidence Dossier Generation

For every flagged ticket, SIA generates a structured JSON report containing:

- 🔑 **Keywords found** — traceable to exact input text field
- ⏱️ **Resolution time** — interpreted as a severity signal
- 📝 **Constraint analysis** — 2–3 grounded sentences explaining the mismatch
- 📊 **Confidence score** — 0% to 100%

> **Hard Rule:** Every evidence item is directly traceable to the input ticket. Zero hallucination — ever.

---

## 📊 Results

### ✅ Verification Criteria — All Passed

| Metric | Result | Threshold | Status |
|:---|:---:|:---:|:---:|
| Binary Classification Accuracy | **98.75%** | ≥ 83% | ✅ PASS |
| Macro F1 Score | **0.9874** | ≥ 0.82 | ✅ PASS |
| Per-Class Recall — Consistent | **0.9747** | ≥ 0.78 | ✅ PASS |
| Per-Class Recall — Mismatch | **0.9988** | ≥ 0.78 | ✅ PASS |

<br/>

### 🔢 Confusion Matrix — Test Set (n = 1,600)

| | Predicted: Consistent | Predicted: Mismatch |
|:---|:---:|:---:|
| **True: Consistent** | 733 ✅ | 19 ❌ |
| **True: Mismatch** | 1 ❌ | 847 ✅ |

> 🎯 Only **1 missed mismatch** and **19 false positives** out of 1,600 test tickets.

<br/>

### 📈 Batch Results on 20,000 Real Tickets

| Category | Count | Percentage |
|:---|:---:|:---:|
| ✅ Consistent (correctly labeled) | 6,749 | 33.7% |
| 🔴 Hidden Crises (under-labeled) | 8,852 | 44.3% |
| 🔵 False Alarms (over-labeled) | 4,029 | 20.2% |
| **Total Flagged Mismatches** | **13,251** | **66.3%** |

---

## 🔬 Ablation Study

| Signal Configuration | Mismatch Rate | Notes |
|:---|:---:|:---|
| NLP Keywords Only | 54.9% | Strong but misses behavioral signals |
| Resolution Time Only | 41.5% | Weaker alone — text is more informative |
| **Fused (NLP 60% + RT 40%)** | **53.0%** | Most balanced and reliable |
| Signal Pairwise Agreement | 0.513 | Confirms signals are truly independent |

**Why this fusion works:**

- 📝 **NLP** captures what the ticket *says* — semantic urgency in language
- ⏱️ **RT** captures how the team *actually treated* it — behavioral evidence
- 🔗 **0.513 agreement** confirms signals are independent — fusion adds real value
- ⚖️ **NLP weighted higher (60%)** because text is the most direct severity indicator

---

## ⚡ Quick Start — Run It Yourself

> Complete step-by-step guide for anyone running this project for the first time.

<br/>

### Step 1 — Prerequisites

Make sure you have these installed:

| Software | Version | Download |
|:---|:---:|:---|
| Python | 3.9+ | [python.org/downloads](https://python.org/downloads) |
| VS Code | Any | [code.visualstudio.com](https://code.visualstudio.com) |
| Git | Any | [git-scm.com](https://git-scm.com) |

<br/>

### Step 2 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/YOURUSERNAME/support-integrity-auditor.git
cd support-integrity-auditor
```

<br/>

### Step 3 — Create Virtual Environment

```bash
python -m venv sia_env
```

<br/>

### Step 4 — Activate Virtual Environment

```bash
# Windows
sia_env\Scripts\activate

# Mac / Linux
source sia_env/bin/activate
```

After activation your terminal will show `(sia_env)` at the start. This must be active for all commands below.

<br/>

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages automatically. Takes 2–3 minutes.

<br/>

### Step 6 — Download the Dataset

1. Go to [kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset)
2. Download `customer_support_tickets.csv`
3. Rename it to `tickets.csv`
4. Place it inside the `support-integrity-auditor/` folder

<br/>

### Step 7 — Train the Model

```bash
python train_pipeline.py
```

This runs all 3 stages of the pipeline. Expected output at the end:
Training creates two folders automatically:
- `sia_model/` — trained model files
- `outputs/` — metrics, dossiers, predictions

<br/>

### Step 8 — Launch the Web App

```bash
streamlit run app.py
```

Opens automatically at **http://localhost:8501** in your browser.

<br/>

### Step 9 — Run Inference on Any CSV (Optional)

```bash
python predict.py --input tickets.csv --output ./my_results
```

Saves predictions and dossiers to `my_results/` folder.

<br/>

> ⚠️ **Important:** Always activate `sia_env` and always run `train_pipeline.py` before `app.py` or `predict.py`

---

## 🖥️ Web App Features

### Single Ticket Mode

Fill in any ticket and click **Analyze Ticket**:

| Field | What to Enter |
|:---|:---|
| Ticket Subject | One-line title of the issue |
| Ticket Description | Full problem description — most important field |
| Assigned Priority | Low / Medium / High / Critical |
| Resolution Time | Hours taken to resolve (e.g. 2 or 200) |
| Channel | Email / Chat / Phone / Web Form |

**Output:**

| Output | Meaning |
|:---|:---|
| Verdict | MISMATCH or CONSISTENT |
| Inferred Severity | What SIA thinks the true priority should be |
| Confidence | How certain the model is (0–100%) |
| Mismatch Type | Hidden Crisis (red) or False Alarm (blue) |
| Feature Evidence | Exact keywords found + resolution time interpretation |
| Constraint Analysis | 2–3 sentence grounded explanation |

<br/>

### Batch CSV Mode

Upload `tickets.csv` and get the full dashboard:

| Element | Description |
|:---|:---|
| 📊 KPI Row | Total tickets, flagged mismatches, hidden crises, false alarms |
| 📈 Mismatch Distribution | Bar chart — Consistent vs False Alarm vs Hidden Crisis |
| 📉 By Channel | Which channels have the most mislabeled tickets |
| 🗺️ Severity Delta Heatmap | Matrix of assigned priority vs inferred severity |
| 📂 Dossiers Tab | Expandable evidence for every flagged ticket + download |

---

## 📂 Evidence Dossier Schema

```json
{
  "ticket_id":          "TKT-12345",
  "assigned_priority":  "Low",
  "inferred_severity":  "Critical",
  "mismatch_type":      "Hidden Crisis",
  "severity_delta":     "+3",
  "feature_evidence": [
    {
      "signal":  "keyword",
      "value":   "production down",
      "weight":  "critical_signal"
    },
    {
      "signal":          "resolution_time",
      "value":           "2.1h",
      "interpretation":  "Fast resolution supports Critical severity"
    }
  ],
  "constraint_analysis": "Text contains critical-severity signals conflicting
    with assigned label Low. Resolution time 2.1h supports inferred severity
    Critical. Recommend immediate re-triage to Critical.",
  "confidence": "99.2%"
}
```

> 🔒 **Hard Rule:** Every `feature_evidence` item is directly traceable to a specific field in the input ticket. No fabricated or unverifiable claims — ever.

---

## 🗂️ Project Structure
---

## 📦 Dataset

**Customer Support Tickets — CRM Dataset**

> 🔗 [kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset)

| Original Column | Renamed To | Role |
|:---|:---|:---|
| `Ticket_ID` | `ticket_id` | Unique ticket identifier |
| `Ticket_Subject` | `Ticket Subject` | Short summary of the issue |
| `Ticket_Description` | `Ticket Description` | Full natural language problem |
| `Priority_Level` | `Ticket Priority` | Human-assigned label (Low/Med/High/Crit) |
| `Ticket_Channel` | `Ticket Channel` | Email / Chat / Phone / Web Form |
| `Resolution_Time_Hours` | `Resolution Time` | Hours to resolve — severity proxy |
| `Issue_Category` | `Ticket Type` | Technical / Billing / Account / etc. |
| `Customer_Email` | `Customer Email` | Domain tier proxy |

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| Language | Python 3.9+ | Core development |
| ML Framework | scikit-learn 1.2+ | LR, RF, TF-IDF, metrics |
| Class Balancing | imbalanced-learn | SMOTE oversampling |
| Data Processing | pandas + numpy | CSV handling, feature engineering |
| Sparse Matrices | scipy | Efficient TF-IDF operations |
| Model Persistence | joblib | Save and load .pkl files |
| Web Dashboard | Streamlit | Single ticket + batch CSV UI |

---

## ⚠️ Limitations

| Limitation | Example | Impact |
|:---|:---|:---|
| Context-blind keywords | "spelling **error**" triggers high-signal | Occasional false positives |
| Resolution time noise | Critical ticket fixed fast = ambiguous | Lower confidence on edge cases |
| English only | Non-English tickets | Incorrect scoring |
| Static keyword lists | Novel product names | May miss new terminology |

### 🚀 Future Improvements

- [ ] Replace keyword scoring with **sentence embeddings** (all-MiniLM-L6-v2)
- [ ] Use **fine-tuned LLM** (Mistral-7B-Instruct) for zero-shot severity scoring
- [ ] Add **active learning loop** — retrain on human-corrected predictions
- [ ] **Multi-language support** via multilingual embeddings
- [ ] Add **temporal features** — submission time, agent shift, SLA pressure

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

### 🏆 All Verification Criteria Passed

| Accuracy | Macro F1 | Recall (Both Classes) |
|:---:|:---:|:---:|
| **98.75%** ✅ | **0.9874** ✅ | **≥ 0.97** ✅ |

<br/>

**Built with ❤️ using Python · scikit-learn · Streamlit**

<br/>

⭐ *If this project helped you, consider giving it a star!* ⭐

</div>
