# Support Integrity Auditor (SIA)

**Semantics-driven, evidence-grounded automated auditor that detects Priority Mismatch in CRM support tickets.**

---

## Problem

In enterprise-scale CRM ecosystems, manual ticket triage suffers from agent fatigue bias, customer favoritism, and keyword anchoring. SIA flags tickets where the *objective characteristics* (text, resolution time, channel, customer domain) conflict with the human-assigned priority — producing an evidence dossier for each flagged case.

---

## Architecture

```
Raw CRM Tickets
      │
      ▼
┌─────────────────────────────────────────────┐
│        Stage 1: Pseudo-Label Generation     │
│                                             │
│  Signal A: Rule-based NLP (weight: 0.60)   │
│   • Keyword density (critical/high/low)     │
│   • Escalation phrase detection             │
│   • Negation-aware scoring                  │
│                                             │
│  Signal B: Resolution-Time Regression       │
│             (weight: 0.40)                  │
│   • Rank-based severity proxy               │
│   • Lower RT → higher severity              │
│                                             │
│  → Fused inferred severity (1–4 scale)      │
│  → Binary mismatch pseudo-label             │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│        Stage 2: Classifier Training         │
│                                             │
│  Features:                                  │
│   • TF-IDF (5000 features, 1–3 grams)      │
│   • Structured: RT, domain tier, channel   │
│   • Interaction: nlp_score – assigned_num  │
│   • One-hot: channel, ticket type, priority│
│                                             │
│  Imbalance: SMOTE oversampling              │
│                                             │
│  Ensemble:                                  │
│   • Logistic Regression (L2, saga, C=2.0)  │
│   • Random Forest (300 trees, balanced)     │
│   • Blend: 0.65×LR + 0.35×RF              │
│   • Threshold tuned on macro-F1             │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│      Stage 3: Evidence Dossier Generation   │
│                                             │
│  Per flagged ticket:                        │
│   • Keyword evidence (traceable to input)  │
│   • Resolution time interpretation         │
│   • Constraint analysis (2–3 sentences)    │
│   • Confidence score                        │
│  Hard rule: zero hallucination              │
└─────────────────────────────────────────────┘
```

---

## Evaluation Metrics

| Metric                     | Result    | Threshold  | Status |
|----------------------------|-----------|------------|--------|
| Binary Classification Acc  | **98.75%**| ≥ 83%      | ✅ PASS |
| Macro F1 Score             | **0.9874** | ≥ 0.82    | ✅ PASS |
| Per-Class Recall (Consist.)| **0.9747** | ≥ 0.78   | ✅ PASS |
| Per-Class Recall (Mismatch)| **0.9988** | ≥ 0.78   | ✅ PASS |

**Confusion Matrix (test set, n=1600):**
```
                 Pred: Consistent   Pred: Mismatch
True: Consistent       733               19
True: Mismatch           1              847
```

---

## Ablation Study

Each signal was evaluated independently vs. the fused result:

| Signal                    | Weight | Mismatch Rate Detected |
|---------------------------|--------|------------------------|
| NLP Keyword Only          | —      | 54.9%                  |
| Resolution Time Only      | —      | 41.5%                  |
| **Fused (NLP 60% + RT 40%)**| —  | **53.0%**              |
| Signal Pairwise Agreement | —      | 0.513                  |

**Fusion Justification:** NLP keyword scoring captures semantic urgency signals that resolution time cannot (e.g., a Critical ticket resolved quickly still has critical language). Resolution time provides an orthogonal behavioral signal — it captures cases where agents implicitly treated tickets as more or less severe than labeled. The 0.513 pairwise agreement confirms the signals are meaningfully independent, justifying their combination. NLP receives higher weight (0.60) because text is the most direct severity signal; RT weight (0.40) accounts for its noise (some critical issues are genuinely fast to resolve).

---

## Dataset

**Customer Support Tickets — CRM Dataset** (synthetic replica matching schema of `kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset`)

| Column               | Role                                   |
|----------------------|----------------------------------------|
| Ticket Subject       | Short summary of the issue             |
| Ticket Description   | Full natural language problem statement|
| Customer Email       | Proxy for customer tier / domain       |
| Ticket Priority      | Human-assigned label (Low/Med/High/Crit)|
| Ticket Channel       | Intake channel (email, chat, phone, SM)|
| Resolution Time      | Time to resolve — indirect severity    |
| Ticket Type          | Category of issue                      |

---

## Files

| File               | Purpose                                          |
|--------------------|--------------------------------------------------|
| `notebook.ipynb`   | Full reproducible pipeline (pseudo-label → train → infer) |
| `train_pipeline.py`| Standalone training script                       |
| `predict.py`       | Inference: accepts CSV, outputs predictions + dossiers |
| `app.py`           | Streamlit dashboard (single ticket + batch mode) |
| `README.md`        | This file                                        |
| `requirements.txt` | Pinned dependencies                              |

---

## Usage

### Training
```bash
python train_pipeline.py
```

### Inference
```bash
python predict.py --input tickets.csv --output ./results
```

### Web App
```bash
streamlit run app.py
```

---

## Evidence Dossier Schema

```json
{
  "ticket_id": "TKT-12345",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "+3",
  "feature_evidence": [
    { "signal": "keyword", "value": "production down", "weight": "critical_signal" },
    { "signal": "resolution_time", "value": "2.1h", "interpretation": "..." }
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>",
  "confidence": "99.2%"
}
```

**Hard Rule:** Every `feature_evidence` item is traceable to a specific field in the input ticket. No fabricated or unverifiable claims.

---

## Mismatch Types

- **Hidden Crisis**: Inferred severity > Assigned priority (ticket is more urgent than labeled — dangerous)
- **False Alarm**: Inferred severity < Assigned priority (ticket is less urgent than labeled — wastes resources)
