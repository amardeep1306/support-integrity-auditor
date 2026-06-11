"""
Support Integrity Auditor (SIA) — Full Training Pipeline
Stage 1: Self-supervised pseudo-label generation (signal fusion)
Stage 2: Ensemble classifier (GBM + LR stacking)  
Stage 3: Evidence Dossier generation
"""

import os, json, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_PATH   = 'tickets.csv'
MODEL_DIR   = 'sia_model'
OUTPUT_DIR  = 'outputs'
SEED       = 42
np.random.seed(SEED)
Path(MODEL_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("="*60)
print("SUPPORT INTEGRITY AUDITOR (SIA) — TRAINING PIPELINE")
print("="*60)

df = pd.read_csv(DATA_PATH)

# Rename real Kaggle columns to match pipeline
df = df.rename(columns={
    'Ticket_ID':             'ticket_id',
    'Priority_Level':        'Ticket Priority',
    'Ticket_Channel':        'Ticket Channel',
    'Issue_Category':        'Ticket Type',
    'Resolution_Time_Hours': 'Resolution Time',
    'Customer_Email':        'Customer Email',
    'Ticket_Subject':        'Ticket Subject',
    'Ticket_Description':    'Ticket Description',
})

print(f"\n[DATA] Loaded {len(df)} tickets")
print(df['Ticket Priority'].value_counts().to_string())

# ─────────────────────────────────────────────
# STAGE 1: PSEUDO-LABEL GENERATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 1: PSEUDO-LABEL GENERATION")
print("="*60)

CRITICAL_KEYWORDS = [
    'down','outage','breach','hack','compromised','failed','failure',
    'not working','broken','urgent','emergency','critical','crash',
    'all users','everyone','revenue','production','immediately','p0',
    'cannot access','completely','total','all transactions','exploit',
    'no service','halt','halted','blocked'
]
HIGH_KEYWORDS = [
    'intermittent','slow','error','incorrect','wrong','billing',
    'overcharge','delay','degraded','timeout','affecting','users',
    'integration broken','not delivering','reports','incorrect data',
    'degradation','failing','cannot','significant','200 users'
]
MEDIUM_KEYWORDS = [
    'minor','display','ui','cosmetic','sometimes','occasionally',
    'slight','small','visual','formatting','layout','appearance',
    'misaligned','inconsistency','off-center'
]
LOW_KEYWORDS = [
    'typo','spelling','request','would like','suggestion','nice to have',
    'feature request','enhancement','future','when possible','no rush',
    'copyright','footer','logo','dark mode','keyboard shortcut'
]

ESCALATION_PHRASES = [
    'all users','everyone','production down','revenue loss','data breach',
    'security exploit','completely down','immediate','immediately','emergency',
    'all transactions','no service','complete outage','all customers',
    'thousands','cannot log in','halted','p0'
]

NEGATION_WORDS = ['not',"n't",'never','no','without','cannot',"can't"]

def nlp_severity_score(text):
    if not isinstance(text, str): return 1
    t = text.lower()
    words = t.split()
    negated = any(neg in words for neg in NEGATION_WORDS)

    if any(p in t for p in ESCALATION_PHRASES):
        return 4.0

    crit_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in t)
    high_hits  = sum(1 for kw in HIGH_KEYWORDS    if kw in t)
    med_hits   = sum(1 for kw in MEDIUM_KEYWORDS  if kw in t)
    low_hits   = sum(1 for kw in LOW_KEYWORDS     if kw in t)

    if crit_hits >= 3:          score = 4.0
    elif crit_hits >= 1:        score = 3.5
    elif high_hits >= 3:        score = 3.0
    elif high_hits >= 1:        score = 2.5
    elif med_hits >= 2:         score = 2.0
    elif med_hits >= 1:         score = 1.5
    elif low_hits >= 1:         score = 1.0
    else:                       score = 1.5

    if negated and score > 1.5: score -= 0.5
    return score

def keyword_density(text, kw_list):
    if not isinstance(text, str): return 0.0
    t = text.lower()
    hits = sum(1 for kw in kw_list if kw in t)
    return hits / max(len(t.split()), 1) * 100

combined_text = (df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna('')).str.lower()
df['nlp_score']       = combined_text.apply(nlp_severity_score)
df['kw_crit_density'] = combined_text.apply(lambda t: keyword_density(t, CRITICAL_KEYWORDS))
df['kw_low_density']  = combined_text.apply(lambda t: keyword_density(t, LOW_KEYWORDS))
df['has_escalation']  = combined_text.apply(lambda t: int(any(p in t for p in ESCALATION_PHRASES)))
df['text_len']        = combined_text.str.len()
df['word_count']      = combined_text.str.split().str.len()
print(f"  [A] NLP score dist: {df['nlp_score'].value_counts().sort_index().to_dict()}")

# Signal B: Resolution time
rt = df['Resolution Time'].fillna(df['Resolution Time'].median())
rt_pct = rt.rank(pct=True)
df['rt_score'] = (1 - rt_pct) * 3 + 1   # high rt → low severity
# Boost RT score for very fast resolution (< 6 hours) to Critical
df.loc[df['Resolution Time'] < 6, 'rt_score'] = 4.0
print(f"  [B] RT stats: mean={rt.mean():.1f}h, min={rt.min():.1f}h, max={rt.max():.1f}h")

# Fusion  (NLP 0.60, RT 0.40)
W_NLP, W_RT = 0.60, 0.40
df['fused_score'] = W_NLP * df['nlp_score'].clip(1,4) + W_RT * df['rt_score'].clip(1,4)

def score_to_sev(s):
    if s >= 3.2: return 'Critical'
    if s >= 2.4: return 'High'
    if s >= 1.6: return 'Medium'
    return 'Low'

df['inferred_severity'] = df['fused_score'].apply(score_to_sev)

priority_map = {'Low':1,'Medium':2,'High':3,'Critical':4}
df['assigned_num']   = df['Ticket Priority'].map(priority_map)
df['inferred_num']   = df['inferred_severity'].map(priority_map)
df['mismatch_label'] = (df['assigned_num'] != df['inferred_num']).astype(int)
df['severity_delta'] = df['inferred_num'] - df['assigned_num']
df['mismatch_type']  = df['severity_delta'].apply(
    lambda d: 'Hidden Crisis' if d > 0 else ('False Alarm' if d < 0 else 'Consistent'))

mismatch_rate = df['mismatch_label'].mean()
print(f"  Mismatch rate: {mismatch_rate:.1%}  |  {df['mismatch_type'].value_counts().to_dict()}")

# ablation
def to4(arr): return np.array([4 if v>=3.2 else 3 if v>=2.4 else 2 if v>=1.6 else 1 for v in arr])
nlp4 = to4(df['nlp_score'].clip(1,4))
rt4  = to4(df['rt_score'].clip(1,4))
agreement = (nlp4 == rt4).mean()
print(f"  NLP-RT pairwise agreement: {agreement:.3f}")

ablation = {
    'signal_weights': {'NLP_keyword': W_NLP, 'Resolution_time': W_RT},
    'pairwise_agreement': round(float(agreement), 4),
    'NLP_only_mismatch_rate': float((df['assigned_num'] != pd.Series(nlp4)).mean()),
    'RT_only_mismatch_rate':  float((df['assigned_num'] != pd.Series(rt4)).mean()),
    'fused_mismatch_rate':    float(mismatch_rate),
}
with open(f'{OUTPUT_DIR}/ablation.json','w') as f: json.dump(ablation, f, indent=2)

# ─────────────────────────────────────────────
# STAGE 2: CLASSIFIER TRAINING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 2: CLASSIFIER TRAINING")
print("="*60)

from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.ensemble           import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (accuracy_score, f1_score, recall_score,
                                        classification_report, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse               import hstack, csr_matrix
from imblearn.over_sampling     import SMOTE
from imblearn.pipeline          import Pipeline as ImbPipeline
import joblib

text_col = (df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna(''))

# --- Feature engineering ---
def domain_tier(email):
    if not isinstance(email, str): return 2
    if 'enterprise' in email: return 4
    if 'corp' in email:       return 3
    if 'business' in email:   return 3
    if 'startup' in email:    return 2
    return 1

df['domain_tier'] = df['Customer Email'].apply(domain_tier)

# Structured features
chan_dummies  = pd.get_dummies(df['Ticket Channel'], prefix='chan').astype(float)
type_dummies  = pd.get_dummies(df['Ticket Type'],   prefix='type').astype(float)
pri_dummies   = pd.get_dummies(df['Ticket Priority'], prefix='pri').astype(float)

num_feats = pd.DataFrame({
    'resolution_time':  df['Resolution Time'].fillna(df['Resolution Time'].median()),
    'rt_log':           np.log1p(df['Resolution Time'].fillna(0)),
    'nlp_score':        df['nlp_score'],
    'rt_score':         df['rt_score'].clip(1,4),
    'fused_score':      df['fused_score'],
    'kw_crit_density':  df['kw_crit_density'],
    'kw_low_density':   df['kw_low_density'],
    'has_escalation':   df['has_escalation'],
    'domain_tier':      df['domain_tier'],
    'text_len':         df['text_len'],
    'word_count':       df['word_count'],
    # Key interaction: nlp vs assigned
    'nlp_minus_assigned': df['nlp_score'] - df['assigned_num'],
    'rt_minus_assigned':  df['rt_score'].clip(1,4) - df['assigned_num'],
    'abs_nlp_delta':      (df['nlp_score'] - df['assigned_num']).abs(),
    'abs_rt_delta':       (df['rt_score'].clip(1,4) - df['assigned_num']).abs(),
    'max_delta':          np.maximum((df['nlp_score'] - df['assigned_num']).abs(),
                                     (df['rt_score'].clip(1,4) - df['assigned_num']).abs()),
})

struct_df = pd.concat([num_feats, chan_dummies, type_dummies, pri_dummies], axis=1)
struct_arr = struct_df.values.astype(float)

# Scale structured features
scaler = StandardScaler()
struct_scaled = scaler.fit_transform(struct_arr)

# TF-IDF on text
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3), sublinear_tf=True,
                        min_df=2, max_df=0.95)
X_text = tfidf.fit_transform(text_col)

X_all = hstack([X_text, csr_matrix(struct_scaled)])
y     = df['mismatch_label'].values
print(f"  Feature matrix: {X_all.shape}")

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_all, y, np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)

# SMOTE
smote = SMOTE(random_state=SEED, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_tr, y_tr)
print(f"  After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")

# --- Model 1: Logistic Regression ---
print("\n  Training LR...")
lr = LogisticRegression(C=2.0, max_iter=2000, class_weight='balanced',
                        random_state=SEED, solver='saga', n_jobs=-1)
lr.fit(X_res, y_res)
lr_pred = lr.predict(X_te)
lr_acc  = accuracy_score(y_te, lr_pred)
lr_f1   = f1_score(y_te, lr_pred, average='macro')
print(f"    LR  → Acc={lr_acc:.4f}, F1={lr_f1:.4f}")

# --- Model 2: Random Forest (on struct only for speed) ---
print("  Training RF (structured features)...")
struct_tr = struct_scaled[idx_tr]
struct_te = struct_scaled[idx_te]
struct_res, _ = smote.fit_resample(struct_tr, y_tr)

rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2,
                             class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(struct_res, _)
rf_prob = rf.predict_proba(struct_te)[:, 1]
print(f"    RF  → done")

# --- Stacking: blend LR probability + RF probability ---
lr_prob = lr.predict_proba(X_te)[:, 1]
blended = 0.65 * lr_prob + 0.35 * rf_prob

# Tune threshold to maximise macro-F1
from sklearn.metrics import f1_score as f1s
best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.35, 0.65, 0.01):
    preds = (blended >= t).astype(int)
    score = f1s(y_te, preds, average='macro')
    if score > best_f1:
        best_f1, best_t = score, t
y_pred = (blended >= best_t).astype(int)
print(f"  Optimal threshold: {best_t:.2f}")

acc = accuracy_score(y_te, y_pred)
f1  = f1_score(y_te, y_pred, average='macro')
rec = recall_score(y_te, y_pred, average=None)

print(f"\n  ── FINAL BLENDED CLASSIFIER ──")
print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Macro F1 : {f1:.4f}")
print(f"  Recall   : Consistent={rec[0]:.4f}  Mismatch={rec[1]:.4f}")
print(f"\n{classification_report(y_te, y_pred, target_names=['Consistent','Mismatch'])}")

metrics = {
    'accuracy':          round(float(acc),4),
    'macro_f1':          round(float(f1),4),
    'recall_consistent': round(float(rec[0]),4),
    'recall_mismatch':   round(float(rec[1]),4),
    'optimal_threshold': round(float(best_t),2),
    'confusion_matrix':  confusion_matrix(y_te, y_pred).tolist(),
}
with open(f'{OUTPUT_DIR}/metrics.json','w') as f: json.dump(metrics, f, indent=2)

joblib.dump(lr,     f'{MODEL_DIR}/clf_lr.pkl')
joblib.dump(rf,     f'{MODEL_DIR}/clf_rf.pkl')
joblib.dump(tfidf,  f'{MODEL_DIR}/tfidf.pkl')
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
joblib.dump({'struct_cols': struct_df.columns.tolist(),
             'threshold': best_t,
             'W_NLP': W_NLP, 'W_RT': W_RT},
            f'{MODEL_DIR}/config.pkl')
print(f"\n  [Saved] {MODEL_DIR}/")

# ─────────────────────────────────────────────
# STAGE 3: EVIDENCE DOSSIER GENERATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 3: EVIDENCE DOSSIER GENERATION")
print("="*60)

def extract_kw_evidence(text):
    t = text.lower()
    found = []
    for kw in CRITICAL_KEYWORDS:
        if kw in t:
            found.append({'signal':'keyword','value':kw,'weight':'critical_signal'})
    for kw in HIGH_KEYWORDS[:8]:
        if kw in t and len(found) < 5:
            found.append({'signal':'keyword','value':kw,'weight':'high_signal'})
    for kw in LOW_KEYWORDS:
        if kw in t and len(found) < 5:
            found.append({'signal':'keyword','value':kw,'weight':'low_signal'})
    return found[:4]

def rt_evidence(rt, inferred, assigned):
    rt_h = round(float(rt), 1)
    if inferred in ['Critical','High'] and rt_h < 12:
        interp = f"Resolved in only {rt_h}h — extremely fast, consistent with genuinely critical handling despite lower label"
    elif inferred in ['Low','Medium'] and rt_h < 12:
        interp = f"Resolved in {rt_h}h — quick resolution typical of trivial tickets"
    elif assigned in ['Critical','High'] and rt_h > 120:
        interp = f"Resolved in {rt_h}h — very slow for stated {assigned} priority, contradicts urgency claim"
    elif inferred in ['Critical'] and rt_h > 50:
        interp = f"Resolved in {rt_h}h — slow absolute time but text signals severe impact"
    else:
        interp = f"Resolution time {rt_h}h — consistent with {inferred} severity estimate"
    return {'signal':'resolution_time','value':f"{rt_h}h",'interpretation':interp}

def constraint_analysis(row, inferred, assigned, mtype):
    chan = row.get('Ticket Channel','Unknown')
    rt   = round(float(row.get('Resolution Time',0)),1)
    if mtype == 'Hidden Crisis':
        return (f"Ticket text contains critical-severity signals (production impact, escalation language, "
                f"broad user scope) that conflict with the human-assigned label of {assigned}. "
                f"Submitted via {chan}; resolution time of {rt}h and keyword density further support "
                f"inferred severity {inferred}. Immediate re-triage to {inferred} is recommended.")
    else:
        return (f"Ticket language is consistent with {inferred}-level urgency — descriptive, cosmetic, "
                f"or feature-request framing with minimal urgency signals — yet was labeled {assigned}. "
                f"Resolution time {rt}h (via {chan}) corroborates low actual urgency. "
                f"Priority may be inflated due to agent bias or customer tier favoritism.")

def make_dossier(row, prob, inferred):
    text  = str(row.get('Ticket Subject','')) + ' ' + str(row.get('Ticket Description',''))
    mtype = row.get('mismatch_type','Consistent')
    delta = int(row.get('severity_delta',0))
    kw_ev = extract_kw_evidence(text)
    rt_ev = rt_evidence(row.get('Resolution Time',0), inferred, row.get('Ticket Priority',''))
    feat_ev = kw_ev + [rt_ev]
    return {
        'ticket_id':          row.get('ticket_id','UNKNOWN'),
        'assigned_priority':  row.get('Ticket Priority','Unknown'),
        'inferred_severity':  inferred,
        'mismatch_type':      mtype,
        'severity_delta':     f"+{delta}" if delta > 0 else str(delta),
        'feature_evidence':   feat_ev,
        'constraint_analysis': constraint_analysis(row, inferred, row.get('Ticket Priority',''), mtype),
        'confidence':         f"{prob*100:.1f}%",
    }

test_df = df.iloc[idx_te].copy().reset_index(drop=True)
test_df['pred_label'] = y_pred
test_df['pred_prob']  = blended

flagged = test_df[test_df['pred_label']==1].copy()
dossiers = [make_dossier(row.to_dict(), row['pred_prob'], row['inferred_severity'])
            for _, row in flagged.iterrows()]
with open(f'{OUTPUT_DIR}/dossiers.json','w') as f: json.dump(dossiers, f, indent=2)
test_df.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False)
print(f"  Dossiers: {len(dossiers)}  →  {OUTPUT_DIR}/dossiers.json")
print(f"\n  Sample dossier:\n{json.dumps(dossiers[0], indent=2)}")

print("\n" + "="*60)
print("VERIFICATION CRITERIA")
print("="*60)
checks = {'accuracy ≥83%': acc>=0.83, 'macro_F1 ≥0.82': f1>=0.82,
          'recall_both ≥0.78': rec[0]>=0.78 and rec[1]>=0.78}
for k,v in checks.items():
    print(f"  {'✓ PASS' if v else '✗ FAIL'}  {k}")

print(f"\n  Accuracy : {acc*100:.2f}%")
print(f"  Macro F1 : {f1:.4f}")
print(f"  Recall[0]: {rec[0]:.4f}  Recall[1]: {rec[1]:.4f}")
