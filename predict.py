"""
predict.py — SIA Inference Script
Usage: python predict.py --input tickets.csv --output results/
"""

import argparse, json, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')

MODEL_DIR = 'sia_model'

# ── Keyword lists (must match training) ───────────────────
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
MEDIUM_KEYWORDS = ['minor','display','ui','cosmetic','sometimes','occasionally',
    'slight','small','visual','formatting','layout','appearance','misaligned','inconsistency']
LOW_KEYWORDS = ['typo','spelling','request','would like','suggestion','nice to have',
    'feature request','enhancement','future','when possible','no rush',
    'copyright','footer','logo','dark mode','keyboard shortcut']
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
    if any(p in t for p in ESCALATION_PHRASES): return 4.0
    crit = sum(1 for kw in CRITICAL_KEYWORDS if kw in t)
    high = sum(1 for kw in HIGH_KEYWORDS    if kw in t)
    med  = sum(1 for kw in MEDIUM_KEYWORDS  if kw in t)
    low  = sum(1 for kw in LOW_KEYWORDS     if kw in t)
    if crit >= 3: s = 4.0
    elif crit >= 1: s = 3.5
    elif high >= 3: s = 3.0
    elif high >= 1: s = 2.5
    elif med >= 2:  s = 2.0
    elif med >= 1:  s = 1.5
    elif low >= 1:  s = 1.0
    else:           s = 1.5
    if negated and s > 1.5: s -= 0.5
    return s

def kw_density(text, kws):
    if not isinstance(text, str): return 0.0
    t = text.lower()
    return sum(1 for kw in kws if kw in t) / max(len(t.split()),1) * 100

def domain_tier(email):
    if not isinstance(email, str): return 2
    if 'enterprise' in email: return 4
    if 'corp' in email:       return 3
    if 'business' in email:   return 3
    if 'startup' in email:    return 2
    return 1

def score_to_sev(s):
    if s >= 3.2: return 'Critical'
    if s >= 2.4: return 'High'
    if s >= 1.6: return 'Medium'
    return 'Low'

PRIORITY_MAP = {'Low':1,'Medium':2,'High':3,'Critical':4}

def predict(df_in: pd.DataFrame, output_dir: str = None):
    """Run SIA inference on a DataFrame. Returns (predictions_df, dossiers_list)."""
    cfg    = joblib.load(f'{MODEL_DIR}/config.pkl')
    tfidf  = joblib.load(f'{MODEL_DIR}/tfidf.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')
    lr     = joblib.load(f'{MODEL_DIR}/clf_lr.pkl')
    rf     = joblib.load(f'{MODEL_DIR}/clf_rf.pkl')
    threshold = cfg['threshold']
    W_NLP, W_RT = cfg['W_NLP'], cfg['W_RT']

    df = df_in.copy()

    # Rename Kaggle columns
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
    combined = (df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna('')).str.lower()

    df['nlp_score']       = combined.apply(nlp_severity_score)
    df['kw_crit_density'] = combined.apply(lambda t: kw_density(t, CRITICAL_KEYWORDS))
    df['kw_low_density']  = combined.apply(lambda t: kw_density(t, LOW_KEYWORDS))
    df['has_escalation']  = combined.apply(lambda t: int(any(p in t for p in ESCALATION_PHRASES)))
    df['text_len']        = combined.str.len()
    df['word_count']      = combined.str.split().str.len()

    rt = df['Resolution Time'].fillna(df['Resolution Time'].median() if len(df) > 1 else 0)
    rt_pct = rt.rank(pct=True)
    df['rt_score'] = (1 - rt_pct) * 3 + 1
    df['fused_score'] = W_NLP * df['nlp_score'].clip(1,4) + W_RT * df['rt_score'].clip(1,4)
    df['inferred_severity'] = df['fused_score'].apply(score_to_sev)
    df['assigned_num'] = df['Ticket Priority'].map(PRIORITY_MAP).fillna(2)
    df['inferred_num'] = df['inferred_severity'].map(PRIORITY_MAP)
    df['severity_delta'] = df['inferred_num'] - df['assigned_num']
    df['mismatch_type'] = df['severity_delta'].apply(
        lambda d: 'Hidden Crisis' if d > 0 else ('False Alarm' if d < 0 else 'Consistent'))
    df['domain_tier'] = df['Customer Email'].apply(domain_tier)

    chan_dummies = pd.get_dummies(df['Ticket Channel'], prefix='chan').astype(float)
    type_dummies = pd.get_dummies(df['Ticket Type'],   prefix='type').astype(float)
    pri_dummies  = pd.get_dummies(df['Ticket Priority'], prefix='pri').astype(float)

    # Align dummy columns to training set
    for col in cfg['struct_cols']:
        if col.startswith('chan_') and col not in chan_dummies:
            chan_dummies[col] = 0.0
        if col.startswith('type_') and col not in type_dummies:
            type_dummies[col] = 0.0
        if col.startswith('pri_') and col not in pri_dummies:
            pri_dummies[col] = 0.0

    num_feats = pd.DataFrame({
        'resolution_time':    rt,
        'rt_log':             np.log1p(rt),
        'nlp_score':          df['nlp_score'],
        'rt_score':           df['rt_score'].clip(1,4),
        'fused_score':        df['fused_score'],
        'kw_crit_density':    df['kw_crit_density'],
        'kw_low_density':     df['kw_low_density'],
        'has_escalation':     df['has_escalation'],
        'domain_tier':        df['domain_tier'],
        'text_len':           df['text_len'],
        'word_count':         df['word_count'],
        'nlp_minus_assigned': df['nlp_score'] - df['assigned_num'],
        'rt_minus_assigned':  df['rt_score'].clip(1,4) - df['assigned_num'],
        'abs_nlp_delta':      (df['nlp_score'] - df['assigned_num']).abs(),
        'abs_rt_delta':       (df['rt_score'].clip(1,4) - df['assigned_num']).abs(),
        'max_delta':          np.maximum((df['nlp_score'] - df['assigned_num']).abs(),
                                         (df['rt_score'].clip(1,4) - df['assigned_num']).abs()),
    })

    struct_df = pd.concat([num_feats, chan_dummies, type_dummies, pri_dummies], axis=1)
    # Reindex to training columns
    struct_cols = [c for c in cfg['struct_cols'] if not c.startswith(('chan_','type_','pri_'))]
    all_struct_cols = list(num_feats.columns) + \
                      [c for c in cfg['struct_cols'] if c.startswith('chan_')] + \
                      [c for c in cfg['struct_cols'] if c.startswith('type_')] + \
                      [c for c in cfg['struct_cols'] if c.startswith('pri_')]
    struct_df = struct_df.reindex(columns=cfg['struct_cols'], fill_value=0.0)
    struct_scaled = scaler.transform(struct_df.values.astype(float))

    text_col = (df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna(''))
    X_text   = tfidf.transform(text_col)
    X_all    = hstack([X_text, csr_matrix(struct_scaled)])

    lr_prob  = lr.predict_proba(X_all)[:, 1]
    rf_prob  = rf.predict_proba(struct_scaled)[:, 1]
    blended  = 0.65 * lr_prob + 0.35 * rf_prob
    y_pred   = (blended >= threshold).astype(int)

    df['pred_label']    = y_pred
    df['pred_prob']     = blended
    df['is_mismatch']   = y_pred.astype(bool)

    # Generate dossiers for flagged tickets
    dossiers = []
    for i, row in df[df['is_mismatch']].iterrows():
        text  = str(row.get('Ticket Subject','')) + ' ' + str(row.get('Ticket Description',''))
        t     = text.lower()
        kw_ev = []
        for kw in CRITICAL_KEYWORDS:
            if kw in t: kw_ev.append({'signal':'keyword','value':kw,'weight':'critical_signal'})
        for kw in HIGH_KEYWORDS[:8]:
            if kw in t and len(kw_ev) < 5:
                kw_ev.append({'signal':'keyword','value':kw,'weight':'high_signal'})
        kw_ev = kw_ev[:4]
        rt_h  = round(float(row.get('Resolution Time',0)),1)
        rt_ev = {'signal':'resolution_time','value':f"{rt_h}h",
                 'interpretation':f"Resolution time {rt_h}h — supports inferred severity {row['inferred_severity']}"}

        mtype = row['mismatch_type']
        assigned = row.get('Ticket Priority','Unknown')
        inferred = row['inferred_severity']
        chan  = row.get('Ticket Channel','Unknown')

        if mtype == 'Hidden Crisis':
            ca = (f"Text contains critical-severity signals conflicting with assigned label {assigned}. "
                  f"Submitted via {chan}; resolution time {rt_h}h supports inferred severity {inferred}. "
                  f"Recommend re-triage to {inferred}.")
        else:
            ca = (f"Language is consistent with {inferred}-level urgency, yet labeled {assigned}. "
                  f"Resolution time {rt_h}h (via {chan}) corroborates lower actual severity. "
                  f"Priority may be inflated.")

        delta = int(row['severity_delta'])
        dossiers.append({
            'ticket_id':          row.get('ticket_id', f"ROW-{i}"),
            'assigned_priority':  assigned,
            'inferred_severity':  inferred,
            'mismatch_type':      mtype,
            'severity_delta':     f"+{delta}" if delta > 0 else str(delta),
            'feature_evidence':   kw_ev + [rt_ev],
            'constraint_analysis': ca,
            'confidence':         f"{row['pred_prob']*100:.1f}%",
        })

    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        df.to_csv(f'{output_dir}/predictions.csv', index=False)
        with open(f'{output_dir}/dossiers.json','w') as f:
            json.dump(dossiers, f, indent=2)
        print(f"Saved predictions → {output_dir}/predictions.csv")
        print(f"Saved dossiers    → {output_dir}/dossiers.json")

    return df, dossiers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SIA — Priority Mismatch Detector')
    parser.add_argument('--input',  required=True, help='Path to input CSV')
    parser.add_argument('--output', default='./sia_output', help='Output directory')
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    print(f"Loaded {len(df_in)} tickets from {args.input}")
    df_out, dossiers = predict(df_in, args.output)
    flagged = df_out['is_mismatch'].sum()
    print(f"\nResults: {flagged}/{len(df_out)} tickets flagged as mismatches ({flagged/len(df_out):.1%})")
    if dossiers:
        print("\nSample dossier:")
        print(json.dumps(dossiers[0], indent=2))
