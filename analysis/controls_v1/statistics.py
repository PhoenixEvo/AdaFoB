"""Frozen paired-query-patient gap closure and decision definitions."""
import numpy as np,pandas as pd
from .core import BASE,ZERO,GREEDY_REF,BUDGETS
ELIGIBLE=sorted([BASE,ZERO,*[f'prefix{k}' for k in BUDGETS],*[f'fps{k}' for k in BUDGETS],
                 *[f'boundary_{v}{k}' for v in ['near','far'] for k in BUDGETS],*[f'random{k}_mean5' for k in BUDGETS]])
def add_seed_means(df):
    rows=[]
    keys=['episode_id','fold','patient_id','organ','post']
    values=[v for v in df if v not in keys+['arm'] and pd.api.types.is_numeric_dtype(df[v])]
    for k in BUDGETS:
        g=df[df.arm.isin([f'random{k}_s{s}' for s in range(42,47)])]
        assert g.groupby(keys).size().eq(5).all()
        m=g.groupby(keys)[values].mean().reset_index();m['arm']=f'random{k}_mean5';rows.append(m)
    return pd.concat([df,*rows],ignore_index=True)
def interval(x):return np.quantile(x,[.025,.975]).tolist()
def gap_analysis(df,replicates=5000,seed=42):
    d=df[df.post=='cc'];pm=d.groupby(['patient_id','organ','arm'])[['dice','hd95','recall']].mean().groupby(['patient_id','arm']).mean()
    arms=ELIGIBLE+[GREEDY_REF]
    patients=sorted(d.patient_id.unique());tables={metric:pm[metric].unstack('arm').reindex(index=patients,columns=arms).to_numpy() for metric in ['dice','hd95','recall']}
    assert all(np.isfinite(x).all() for x in tables.values())
    rng=np.random.default_rng(seed);sample=rng.integers(0,len(patients),size=(replicates,len(patients)))
    point={k:v.mean(0) for k,v in tables.items()};boot={k:v[sample].mean(1) for k,v in tables.items()}
    b=arms.index(BASE);g=arms.index(GREEDY_REF);den=point['dice'][g]-point['dice'][b];bd=boot['dice'][:,g]-boot['dice'][:,b]
    valid=bd>0;best_index=int(np.argmax(point['dice'][:len(ELIGIBLE)]));best=arms[best_index]
    selection=np.argmax(boot['dice'][:,:len(ELIGIBLE)],axis=1)
    result=[]
    for i,arm in enumerate(ELIGIBLE):
        gap=float((point['dice'][i]-point['dice'][b])/den) if den>0 else None
        ci=interval((boot['dice'][:,i]-boot['dice'][:,b])/bd) if valid.all() else None
        result.append({'arm':arm,'gap_closure':gap,'ci95':ci,'undefined_replicates':int((~valid).sum()),
                       'gain_pp':float((point['dice'][i]-point['dice'][b])*100),'gain_ci95_pp':interval((boot['dice'][:,i]-boot['dice'][:,b])*100)})
    residual=(boot['dice'][:,g]-boot['dice'][np.arange(replicates),selection])*100
    hd=point['hd95'][g]-point['hd95'][best_index];rec=point['recall'][g]-point['recall'][best_index]
    closure=(point['dice'][best_index]-point['dice'][b])/den if den>0 else None
    best_ci=interval((boot['dice'][np.arange(replicates),selection]-boot['dice'][:,b])/bd) if valid.all() else None
    no_hidden_point_degradation=bool(hd<=0 and rec>=0)
    branch='REVIEW'
    if closure is not None and closure>=.5:branch='STOP-SELECTOR'
    elif closure is not None and valid.all() and closure<.25 and interval(residual)[0]>1 and no_hidden_point_degradation:branch='CONTINUE'
    return {'best_simple_control':best,'corrected_gap_pp':float(den*100),'corrected_gap_ci95_pp':interval(bd*100),
            'best_gap_closure':float(closure) if closure is not None else None,'best_reselected_ci95':best_ci,
            'undefined_replicates':int((~valid).sum()),'greedy_minus_best_ci95_pp':interval(residual),
            'greedy_minus_best_point_pp':float((point['dice'][g]-point['dice'][best_index])*100),
            'greedy_minus_best_hd95_mm':float(hd),'greedy_minus_best_recall':float(rec),
            'greedy_minus_best_hd95_ci95':interval(boot['hd95'][:,g]-boot['hd95'][:,best_index]),
            'greedy_minus_best_recall_ci95':interval(boot['recall'][:,g]-boot['recall'][:,best_index]),
            'no_hidden_point_degradation':no_hidden_point_degradation,'branch':branch,'per_control':result,
            'bootstrap_replicates':replicates,'seed':seed,'conditional':'fixed supports and checkpoints; exploratory pointwise intervals; selection-aware winner reselected per replicate'}