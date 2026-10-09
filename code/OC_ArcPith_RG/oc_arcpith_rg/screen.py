from __future__ import annotations
from collections import defaultdict
import numpy as np
from .candidates import build
from .objectives import objective

def screen(prep,cfg,model='m0'):
    lib=build(prep,cfg); rows=[]
    for c in lib:
        L=float(objective(c['h'],prep,cfg,model)); rows.append({k:v for k,v in c.items() if k!='h'}|{'loss':L})
    gt=next(r for r in rows if r['kind']=='truth'); gl=gt['loss']; by=defaultdict(list)
    for r in rows:
        if r['kind']!='truth': by[r['kind']].append(r['loss']-gl)
    types={k:{'min_gap':float(min(v)),'median_gap':float(np.median(v)),'truth_beats_fraction':float(np.mean(np.asarray(v)>0))} for k,v in by.items()}
    allg=[r['loss']-gl for r in rows if r['kind']!='truth']
    # global excludes local translation to distinguish separability from GT stationarity/model bias
    globalg=[]
    for r in rows:
        if r['kind'] in ('direction','distance','reverse','random'): globalg.append(r['loss']-gl)
    return {'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,'model':model,'gt_loss':gl,
            'hard_gap':float(min(allg)),'global_hard_gap':float(min(globalg)) if globalg else None,
            'truth_beats_fraction':float(np.mean(np.asarray(allg)>0)),'global_truth_beats_fraction':float(np.mean(np.asarray(globalg)>0)) if globalg else None,
            'candidate_types':types,'candidates':rows}

def summarize(rows,cfg):
    by=defaultdict(list)
    for r in rows: by[r['tree_id']].append(r)
    trees=[]
    tol=float(cfg['candidate_screen']['local_loss_tolerance'])
    for tid,rr in sorted(by.items()):
        gtruth=float(np.median([x['global_truth_beats_fraction'] for x in rr]))
        dirgap=float(np.median([x['candidate_types'].get('direction',{}).get('min_gap',np.nan) for x in rr]))
        rangegap=float(np.median([x['candidate_types'].get('distance',{}).get('min_gap',np.nan) for x in rr]))
        transgap=float(np.median([x['candidate_types'].get('translation',{}).get('min_gap',np.nan) for x in rr]))
        trees.append({'tree_id':tid,'global_truth_beats_fraction':gtruth,'direction_gap':dirgap,'range_gap':rangegap,'translation_gap':transgap,'local_bias_flag':bool(transgap<-tol)})
    global_pass=bool(trees and np.median([t['global_truth_beats_fraction'] for t in trees])>=float(cfg['candidate_screen']['global_truth_beats_min']))
    dir_pass=float(np.mean([t['direction_gap']>0 for t in trees]))>=float(cfg['candidate_screen']['direction_tree_pass_fraction']) if trees else False
    range_pass=float(np.mean([t['range_gap']>0 for t in trees]))>=float(cfg['candidate_screen']['range_tree_pass_fraction']) if trees else False
    local_ok=float(np.mean([not t['local_bias_flag'] for t in trees]))>=float(cfg['candidate_screen']['local_bias_tree_pass_fraction']) if trees else False
    if not global_pass or not dir_pass: decision='NO_GO_GLOBAL_GEOMETRY'
    elif not local_ok: decision='BIAS_REPAIR_REQUIRED'
    else: decision='PASS_TO_BLIND_INVERSION'
    return {'tree_metrics':trees,'global_pass':global_pass,'direction_pass':dir_pass,'range_pass':range_pass,'local_stationarity_pass':local_ok,'decision':decision}
