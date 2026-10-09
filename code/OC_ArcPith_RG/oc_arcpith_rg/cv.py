from __future__ import annotations
import copy,hashlib
from collections import defaultdict
import numpy as np
from .optimizer import solve,refine
from .metrics import solution_metrics

def fold_map(tree_ids,k,seed):
    ids=sorted(set(tree_ids),key=lambda t:hashlib.sha256((str(seed)+t).encode()).hexdigest())
    return {t:i%k for i,t in enumerate(ids)}

def _tree_median(rows,key,tree_filter=None):
    by=defaultdict(list)
    for r in rows:
        if tree_filter is not None and r['tree_id'] not in tree_filter:continue
        v=r.get(key)
        if v is not None and np.isfinite(v):by[r['tree_id']].append(float(v))
    return {t:float(np.median(v)) for t,v in by.items()}

def run_group_cv(preps,cfg):
    tree_ids=[p.sample.tree_id for p in preps]; nfold=min(int(cfg['cv']['folds']),len(set(tree_ids))); fmap=fold_map(tree_ids,nfold,int(cfg['seed']))
    # full blind M0 once
    m0_results={}; m0_metrics=[]
    for p in preps:
        r=solve(p,cfg,'m0');m0_results[p.sample.sample_id]=r;m0_metrics.append(solution_metrics(p,r)|{'model':'m0','fold':fmap[p.sample.tree_id]})
    # training-only lambda sweep using local M1RPC refinement from blind M0 basin
    tuning={}
    for lam in cfg['model']['m1_lambda_shape_grid']:
        c=copy.deepcopy(cfg);c['model']['m1_lambda_shape']=float(lam);rows=[]
        for p in preps:
            r=refine(p,c,'m1rpc',m0_results[p.sample.sample_id]['best']['h']);rows.append(solution_metrics(p,r)|{'model':'m1rpc','lambda_shape':float(lam),'fold':fmap[p.sample.tree_id]})
        tuning[float(lam)]=rows
    held=[];selected={}
    for f in range(nfold):
        train={t for t,ff in fmap.items() if ff!=f}; test={t for t,ff in fmap.items() if ff==f}
        scores=[]
        for lam,rows in tuning.items():
            tm=_tree_median(rows,'point_error_norm',train);score=float(np.median(list(tm.values()))) if tm else float('inf');scores.append((score,lam))
        scores.sort();lam=scores[0][1];selected[str(f)]=lam;c=copy.deepcopy(cfg);c['model']['m1_lambda_shape']=float(lam)
        for p in preps:
            if p.sample.tree_id not in test:continue
            r=solve(p,c,'m1rpc');held.append(solution_metrics(p,r)|{'model':'m1rpc','lambda_shape':float(lam),'fold':f})
    # paired tree-level gate
    finite_max=float(cfg['cv']['finite_true_distance_max_norm']); folds=[]
    for f in range(nfold):
        test={t for t,ff in fmap.items() if ff==f}
        a=[r for r in m0_metrics if r['tree_id'] in test and r.get('true_distance_norm',99)<=finite_max]
        b=[r for r in held if r['tree_id'] in test and r.get('true_distance_norm',99)<=finite_max]
        A=_tree_median(a,'point_error_norm');B=_tree_median(b,'point_error_norm');common=sorted(set(A)&set(B))
        imp=float(np.median([A[t]-B[t] for t in common])) if common else None
        folds.append({'fold':f,'trees':common,'point_improvement_norm':imp,'lambda_shape':selected[str(f)]})
    improved=sum(x['point_improvement_norm'] is not None and np.isfinite(x['point_improvement_norm']) and x['point_improvement_norm']>0 for x in folds)
    A=_tree_median([r for r in m0_metrics if r.get('true_distance_norm',99)<=finite_max],'point_error_norm');B=_tree_median([r for r in held if r.get('true_distance_norm',99)<=finite_max],'point_error_norm');common=set(A)&set(B)
    medimp=float(np.median([A[t]-B[t] for t in common])) if common else None
    # far axis guardrail
    Af=_tree_median([r for r in m0_metrics if r.get('true_distance_norm',0)>finite_max],'axis_error_deg');Bf=_tree_median([r for r in held if r.get('true_distance_norm',0)>finite_max],'axis_error_deg');cf=set(Af)&set(Bf)
    far_deg=float(np.median([Bf[t]-Af[t] for t in cf])) if cf else 0.0
    required=min(int(cfg['cv']['min_improved_folds']),nfold); passed=(improved>=required and medimp is not None and np.isfinite(medimp) and medimp>=float(cfg['cv']['min_median_point_improvement_norm']) and far_deg<=float(cfg['cv']['max_far_axis_degradation_deg']))
    decision={'pass':passed,'decision':'KEEP_M1RPC' if passed else 'USE_M0_BASELINE','selected_lambda_by_fold':selected,'folds':folds,'nfold':nfold,'required_improved_folds':required,'improved_folds':improved,'median_point_improvement_norm':medimp,'far_axis_degradation_deg':far_deg}
    return m0_metrics,held,tuning,decision
