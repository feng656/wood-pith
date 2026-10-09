from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .geometry import to_norm,h_from_point,radial_tangential_error
from .objectives import objective

def audit(prep,cfg,model='m0'):
    if prep.sample.pith_px is None:return None
    gt=to_norm(prep.sample.pith_px,prep.center_px,prep.scale_px)
    rad=float(cfg['bias_audit']['radius_fraction']); n=int(cfg['bias_audit']['coarse_grid'])
    xs=np.linspace(-rad,rad,n); best=(float('inf'),gt.copy())
    for dx in xs:
        for dy in xs:
            if dx*dx+dy*dy>rad*rad+1e-15:continue
            p=gt+np.array([dx,dy]); L=objective(h_from_point(p),prep,cfg,model)
            if L<best[0]:best=(L,p)
    def f(p):return objective(h_from_point(np.asarray(p)),prep,cfg,model)
    constraint={'type':'ineq','fun':lambda p:rad*rad-float(np.sum((np.asarray(p)-gt)**2))}
    res=minimize(f,best[1],method='SLSQP',bounds=[(gt[0]-rad,gt[0]+rad),(gt[1]-rad,gt[1]+rad)],constraints=[constraint],options={'maxiter':500,'ftol':1e-10})
    candidate=np.asarray(res.x,float)
    feasible=bool(np.all(np.isfinite(candidate)) and np.linalg.norm(candidate-gt)<=rad+1e-6)
    p=candidate if feasible and f(candidate)<=best[0]+1e-9 else best[1]
    b=p-gt; radial,tangential=radial_tangential_error(p,gt)
    Lgt=f(gt); Lhat=f(p); mag=float(np.linalg.norm(b)); tol=float(cfg['bias_audit']['gt_tolerance_fraction'])
    return {'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,'model':model,'gt':gt.tolist(),'local_optimum':p.tolist(),
            'bias_norm':mag,'bias_px':mag*prep.scale_px,'radial_bias_norm':radial,'tangential_bias_norm':tangential,
            'loss_gt':float(Lgt),'loss_local':float(Lhat),'loss_gain':float(Lgt-Lhat),'within_gt_tolerance':bool(mag<=tol),'optimizer_success':bool(res.success and feasible)}

def compare_repairs(m0,m1,cfg):
    q={r['sample_id']:r for r in m1}; rows=[]
    for a in m0:
        if a['sample_id'] not in q:continue
        b=q[a['sample_id']]; red=1-b['bias_norm']/(a['bias_norm']+1e-12)
        rows.append({'sample_id':a['sample_id'],'tree_id':a['tree_id'],'m0_bias':a['bias_norm'],'m1_bias':b['bias_norm'],'reduction':red})
    by={}
    for r in rows:by.setdefault(r['tree_id'],[]).append(r['reduction'])
    tree=[float(np.median(v)) for v in by.values()]
    good=float(np.mean(np.asarray(tree)>=float(cfg['bias_audit']['repair_min_bias_reduction']))) if tree else 0.0
    passed=good>=float(cfg['bias_audit']['repair_min_tree_fraction'])
    return {'paired_samples':len(rows),'tree_count':len(tree),'median_tree_bias_reduction':float(np.median(tree)) if tree else None,'tree_pass_fraction':good,'pass':passed,'decision':'KEEP_M1RPC' if passed else 'M1RPC_NOT_PROVEN'}
