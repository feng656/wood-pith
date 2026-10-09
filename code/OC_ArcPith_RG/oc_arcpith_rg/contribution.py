from __future__ import annotations
import numpy as np
from .optimizer import solve
from .geometry import point_from_h,projective_angle,to_norm
from .objectives import objective

def _hessian_xy(prep,cfg,model,h,exclude_rings=None,step=None):
    step=float(cfg['contribution']['hessian_step']) if step is None else float(step)
    p=point_from_h(h)
    if p is None:return None
    from .geometry import h_from_point
    from .objectives import objective
    e=np.eye(2)*step; H=np.zeros((2,2)); f0=objective(h,prep,cfg,model,exclude_ring_ids=exclude_rings)
    for i in range(2):
        fp=objective(h_from_point(p+e[i]),prep,cfg,model,exclude_ring_ids=exclude_rings); fm=objective(h_from_point(p-e[i]),prep,cfg,model,exclude_ring_ids=exclude_rings); H[i,i]=(fp-2*f0+fm)/(step*step)
    fpp=objective(h_from_point(p+e[0]+e[1]),prep,cfg,model,exclude_ring_ids=exclude_rings); fpm=objective(h_from_point(p+e[0]-e[1]),prep,cfg,model,exclude_ring_ids=exclude_rings); fmp=objective(h_from_point(p-e[0]+e[1]),prep,cfg,model,exclude_ring_ids=exclude_rings); fmm=objective(h_from_point(p-e[0]-e[1]),prep,cfg,model,exclude_ring_ids=exclude_rings)
    H[0,1]=H[1,0]=(fpp-fpm-fmp+fmm)/(4*step*step); return H

def evaluate(prep,cfg,model='m0'):
    full=solve(prep,cfg,model); h=full['best']['h']; p=point_from_h(h); H=_hessian_xy(prep,cfg,model,h); ridge=float(cfg['contribution']['ridge'])
    def hessian_stats(matrix):
        if matrix is None:return None,None
        eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2+ridge*np.eye(2))
        return (float(np.sum(np.log(eigenvalues))),float(eigenvalues.min())) if np.all(eigenvalues>0) else (None,float(eigenvalues.min()))
    base_ld,base_min_eig=hessian_stats(H)
    gt=to_norm(prep.sample.pith_px,prep.center_px,prep.scale_px) if prep.sample.pith_px is not None else None
    groups={}
    for a in prep.arcs:groups.setdefault(a.ring_id,[]).append(a.arc_id)
    out=[]
    for gid,excl in list(groups.items())[:int(cfg['contribution']['max_groups'])]:
        from scipy.optimize import minimize
        from .geometry import phi_kappa_from_h,h_from_phi_kappa
        phi0,k0=phi_kappa_from_h(h); kmax=float(cfg['profile']['kappa_max'])
        def f(z): return objective(h_from_phi_kappa(float(z[0]),float(z[1])),prep,cfg,model,exclude_ring_ids={gid})
        rr=minimize(f,np.array([phi0,min(k0,kmax)]),method='L-BFGS-B',bounds=[(-np.pi,np.pi),(0.0,kmax)],options={'maxiter':int(cfg['model']['optimizer_maxiter'])})
        hr=h_from_phi_kappa(float(rr.x[0]),float(rr.x[1])); pr=point_from_h(hr); Hr=_hessian_xy(prep,cfg,model,h,{gid})
        ld,min_eig=hessian_stats(Hr)
        shift=float(np.linalg.norm(pr-p)) if p is not None and pr is not None else None
        gt_help=None
        if gt is not None and p is not None and pr is not None:gt_help=float(np.linalg.norm(pr-gt)-np.linalg.norm(p-gt))
        group_arcs=[a for a in prep.arcs if a.ring_id==gid]
        support_budget=float(sum(a.omega for a in group_arcs));support_length=float(sum(np.sum(a.ds) for a in group_arcs));point_count=int(sum(len(a.points) for a in group_arcs))
        out.append({'group_id':gid,'arc_count':len(excl),'support_budget':support_budget,'support_length_norm':support_length,'point_count':point_count,
                    'projective_shift_rad':projective_angle(h,hr),'point_shift_norm':shift,
                    'full_hessian_min_eigenvalue':base_min_eig,'deleted_hessian_min_eigenvalue':min_eig,
                    'info_logdet_drop':None if base_ld is None or ld is None else float(base_ld-ld),
                    'gt_error_increase_when_deleted':gt_help,'gt_conflict':bool(gt_help is not None and gt_help<0)})
    return out
