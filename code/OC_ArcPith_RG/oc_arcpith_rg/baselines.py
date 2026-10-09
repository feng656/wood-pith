"""Stage-1 S0 comparison baselines.

These are deliberately separate from the production M0/M1-RPC path so a
paired experiment can freeze the same observations and search settings.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize

from .geometry import absolute_residual,h_from_phi_kappa,h_from_point,phi_kappa_from_h,point_from_h,projective_angle
from .objectives import m0_components,objective

def b0_one_ring_pseudocenters(prep):
    """B0: one pseudocenter per parent ring, then projective medoid.

    A ring's pseudocenter is the centroid of all its visible normalized points.
    The medoid is selected among those ring hypotheses using RP² distance, so
    no pooled point count can dominate the baseline.
    """
    centers=[]
    for rid in prep.ring_order:
        fragments=prep.points_by_ring.get(rid,[])
        if fragments:
            points=np.vstack(fragments)
            centers.append(h_from_point(np.mean(points,axis=0)))
    if not centers:return None
    scores=[sum(projective_angle(h,g) for g in centers) for h in centers]
    return centers[int(np.argmin(scores))]

def b1_components(h,prep,cfg):
    """B1 pooled micro-arcs objective without equal-parent normalization."""
    nu=float(cfg['model']['student_t_nu']); sigma=float(cfg['model']['sigma_abs'])
    masses=np.asarray([float(np.sum(a.ds)) for a in prep.arcs],float)
    total_mass=float(np.sum(masses))
    if not len(masses) or not np.isfinite(total_mass) or total_mass<=0:return {'total':float('inf')}
    total=0.0
    for a,mass in zip(prep.arcs,masses):
        e=absolute_residual(h,a.points,a.tangents); q=a.ds/(np.sum(a.ds)+1e-12)
        robust=0.5*(nu+1.0)*np.log1p((e/sigma)**2/nu)
        total += (mass/total_mass)*float(np.sum(q*robust))
    return {'total':float(total),'m0':float(total),'shape':0.0,'order':0.0}

def b1_objective(h,prep,cfg):
    return b1_components(h,prep,cfg)['total']

def b0_result(prep,cfg):
    h=b0_one_ring_pseudocenters(prep)
    return {'h':h,'loss':None if h is None else float(objective(h,prep,cfg,'m0')),
            'point':None if h is None else point_from_h(h)}

def _grid_candidates(prep,cfg,model,phi_bins,kappa_bins,kappa_max):
    phis=np.linspace(-math.pi,math.pi,int(phi_bins),endpoint=False)
    kappas=np.r_[0.0,np.geomspace(1e-3,float(kappa_max),int(kappa_bins)-1)]
    # Evaluate the whole grid in batches.  This is algebraically identical to
    # the scalar objectives but avoids a Python loop over every candidate.
    phi_grid,kappa_grid=np.meshgrid(phis,kappas)
    raw=np.column_stack([np.cos(phi_grid.ravel()),np.sin(phi_grid.ravel()),kappa_grid.ravel()])
    hs=raw/(np.linalg.norm(raw,axis=1,keepdims=True)+1e-300)
    nu=float(cfg['model']['student_t_nu']);sigma=float(cfg['model']['sigma_abs'])
    values=np.zeros(len(hs),float)
    masses=np.asarray([float(np.sum(a.ds)) for a in prep.arcs],float)
    total_mass=float(np.sum(masses))
    for start in range(0,len(hs),256):
        h=hs[start:start+256]; subtotal=np.zeros(len(h),float)
        for arc,mass in zip(prep.arcs,masses):
            v=h[:,None,:2]-h[:,None,2:3]*arc.points[None,:,:]
            e=np.sum(arc.tangents[None,:,:]*v,axis=2)/np.sqrt(np.sum(v*v,axis=2)+1e-18)
            q=arc.ds/(np.sum(arc.ds)+1e-12)
            robust=0.5*(nu+1.0)*np.log1p((e/sigma)**2/nu)
            weight=arc.omega if model=='m0' else (mass/total_mass if total_mass>0 else 0.0)
            subtotal += float(weight)*np.sum(robust*q[None,:],axis=1)
        values[start:start+len(h)]=subtotal
    idx=int(np.argmin(values));return float(values[idx]),hs[idx]

def b3_dense_search(prep,cfg,phi_bins=144,kappa_bins=80,kappa_max=None):
    """B3 dense/offline reference search followed by local refinement."""
    kmax=float(kappa_max if kappa_max is not None else cfg.get('profile',{}).get('kappa_max',50.0))
    loss,h0=_grid_candidates(prep,cfg,'m0',phi_bins,kappa_bins,kmax)
    if h0 is None:return None,float('inf')
    phi,kappa=phi_kappa_from_h(h0); kappa=min(max(float(kappa),0.0),kmax)
    def fn(z):return float(objective(h_from_phi_kappa(float(z[0]),float(z[1])),prep,cfg,'m0'))
    res=minimize(fn,np.array([phi,kappa]),method='L-BFGS-B',bounds=[(-math.pi,math.pi),(0.0,kmax)],options={'maxiter':max(200,int(cfg.get('model',{}).get('optimizer_maxiter',300)))})
    h=h_from_phi_kappa(float(res.x[0]),float(res.x[1]));return h,float(res.fun)

def compare_baselines(prep,cfg,h_b2=None,**kwargs):
    """Return B0/B1/B2/B3 results on one frozen crop."""
    if h_b2 is None:
        from .optimizer import solve
        h_b2=solve(prep,cfg,'m0')['best']['h']
    b0=b0_result(prep,cfg)
    b1_loss,b1_h=_grid_candidates(prep,cfg,'b1',kwargs.get('b1_phi_bins',72),kwargs.get('b1_kappa_bins',40),float(kwargs.get('kappa_max',cfg.get('profile',{}).get('kappa_max',50.0))))
    b3_h,b3_loss=b3_dense_search(prep,cfg,kwargs.get('b3_phi_bins',144),kwargs.get('b3_kappa_bins',80),kwargs.get('kappa_max'))
    rows={'B0':b0,'B1':{'h':b1_h,'loss':b1_loss,'point':point_from_h(b1_h)},
          'B2':{'h':h_b2,'loss':float(objective(h_b2,prep,cfg,'m0')),'point':point_from_h(h_b2)},
          'B3':{'h':b3_h,'loss':b3_loss,'point':point_from_h(b3_h)}}
    for row in rows.values():
        row['d_to_b2']=None if row.get('h') is None else projective_angle(row['h'],h_b2)
    return rows
