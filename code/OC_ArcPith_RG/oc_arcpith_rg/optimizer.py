from __future__ import annotations
import math,hashlib
import numpy as np
from scipy.optimize import minimize
from .geometry import h_from_point,h_from_phi_kappa,phi_kappa_from_h,point_from_h
from .objectives import objective

def svd_seed(prep):
    rows=[]; w=[]
    for a in prep.arcs:
        den=np.sum(a.ds)+1e-12
        for x,t,ds in zip(a.points,a.tangents,a.ds):
            rows.append([t[0],t[1],-float(t@x)]); w.append(a.omega*ds/den)
    if len(rows)<3:return h_from_phi_kappa(0,0)
    A=np.asarray(rows)*np.sqrt(np.asarray(w))[:,None]
    _,_,vh=np.linalg.svd(A,full_matrices=False)
    h=vh[-1]
    if h[2]<0:h=-h
    return h/np.linalg.norm(h)

def ransac_seeds(prep,cfg,rng):
    pts=np.vstack([a.points for a in prep.arcs]); tan=np.vstack([a.tangents for a in prep.arcs])
    cand=[]
    for _ in range(int(cfg['model']['ransac_trials'])):
        if len(pts)<2:break
        i,j=rng.choice(len(pts),2,replace=False); A=np.vstack([tan[i],tan[j]]); b=np.array([tan[i]@pts[i],tan[j]@pts[j]])
        if abs(np.linalg.det(A))<1e-4:continue
        p=np.linalg.solve(A,b); h=h_from_point(p); score=objective(h,prep,cfg,'m0'); cand.append((score,h))
    cand.sort(key=lambda z:z[0]); return [h for _,h in cand[:int(cfg['model']['ransac_keep'])]]

def solve(prep,cfg,model='m0',warm=None,seed=None):
    sid=prep.sample.sample_id; hs=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16); rng=np.random.default_rng((cfg['seed'] if seed is None else seed)^hs)
    starts=[svd_seed(prep)]+ransac_seeds(prep,cfg,rng)
    # axis + finite far starts from tangent null direction
    T=np.vstack([a.tangents for a in prep.arcs]); C=T.T@T/max(len(T),1); _,V=np.linalg.eigh(C); u=V[:,0]; ph=math.atan2(u[1],u[0])
    starts += [h_from_phi_kappa(ph,0),h_from_phi_kappa(ph+math.pi,0),h_from_phi_kappa(ph,0.2),h_from_phi_kappa(ph+math.pi,0.2)]
    if warm:starts=list(warm)+starts
    for _ in range(int(cfg['model']['random_starts'])):starts.append(h_from_phi_kappa(rng.uniform(-math.pi,math.pi),10**rng.uniform(-1.5,0.8)))
    sols=[]
    for h0 in starts:
        phi,k=phi_kappa_from_h(h0); kmax=float(cfg['profile']['kappa_max']); k=min(k if np.isfinite(k) else kmax,kmax)
        def f(z):return objective(h_from_phi_kappa(float(z[0]),float(z[1])),prep,cfg,model)
        r=minimize(f,np.array([phi,k]),method='L-BFGS-B',bounds=[(-math.pi,math.pi),(0.0,kmax)],options={'maxiter':int(cfg['model']['optimizer_maxiter'])})
        h=h_from_phi_kappa(float(r.x[0]),float(r.x[1])); p=point_from_h(h); sols.append({'h':h,'loss':float(r.fun),'success':bool(r.success or np.linalg.norm(getattr(r,'jac',np.array([99,99])))<1e-3),'pith':p})
    sols.sort(key=lambda z:z['loss']); return {'best':sols[0] if sols else None,'solutions':sols,'success_fraction':float(np.mean([s['success'] for s in sols])) if sols else 0.0}


def refine(prep,cfg,model,h0):
    """Local refinement used only inside training-fold lambda tuning."""
    phi,k=phi_kappa_from_h(h0); kmax=float(cfg['profile']['kappa_max']); k=min(k if np.isfinite(k) else kmax,kmax)
    def f(z):return objective(h_from_phi_kappa(float(z[0]),float(z[1])),prep,cfg,model)
    r=minimize(f,np.array([phi,k]),method='L-BFGS-B',bounds=[(-math.pi,math.pi),(0.0,kmax)],options={'maxiter':int(cfg['model']['optimizer_maxiter'])})
    h=h_from_phi_kappa(float(r.x[0]),float(r.x[1])); return {'best':{'h':h,'loss':float(r.fun),'success':bool(r.success),'pith':point_from_h(h)},'solutions':[],'success_fraction':1.0 if r.success else 0.0,'refinement_only':True}
