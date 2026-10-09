import math
import numpy as np
from .geometry import h_from_phi_kappa
from .objectives import objective

def _m0_profile_grid(prep,cfg,phis,kappas,batch_size=256):
    """Evaluate the unchanged scalar M0 loss in candidate batches."""
    phi_grid,kappa_grid=np.meshgrid(phis,kappas)
    raw=np.column_stack([np.cos(phi_grid.ravel()),np.sin(phi_grid.ravel()),kappa_grid.ravel()])
    hs=raw/(np.linalg.norm(raw,axis=1,keepdims=True)+1e-300)
    values=np.zeros(len(hs),float);nu=float(cfg['model']['student_t_nu']);sigma=float(cfg['model']['sigma_abs'])
    for start in range(0,len(hs),batch_size):
        h=hs[start:start+batch_size]; subtotal=np.zeros(len(h),float)
        for arc in prep.arcs:
            # Shapes: candidate x point x coordinate. This is the same
            # projective tangent residual used by absolute_residual/objective.
            v=h[:,None,:2]-h[:,None,2:3]*arc.points[None,:,:]
            e=np.sum(arc.tangents[None,:,:]*v,axis=2)/np.sqrt(np.sum(v*v,axis=2)+1e-18)
            q=arc.ds/(np.sum(arc.ds)+1e-12)
            robust=0.5*(nu+1.0)*np.log1p((e/sigma)**2/nu)
            subtotal+=arc.omega*np.sum(robust*q[None,:],axis=1)
        values[start:start+len(h)]=subtotal
    return values.reshape(len(kappas),len(phis))

def support_components(support):
    """Count 4-connected support modes; phi wraps, kappa does not."""
    support=np.asarray(support,bool);seen=np.zeros_like(support,bool);sizes=[]
    rows,cols=support.shape
    for i,j in np.argwhere(support):
        if seen[i,j]:continue
        stack=[(int(i),int(j))];seen[i,j]=True;size=0
        while stack:
            r,c=stack.pop();size+=1
            for rr,cc in ((r-1,c),(r+1,c),(r,(c-1)%cols),(r,(c+1)%cols)):
                if 0<=rr<rows and support[rr,cc] and not seen[rr,cc]:seen[rr,cc]=True;stack.append((rr,cc))
        sizes.append(size)
    return sorted(sizes,reverse=True)

def profile(prep,cfg,model='m0'):
    ph=np.linspace(-math.pi,math.pi,int(cfg['profile']['phi_bins']),endpoint=False)
    kb=np.r_[0.0,np.geomspace(1e-3,float(cfg['profile']['kappa_max']),int(cfg['profile']['kappa_bins'])-1)]
    if model=='m0':
        Z=_m0_profile_grid(prep,cfg,ph,kb)
    else:
        Z=np.empty((len(kb),len(ph)))
        for i,k in enumerate(kb):
            for j,p in enumerate(ph):Z[i,j]=objective(h_from_phi_kappa(p,k),prep,cfg,model)
    m=float(Z.min()); S=Z<=m+float(cfg['profile']['support_delta']); includes_axis=bool(np.any(S[0]))
    # circular phi support width by largest gap
    ids=np.argwhere(S); pp=ph[ids[:,1]] if len(ids) else np.array([])
    if len(pp):
        q=np.sort(np.mod(pp,2*np.pi)); gaps=np.diff(np.r_[q,q[0]+2*np.pi]); width=2*np.pi-np.max(gaps)
    else:width=2*np.pi
    sizes=support_components(S);supported_kappa=kb[np.any(S,axis=1)]
    return {'min_loss':m,'includes_axis':includes_axis,'phi_width_deg':math.degrees(width),'loss':Z,'phis':ph,'kappas':kb,'support':S,
            'support_component_sizes':sizes,'support_component_count':len(sizes),'support_cells':int(np.sum(S)),
            'support_kappa_min':float(supported_kappa.min()) if len(supported_kappa) else None,
            'support_kappa_max':float(supported_kappa.max()) if len(supported_kappa) else None}

def classify(prof,cfg):
    if int(prof.get('support_component_count',1))>1:return 'MULTIMODAL'
    if prof['includes_axis']:
        return 'AXIS' if prof['phi_width_deg']<=float(cfg['profile']['axis_phi_width_deg']) else 'REJECT'
    if prof['phi_width_deg']<=float(cfg['profile']['point_phi_width_deg']):
        # A narrow angular support can still be distance-ambiguous.  Use a
        # provisional ratio only; final state thresholds remain an E3 task.
        kmin=prof.get('support_kappa_min'); kmax=prof.get('support_kappa_max')
        ratio=float(kmax)/max(float(kmin),1e-12) if kmin is not None and kmax is not None else 1.0
        range_ratio=float(cfg.get('profile',{}).get('range_kappa_ratio',float('inf')))
        range_width=float(cfg.get('profile',{}).get('range_phi_width_deg',45.0))
        if ratio>=range_ratio and prof['phi_width_deg']<=range_width:
            return 'RANGE_UNCERTAIN'
        return 'POINT'
    kmin=prof.get('support_kappa_min'); kmax=prof.get('support_kappa_max')
    ratio=float(kmax)/max(float(kmin),1e-12) if kmin is not None and kmax is not None else 1.0
    range_ratio=float(cfg.get('profile',{}).get('range_kappa_ratio',float('inf')))
    if kmin is not None and kmax is not None and ratio>=range_ratio and prof['phi_width_deg']<=float(cfg.get('profile',{}).get('range_phi_width_deg',45.0)):
        return 'RANGE_UNCERTAIN'
    return 'REJECT'
