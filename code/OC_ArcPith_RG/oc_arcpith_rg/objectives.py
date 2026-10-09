from __future__ import annotations
import math
import numpy as np
from .geometry import absolute_residual,point_from_h

def student(z,nu=4.0): return 0.5*(nu+1.0)*np.log1p((np.asarray(z,float)**2)/nu)

def m0_components(h,prep,cfg,exclude_arc_ids=None,exclude_ring_ids=None):
    excl=set(exclude_arc_ids or []); exrings=set(exclude_ring_ids or [])
    total=0.0
    for a in prep.arcs:
        if a.arc_id in excl or a.ring_id in exrings: continue
        e=absolute_residual(h,a.points,a.tangents)
        q=a.ds/(np.sum(a.ds)+1e-12)
        total += a.omega*float(np.sum(q*student(e/float(cfg['model']['sigma_abs']),cfg['model']['student_t_nu'])))
    return {'m0':float(total),'shape':0.0,'order':0.0,'total':float(total)}

def _circular_cut(theta):
    a=np.mod(theta,2*np.pi); s=np.sort(a)
    if len(s)<2:return 0.0
    gaps=np.diff(np.r_[s,s[0]+2*np.pi]); k=int(np.argmax(gaps))
    return float((s[(k+1)%len(s)])%(2*np.pi))

def _unwrap_with_cut(theta,cut): return (np.mod(theta-cut,2*np.pi))+cut

def _profile(points,p,cut):
    d=points-p; r=np.linalg.norm(d,axis=1); th=_unwrap_with_cut(np.arctan2(d[:,1],d[:,0]),cut)
    o=np.argsort(th); th=th[o]; r=r[o]
    # collapse near duplicate angles
    keep=np.r_[True,np.diff(th)>1e-5]
    return th[keep],r[keep]

def radial_pair_residuals(h,pa,pb,cfg):
    p=point_from_h(h)
    if p is None:return np.zeros(0),np.zeros(0)
    # fade out when candidate is effectively at infinity
    kappa=1.0/(np.linalg.norm(p)+1e-12)
    if kappa<float(cfg['model']['m1_axis_taper_kappa']): return np.zeros(0),np.zeros(0)
    ta=np.arctan2(pa[:,1]-p[1],pa[:,0]-p[0]); tb=np.arctan2(pb[:,1]-p[1],pb[:,0]-p[0])
    cut=_circular_cut(np.r_[ta,tb]); tha,ra=_profile(pa,p,cut); thb,rb=_profile(pb,p,cut)
    lo=max(tha.min(),thb.min()); hi=min(tha.max(),thb.max())
    if hi-lo<math.radians(float(cfg['model']['radial_profile_min_overlap_deg'])): return np.zeros(0),np.zeros(0)
    n=int(cfg['model']['radial_profile_bins']); grid=np.linspace(lo,hi,n)
    ria=np.interp(grid,tha,ra); rib=np.interp(grid,thb,rb)
    # pa/pb follow frozen parent-ring order. Do NOT re-order by the candidate pith.
    # A wrong candidate should be allowed to violate nesting; otherwise ordering becomes self-fulfilling.
    dla=np.gradient(np.log(ria+1e-9),grid); dlb=np.gradient(np.log(rib+1e-9),grid)
    shape=dla-dlb
    gap=rib-ria
    # one-sided soft order residual; zero when nested correctly with positive gap
    order=np.maximum(0.0,-gap)
    return shape,order

def m1rpc_components(h,prep,cfg,exclude_arc_ids=None,exclude_ring_ids=None):
    exrings=set(exclude_ring_ids or [])
    base=m0_components(h,prep,cfg,exclude_arc_ids,exclude_ring_ids)
    shape_loss=0.0; order_loss=0.0; pairs=0
    nu=float(cfg['model']['student_t_nu'])
    for ra,rb in zip(prep.ring_order[:-1],prep.ring_order[1:]):
        if ra in exrings or rb in exrings or ra not in prep.points_by_ring or rb not in prep.points_by_ring: continue
        parent_shape=[]; parent_order=[]
        for pa in prep.points_by_ring[ra]:
            for pb in prep.points_by_ring[rb]:
                sh,od=radial_pair_residuals(h,pa,pb,cfg)
                if len(sh):
                    parent_shape.append(sh); parent_order.append(od)
        if not parent_shape: continue
        pairs+=1
        # Each adjacent parent-ring pair receives equal weight. Its disconnected
        # visible overlaps are observations of that same biological pair.
        shape_loss += float(np.mean(student(np.concatenate(parent_shape)/float(cfg['model']['m1_shape_sigma']),nu)))
        order_loss += float(np.mean(student(np.concatenate(parent_order)/float(cfg['model']['m1_order_sigma']),nu)))
    if pairs:
        shape_loss/=pairs; order_loss/=pairs
    total=base['m0']+float(cfg['model']['m1_lambda_shape'])*shape_loss+float(cfg['model']['m1_lambda_order'])*order_loss
    return {'m0':base['m0'],'shape':shape_loss,'order':order_loss,'pair_count':pairs,'total':float(total)}

def objective(h,prep,cfg,model='m0',exclude_arc_ids=None,exclude_ring_ids=None):
    return (m0_components if model=='m0' else m1rpc_components)(h,prep,cfg,exclude_arc_ids,exclude_ring_ids)['total']
