import math
import numpy as np
from .geometry import to_norm,point_from_h,phi_kappa_from_h

def solution_metrics(prep,result):
    b=result['best']; out={'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,'success_fraction':result['success_fraction'],'loss':b['loss']}
    if prep.sample.pith_px is None:return out
    gt=to_norm(prep.sample.pith_px,prep.center_px,prep.scale_px); p=point_from_h(b['h']); phi,k=phi_kappa_from_h(b['h']); gtphi=math.atan2(gt[1],gt[0]) if np.linalg.norm(gt)>1e-12 else 0.0
    d=abs(((phi-gtphi+math.pi/2)%math.pi)-math.pi/2); out['axis_error_deg']=math.degrees(d)
    out['true_distance_norm']=float(np.linalg.norm(gt)); out['pred_kappa']=k
    if p is not None:out['point_error_norm']=float(np.linalg.norm(p-gt)); out['point_error_px']=out['point_error_norm']*prep.scale_px
    else:out['point_error_norm']=None; out['point_error_px']=None
    return out
