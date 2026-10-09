"""Candidate-free feasibility checks used before expensive S0 search."""
from __future__ import annotations

import numpy as np

from .evidence import split_parent_rings

def _features(prep):
    rings=[rid for rid in prep.ring_order if rid in prep.points_by_ring and
           any(a.ring_id==rid for a in prep.arcs)]
    arc_lengths=[float(np.sum(a.ds)) for a in prep.arcs]
    total_arc=float(np.sum(arc_lengths)) if arc_lengths else 0.0
    tangents=np.vstack([a.tangents for a in prep.arcs]) if prep.arcs else np.empty((0,2))
    if len(tangents):
        # Tangent orientation is axial: t and -t represent the same direction.
        theta=np.mod(np.arctan2(tangents[:,1],tangents[:,0]),np.pi)
        bins=max(4,min(36,int(np.sqrt(len(theta))*2)))
        hist,_=np.histogram(theta,bins=bins,range=(0,np.pi),weights=None)
        angular_coverage=float(np.mean(hist>0))
        T=tangents.T@tangents/max(len(tangents),1)
        eig=np.linalg.eigvalsh(T); eig=np.maximum(eig,0)
        lambda_min=float(eig[0]); condition=float(eig[-1]/max(eig[0],1e-12))
    else:
        angular_coverage=0.0; lambda_min=0.0; condition=float('inf')
    # Parent-ring count is the conservative independent support proxy.  Point
    # duplication and fragment splitting therefore cannot inflate it.
    return {'eligible_ring_count':len(rings),'total_arc_norm':total_arc,
            'angular_coverage':angular_coverage,'lambda_min':lambda_min,
            'condition':condition,'n_seg_equiv':len(rings)}

def preflight(prep, cfg, level='P0'):
    """Return a candidate-free route and features; never computes a candidate loss."""
    level=str(level).upper()
    if level not in {'P0','P1','P2','P3'}: raise ValueError('level must be P0, P1, P2 or P3')
    f=_features(prep); pcfg=cfg.get('preflight',{})
    min_rings=int(pcfg.get('min_eligible_rings',2)); min_arc=float(pcfg.get('min_total_arc_norm',0.05))
    min_cov=float(pcfg.get('min_angular_coverage',0.20)); min_lam=float(pcfg.get('min_lambda_min',0.02))
    min_seg=int(pcfg.get('min_n_seg_equiv',2))
    invalid=not np.isfinite(f['total_arc_norm']) or f['eligible_ring_count']==0
    if invalid: route='INVALID'
    elif f['eligible_ring_count']<min_rings: route='INSUFFICIENT'
    elif level in {'P1','P2','P3'} and f['total_arc_norm']<min_arc: route='INSUFFICIENT'
    elif level in {'P2','P3'} and f['angular_coverage']<min_cov: route='AXIS_ONLY'
    elif level in {'P2','P3'} and f['lambda_min']<min_lam: route='AXIS_ONLY'
    elif level=='P3' and f['n_seg_equiv']<min_seg: route='INSUFFICIENT'
    else: route='POINT_CANDIDATE'
    partition=split_parent_rings(prep,cfg)
    f.update({'audit_feasible':partition.audit_feasible,
              'fit_ring_ids':partition.fit_ring_ids,
              'buffer_ring_ids':partition.buffer_ring_ids,
              'audit_ring_ids':partition.audit_ring_ids})
    if not partition.audit_feasible: f['audit_route']='NO_POWERED_AUDIT_AVAILABLE'
    else: f['audit_route']='BUFFERED_AUDIT_FEASIBLE'
    if route in {'INVALID','INSUFFICIENT'} or not partition.audit_feasible: final='REJECT'
    elif route=='AXIS_ONLY': final='AXIS_ONLY'
    else: final='POINT_CANDIDATE'
    return {'level':level,'route':final,'features':f,'partition_reason':partition.reason}
