"""Evidence partitioning and fixed-hypothesis independent audit helpers."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np

from .types import PreparedSample
from .objectives import objective

@dataclass(frozen=True)
class EvidencePartition:
    fit_ring_ids: tuple[str, ...]
    buffer_ring_ids: tuple[str, ...]
    audit_ring_ids: tuple[str, ...]
    reason: str = 'OK'

    @property
    def fit_feasible(self):
        return bool(self.fit_ring_ids)

    @property
    def audit_feasible(self):
        return bool(self.audit_ring_ids)

def split_parent_rings(prep: PreparedSample, cfg: dict) -> EvidencePartition:
    """Create a deterministic fit|buffer|audit split using parent rings only.

    No candidate, loss, profile, or ground truth is read.  The default holds
    out outer rings and inserts a buffer immediately before them.  Small
    samples are reported as underpowered instead of silently reusing fit data.
    """
    observed=[rid for rid in prep.ring_order if rid in prep.points_by_ring and
              any(a.ring_id == rid for a in prep.arcs)]
    ecfg=cfg.get('evidence',{})
    min_fit=int(ecfg.get('min_fit_rings',2)); min_audit=int(ecfg.get('min_audit_rings',1))
    buffer_n=int(ecfg.get('buffer_rings',1)); audit_n=int(ecfg.get('audit_rings',0))
    if audit_n<=0:
        audit_n=max(min_audit,int(math.ceil(len(observed)*float(ecfg.get('audit_fraction',0.25)))))
    if len(observed) < min_fit + min_audit:
        return EvidencePartition(tuple(observed),tuple(),tuple(), 'UNDERPOWERED_TOO_FEW_PARENT_RINGS')
    audit_n=min(audit_n,len(observed)-min_fit)
    audit=list(observed[-audit_n:])
    remaining=observed[:-audit_n]
    buffer_n=min(buffer_n,max(0,len(remaining)-min_fit))
    buffer=list(remaining[-buffer_n:]) if buffer_n else []
    fit=remaining[:-buffer_n] if buffer_n else remaining
    if len(fit)<min_fit or len(audit)<min_audit:
        return EvidencePartition(tuple(fit),tuple(buffer),tuple(audit),'UNDERPOWERED')
    return EvidencePartition(tuple(fit),tuple(buffer),tuple(audit),'OK')

def subset_prepared(prep: PreparedSample, ring_ids) -> PreparedSample:
    """Return a shallow immutable-data subset containing only selected rings."""
    wanted={str(x) for x in ring_ids}
    return PreparedSample(prep.sample,prep.center_px,prep.scale_px,
        {k:v for k,v in prep.points_by_ring.items() if k in wanted},
        {k:v for k,v in prep.tangents_by_ring.items() if k in wanted},
        [a for a in prep.arcs if a.ring_id in wanted],
        [k for k in prep.ring_order if k in wanted])

def fit_objective(h, prep, cfg, partition: EvidencePartition, model='m0'):
    """Evaluate fit evidence only; audit content cannot affect this value."""
    if not partition.fit_feasible:
        return None
    return float(objective(h,subset_prepared(prep,partition.fit_ring_ids),cfg,model))

def independent_audit(h, prep, cfg, partition: EvidencePartition, model='m0', alternatives=()):
    """Audit a locked hypothesis without refitting or switching candidates."""
    if not partition.audit_feasible:
        return {'outcome':'HOLDOUT_UNDERPOWERED','reason':partition.reason,
                'audit_ring_count':0,'score':None,'alternative_margin':None}
    audit_prep=subset_prepared(prep,partition.audit_ring_ids)
    score=float(objective(h,audit_prep,cfg,model))
    alt_scores=[float(objective(a,audit_prep,cfg,model)) for a in alternatives]
    margin=None if not alt_scores else float(min(alt_scores)-score)
    # Without a locked applicable alternative/sentinel, non-rejection cannot
    # establish power.  Keep the conservative three-way interface explicit;
    # final cutoffs and power calibration remain an E3 responsibility.
    if margin is None:
        outcome='HOLDOUT_UNDERPOWERED'
    else:
        outcome='AUDIT_SUPPORTED' if margin >= 0 else 'HOLDOUT_FAILED'
    return {'outcome':outcome,'reason':partition.reason,
            'audit_ring_count':len(partition.audit_ring_ids),'score':score,
            'alternative_margin':margin}
