from collections import defaultdict
import numpy as np
from .types import Arc,PreparedSample
from .geometry import frame,to_norm
from .ring import resample

def _ring_order(sample):
    full_order=(sample.metadata or {}).get('ring_order_audit',{}).get('full_parent_ring_order')
    if full_order:
        explicit_ids={r.ring_id for r in sample.rings}
        if not explicit_ids.issubset(set(map(str,full_order))):
            raise ValueError('observed parent ring is absent from audited full ring order')
        # Keep unobserved rings in the sequence so M1-RPC never treats rings
        # separated by an invisible parent ring as biologically adjacent.
        return [str(ring_id) for ring_id in full_order]
    by_id=defaultdict(list)
    for r in sample.rings: by_id[r.ring_id].append(r)
    explicit=[r for r in sample.rings if r.order is not None]
    if len(explicit)==len(sample.rings):
        order_by_id={}
        for rid,fragments in by_id.items():
            values={int(r.order) for r in fragments}
            if len(values)!=1: raise ValueError(f'conflicting explicit order for parent ring {rid}: {sorted(values)}')
            order_by_id[rid]=values.pop()
        values=list(order_by_id.values())
        if len(values)!=len(set(values)): raise ValueError('different parent rings share the same explicit order')
        return [rid for rid,_ in sorted(order_by_id.items(),key=lambda z:z[1])]
    def key(r):
        try:return (0,float(r.ring_id))
        except:return (1,r.ring_id)
    return [r.ring_id for r in sorted((fragments[0] for fragments in by_id.values()),key=key)]

def prepare(sample,cfg):
    c,s=frame(sample.image_size)
    if cfg.get('model',{}).get('require_explicit_ring_order',False) and any(r.order is None for r in sample.rings):
        raise ValueError(f'{sample.sample_id}: explicit parent-ring order is required for M1-RPC')
    pts_by=defaultdict(list); tan_by=defaultdict(list); raw=[]
    fragment_index=defaultdict(int)
    for r in sample.rings:
        if len(r.points_px)<cfg['preprocess']['min_points_per_ring']: continue
        pts,tan,ds=resample(r.points_px,cfg['preprocess']['resample_spacing_px'],cfg['preprocess']['spline_smoothing_px'])
        pts=to_norm(pts,c,s); ds=ds/s
        frag=fragment_index[r.ring_id]; fragment_index[r.ring_id]+=1
        pts_by[r.ring_id].append(pts); tan_by[r.ring_id].append(tan)
        m=int(cfg['preprocess']['micro_arc_points'])
        starts=list(range(0,len(pts),m))
        for j,a in enumerate(starts):
            b=min(len(pts),a+m)
            if b-a>=5: raw.append((r.ring_id,frag,j,pts[a:b],tan[a:b],ds[a:b]))
    by=defaultdict(list)
    for q in raw: by[q[0]].append(q)
    arcs=[]; ommax=float(cfg['preprocess']['ring_budget_max'])
    if not np.isfinite(ommax) or ommax <= 0:
        raise ValueError('preprocess.ring_budget_max must be finite and positive')
    for rid,items in by.items():
        # Equal-parent is an evidence-budget rule: every eligible biological
        # parent ring receives the same total budget.  The previous
        # length-saturating formula made Omega depend on visible arc length,
        # so short/occluded rings were silently down-weighted and the result
        # changed with coverage rather than information quality.  Arc-length
        # quadrature still distributes that fixed parent budget within the
        # ring; it must not determine the parent-ring total.
        mass=sum(float(np.sum(q[5])) for q in items)
        if not np.isfinite(mass) or mass <= 0:
            continue
        Omega=ommax
        for rid,frag,j,p,t,ds in items:
            om=Omega*float(np.sum(ds))/max(mass,1e-12)
            arcs.append(Arc(f'{rid}:f{frag}:m{j}',rid,p,t,ds,float(om)))
    order=_ring_order(sample)
    return PreparedSample(sample,c,float(s),dict(pts_by),dict(tan_by),arcs,order)
