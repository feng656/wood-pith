from __future__ import annotations
import math,hashlib
import numpy as np
from .geometry import to_norm,h_from_point,h_from_phi_kappa

def _rng(seed,sample_id):
    h=int(hashlib.sha256(sample_id.encode()).hexdigest()[:8],16)
    return np.random.default_rng(int(seed)^h)

def build(prep,cfg):
    if prep.sample.pith_px is None: return []
    gt=to_norm(prep.sample.pith_px,prep.center_px,prep.scale_px); R=float(np.linalg.norm(gt)); phi=math.atan2(gt[1],gt[0]) if R>1e-12 else 0.0
    out=[{'name':'GT','kind':'truth','h':h_from_point(gt)}]
    # normalized, scale-invariant translations
    for f in cfg['candidate_screen']['translation_fractions']:
        d=float(f)
        for tag,v in [('+x',[d,0]),('-x',[-d,0]),('+y',[0,d]),('-y',[0,-d])]:
            out.append({'name':f'trans{f:g}{tag}','kind':'translation','h':h_from_point(gt+np.asarray(v)),'offset_norm':d})
    if R>1e-8:
        for deg in cfg['candidate_screen']['direction_deg']:
            for sg in (-1,1):
                ph=phi+math.radians(sg*float(deg)); out.append({'name':f'rot{sg*float(deg):g}','kind':'direction','h':h_from_point(R*np.array([math.cos(ph),math.sin(ph)])),'angle_deg':sg*float(deg)})
        for sc in cfg['candidate_screen']['distance_scales']:
            out.append({'name':f'distx{sc:g}','kind':'distance','h':h_from_point(float(sc)*gt),'scale':float(sc)})
        out.append({'name':'infinity','kind':'distance','h':h_from_phi_kappa(phi,0.0)})
        out.append({'name':'reverse','kind':'reverse','h':h_from_point(-gt)})
    rng=_rng(cfg['seed'],prep.sample.sample_id)
    for i in range(int(cfg['candidate_screen']['random_candidates'])):
        ph=rng.uniform(-math.pi,math.pi); kap=10**rng.uniform(-1.5,0.8)
        out.append({'name':f'random{i}','kind':'random','h':h_from_phi_kappa(ph,kap)})
    return out
