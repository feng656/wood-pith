#!/usr/bin/env python3
"""Check density/duplication/interpolation/fragment invariance of M0."""
from pathlib import Path
import argparse,json
import numpy as np,yaml
import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oc_arcpith_rg.io import read_jsonl,sample_from_dict
from oc_arcpith_rg.config import load_config
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.optimizer import solve
from oc_arcpith_rg.geometry import projective_angle

def variant(d,kind):
    x=json.loads(json.dumps(d)); curves=x.get('curves',x.get('rings',[]))
    if kind=='duplicate':
        for c in curves:c['points_px']=[p for p in c['points_px'] for _ in (0,1)]
    elif kind=='interpolate':
        for c in curves:
            pts=np.asarray(c['points_px'],float); q=(pts[:-1]+pts[1:])/2
            out=[]
            for i,p in enumerate(pts[:-1]): out.extend([p,q[i]])
            out.append(pts[-1]); c['points_px']=np.asarray(out).tolist()
    elif kind=='fragment_split':
        out=[]
        for c in curves:
            pts=c['points_px']; mid=max(4,len(pts)//2); a=dict(c);b=dict(c)
            a['fragment_id']=str(c.get('fragment_id',c['ring_id'])+':a');b['fragment_id']=str(c.get('fragment_id',c['ring_id'])+':b')
            a['points_px']=pts[:mid];b['points_px']=pts[mid-1:];out.extend([a,b])
        x['curves']=out
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--manifest',required=True);p.add_argument('--limit',type=int,default=8);a=p.parse_args()
    cfg=load_config(a.config); base=read_jsonl(a.manifest); rows=[]
    for d in base[:a.limit]:
        ref=prepare(sample_from_dict(d),cfg); h=solve(ref,cfg,'m0')['best']['h']; row={'sample_id':d['sample_id']}
        for kind in ['duplicate','interpolate','fragment_split']:
            q=prepare(sample_from_dict(variant(d,kind)),cfg); hq=solve(q,cfg,'m0')['best']['h']; row[kind]={'d_rp2':projective_angle(h,hq),'ring_weights':{}}
            for arc in q.arcs:row[kind]['ring_weights'][arc.ring_id]=row[kind]['ring_weights'].get(arc.ring_id,0)+arc.omega
        rows.append(row)
    out=Path('results_urudendro_no_background/stage1_s0_value/equal_parent_invariance.json');out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'records':len(rows),'max_d_rp2':max((v[k]['d_rp2'] for v in rows for k in ['duplicate','interpolate','fragment_split']),default=None)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
