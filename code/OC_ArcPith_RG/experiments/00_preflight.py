#!/usr/bin/env python3
"""Run candidate-free P0--P3 feasibility checks before candidate search."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,write_jsonl,sample_from_dict
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.preflight import preflight

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'))
    p.add_argument('--manifest',default=None);p.add_argument('--level',choices=['P0','P1','P2','P3'],default='P0')
    a=p.parse_args(); cfg=load_config(a.config)
    src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest'])
    out=resolve_path(cfg,cfg['paths']['results_dir'])/'preflight'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in read_jsonl(src):
        prep=prepare(sample_from_dict(d),cfg); result=preflight(prep,cfg,a.level)
        rows.append({'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,
                     'input_hash':prep.sample.metadata.get('input_hash'),
                     'observation_hash':prep.sample.metadata.get('observation_hash'),
                     'lineage_hash':prep.sample.metadata.get('lineage_hash'),**result})
    write_jsonl(out/f'{a.level.lower()}.jsonl',rows)
    counts={}
    for r in rows: counts[r['route']]=counts.get(r['route'],0)+1
    summary={'level':a.level,'records':len(rows),'route_counts':counts,
             'audit_feasible':sum(bool(r['features']['audit_feasible']) for r in rows)}
    (out/f'{a.level.lower()}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
