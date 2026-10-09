#!/usr/bin/env python3
"""Small structural Independent Audit run on locked E_fit hypotheses.

This verifies split/firewall ordering; it is not a calibrated power estimate.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,write_jsonl,sample_from_dict
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.evidence import split_parent_rings,subset_prepared,fit_objective,independent_audit
from oc_arcpith_rg.optimizer import solve

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'))
    p.add_argument('--manifest',default=None);p.add_argument('--limit',type=int,default=64)
    a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest'])
    out=resolve_path(cfg,cfg['paths']['results_dir'])/'independent_audit';out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,d in enumerate(read_jsonl(src)):
        if i>=a.limit: break
        prep=prepare(sample_from_dict(d),cfg);part=split_parent_rings(prep,cfg)
        if not part.fit_feasible:
            rows.append({'sample_id':prep.sample.sample_id,'outcome':'HOLDOUT_UNDERPOWERED','reason':part.reason});continue
        fit=solve(subset_prepared(prep,part.fit_ring_ids),cfg,'m0')
        if not fit['best']:
            rows.append({'sample_id':prep.sample.sample_id,'outcome':'HOLDOUT_UNDERPOWERED','reason':'FIT_SEARCH_FAILED'});continue
        h=fit['best']['h'];before=fit_objective(h,prep,cfg,part)
        # No arbitrary alternative is used here: without a preregistered
        # applicable sentinel this is intentionally UNDERPOWERED.  The run
        # verifies split/firewall ordering only, not audit power.
        audit=independent_audit(h,prep,cfg,part,'m0',[]);after=fit_objective(h,prep,cfg,part)
        rows.append({'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,
                     'input_hash':prep.sample.metadata.get('input_hash'),
                     'fit_ring_ids':part.fit_ring_ids,'buffer_ring_ids':part.buffer_ring_ids,
                     'audit_ring_ids':part.audit_ring_ids,'fit_score_before':before,
                     'fit_score_after':after,'firewall_ok':before==after,**audit})
    write_jsonl(out/'audit_records.jsonl',rows)
    summary={'records':len(rows),'firewall_failures':sum(not r.get('firewall_ok',True) for r in rows),
             'outcomes':{k:sum(r.get('outcome')==k for r in rows) for k in ['AUDIT_SUPPORTED','HOLDOUT_FAILED','HOLDOUT_UNDERPOWERED']}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
