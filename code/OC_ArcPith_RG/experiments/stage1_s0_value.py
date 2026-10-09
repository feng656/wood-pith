#!/usr/bin/env python3
"""Paired B0/B1/B2/B3 S0 value check on frozen crops.

The default limit is intentionally small: this is a deterministic rapid
screen, not a tree-level accuracy estimate or E3 calibration.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,write_jsonl,sample_from_dict
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.optimizer import solve
from oc_arcpith_rg.baselines import compare_baselines,_grid_candidates
from oc_arcpith_rg.geometry import projective_angle,h_from_point

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'))
    p.add_argument('--manifest',default=None);p.add_argument('--limit',type=int,default=24)
    p.add_argument('--b3-phi-bins',type=int,default=72);p.add_argument('--b3-kappa-bins',type=int,default=40)
    a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest'])
    out=resolve_path(cfg,cfg['paths']['results_dir'])/'stage1_s0_value';out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,d in enumerate(read_jsonl(src)):
        if i>=a.limit: break
        prep=prepare(sample_from_dict(d),cfg)
        # B1 and B2 share the same rapid grid budget. B3 is intentionally a
        # separate high-budget reference used only for search adequacy.
        _,h2=_grid_candidates(prep,cfg,'m0',72,40,float(cfg.get('profile',{}).get('kappa_max',50.0)))
        result=compare_baselines(prep,cfg,h_b2=h2,b1_phi_bins=72,b1_kappa_bins=40,
                                 b3_phi_bins=a.b3_phi_bins,b3_kappa_bins=a.b3_kappa_bins)
        gt=None if prep.sample.pith_px is None else h_from_point((prep.sample.pith_px-prep.center_px)/prep.scale_px)
        row={'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,
             'input_hash':prep.sample.metadata.get('input_hash'),'models':{}}
        for name,val in result.items():
            row['models'][name]={'loss':val['loss'],'d_to_b2':val['d_to_b2'],
                                 'd_to_gt':None if gt is None or val['h'] is None else projective_angle(val['h'],gt),
                                 'point':None if val['point'] is None else np.asarray(val['point']).tolist()}
        rows.append(row)
    write_jsonl(out/'paired.jsonl',rows)
    def median(values):
        return None if not values else float(np.median(values))
    tree_metrics={}
    for tree in sorted({r['tree_id'] for r in rows}):
        rr=[r for r in rows if r['tree_id']==tree]
        d20=[r['models']['B2']['d_to_gt']-r['models']['B0']['d_to_gt'] for r in rr if r['models']['B2']['d_to_gt'] is not None and r['models']['B0']['d_to_gt'] is not None]
        d21=[r['models']['B2']['d_to_gt']-r['models']['B1']['d_to_gt'] for r in rr if r['models']['B2']['d_to_gt'] is not None and r['models']['B1']['d_to_gt'] is not None]
        d23=[r['models']['B2']['d_to_gt']-r['models']['B3']['d_to_gt'] for r in rr if r['models']['B2']['d_to_gt'] is not None and r['models']['B3']['d_to_gt'] is not None]
        lg=[r['models']['B2']['loss']-r['models']['B3']['loss'] for r in rr]
        tree_metrics[tree]={'records':len(rr),'b2_vs_b0_distance_median':median(d20),'b2_vs_b0_wins':sum(x<0 for x in d20),
                            'b2_vs_b1_distance_median':median(d21),'b2_vs_b1_wins':sum(x<0 for x in d21),
                            'b2_vs_b3_distance_median':median(d23),'b2_vs_b3_wins':sum(x<0 for x in d23),
                            'b2_minus_b3_loss_median':median(lg)}
    summary={'records':len(rows),'trees':len({r['tree_id'] for r in rows}),'tree_metrics':tree_metrics,
             'b2_loss_minus_b0_loss':float(np.median([r['models']['B2']['loss']-r['models']['B0']['loss'] for r in rows])) if rows else None,
             'b2_gt_distance_minus_b0_gt_distance':float(np.median([r['models']['B2']['d_to_gt']-r['models']['B0']['d_to_gt'] for r in rows if r['models']['B2']['d_to_gt'] is not None and r['models']['B0']['d_to_gt'] is not None])) if rows else None,
             'search_budget':{'B1_phi_bins':72,'B1_kappa_bins':40,'B2_phi_bins':72,'B2_kappa_bins':40,
                              'B3_phi_bins':a.b3_phi_bins,'B3_kappa_bins':a.b3_kappa_bins},
             'note':'B1/B2 share the rapid grid budget; B3 is a separate high-budget reference. Four trees remain exploratory and do not support E3 calibration.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
