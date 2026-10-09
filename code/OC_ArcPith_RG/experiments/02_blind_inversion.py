#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,sample_from_dict,write_jsonl,dump_json
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.cv import run_group_cv

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',default=None);a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest']);base=resolve_path(cfg,cfg['paths']['results_dir']);preps=[prepare(sample_from_dict(d),cfg) for d in read_jsonl(src)];m0,m1,tune,dec=run_group_cv(preps,cfg);out=base/'blind';write_jsonl(out/'metrics_m0.jsonl',m0);write_jsonl(out/'metrics_m1rpc_heldout.jsonl',m1)
 for lam,rows in tune.items():write_jsonl(out/f'tuning_lambda_{lam:g}.jsonl',rows)
 dump_json(out/'g2.json',dec);dump_json(out/'model.json',{'selected_model':'m1rpc' if dec['pass'] else 'm0'});print((out/'g2.json').read_text())
if __name__=='__main__':main()
