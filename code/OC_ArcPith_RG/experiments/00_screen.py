#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,sample_from_dict,write_jsonl,dump_json
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.screen import screen,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',default=None);a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest']);out=resolve_path(cfg,cfg['paths']['results_dir'])/'step3_v2';out.mkdir(parents=True,exist_ok=True)
 rows0=[];rows1=[]
 for d in read_jsonl(src):
  prep=prepare(sample_from_dict(d),cfg);rows0.append(screen(prep,cfg,'m0'));rows1.append(screen(prep,cfg,'m1rpc'))
 write_jsonl(out/'candidate_m0.jsonl',rows0);write_jsonl(out/'candidate_m1rpc.jsonl',rows1);dump_json(out/'m0_screen.json',summarize(rows0,cfg));dump_json(out/'m1rpc_screen.json',summarize(rows1,cfg))
 print((out/'m0_screen.json').read_text())
if __name__=='__main__':main()
