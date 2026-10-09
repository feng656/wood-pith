#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl

def load(p):return json.loads(Path(p).read_text()) if Path(p).exists() else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');a=p.parse_args();cfg=load_config(a.config);base=resolve_path(cfg,cfg['paths']['results_dir']);m0=load(base/'step3_v2'/'m0_screen.json');m1=load(base/'step3_v2'/'m1rpc_screen.json');rep=load(base/'bias_audit'/'repair_decision.json');g2=load(base/'blind'/'g2.json');states=read_jsonl(base/'observability'/'states.jsonl') if (base/'observability'/'states.jsonl').exists() else [];md=['# OC-ArcPith-RG Adjusted Experiment Report','']
 for title,obj in [('M0 candidate gate',m0),('M1-RPC candidate gate',m1),('Bias repair',rep),('Blind G2',g2)]:md += [f'## {title}','```json',json.dumps(obj,indent=2,ensure_ascii=False) if obj else 'null','```','']
 md += ['## Observability states',json.dumps(dict(Counter(r['state'] for r in states)),ensure_ascii=False),'']
 (base/'FINAL_REPORT.md').write_text('\n'.join(md),encoding='utf-8');print(base/'FINAL_REPORT.md')
if __name__=='__main__':main()
