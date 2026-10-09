#!/usr/bin/env python3
from pathlib import Path
import argparse,json,multiprocessing as mp,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,sample_from_dict,write_jsonl,dump_json
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.bias import audit,compare_repairs

_WORKER_CFG=None

def _init_worker(cfg):
 _WORKER_CFG=cfg
 globals()['_WORKER_CFG']=cfg

def _audit_record(d):
 prep=prepare(sample_from_dict(d),_WORKER_CFG)
 return {'sample_id':prep.sample.sample_id,'m0':audit(prep,_WORKER_CFG,'m0'),'m1rpc':audit(prep,_WORKER_CFG,'m1rpc')}

def _iter_jsonl(path):
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',default=None);p.add_argument('--workers',type=int,default=1);p.add_argument('--progress-every',type=int,default=50);p.add_argument('--no-resume',action='store_true');a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest']);out=resolve_path(cfg,cfg['paths']['results_dir'])/'bias_audit';out.mkdir(parents=True,exist_ok=True)
 checkpoint=out/'bias_checkpoint.jsonl';done={}
 if checkpoint.exists() and not a.no_resume:
  for row in _iter_jsonl(checkpoint):done[row['sample_id']]=row
 total=sum(1 for _ in _iter_jsonl(src));pending=(d for d in _iter_jsonl(src) if d['sample_id'] not in done);start=time.perf_counter();mode='w' if a.no_resume else 'a'
 with checkpoint.open(mode,encoding='utf-8') as handle:
  if int(a.workers)>1:
   with mp.get_context('spawn').Pool(int(a.workers),initializer=_init_worker,initargs=(cfg,)) as pool:
    iterator=pool.imap(_audit_record,pending,chunksize=1)
    for row in iterator:
     done[row['sample_id']]=row;handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
     if len(done)%int(a.progress_every)==0:print(f'bias audit {len(done)}/{total}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
  else:
   _init_worker(cfg)
   for d in pending:
    row=_audit_record(d);done[row['sample_id']]=row;handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
    if len(done)%int(a.progress_every)==0:print(f'bias audit {len(done)}/{total}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
 order=[d['sample_id'] for d in _iter_jsonl(src)];missing=[sid for sid in order if sid not in done]
 if missing:raise RuntimeError(f'bias audit incomplete: {len(missing)} samples missing')
 A=[done[sid]['m0'] for sid in order];B=[done[sid]['m1rpc'] for sid in order]
 write_jsonl(out/'m0_bias.jsonl',A);write_jsonl(out/'m1rpc_bias.jsonl',B);dump_json(out/'repair_decision.json',compare_repairs(A,B,cfg));print((out/'repair_decision.json').read_text())
if __name__=='__main__':main()
