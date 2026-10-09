#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,multiprocessing as mp,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,sample_from_dict,write_jsonl
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.contribution import evaluate

_WORKER_CFG=None
_WORKER_MODEL=None

def _init_worker(cfg,model):globals()['_WORKER_CFG']=cfg;globals()['_WORKER_MODEL']=model
def _evaluate_record(d):
 prep=prepare(sample_from_dict(d),_WORKER_CFG)
 return {'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,'rows':[{'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,'model':_WORKER_MODEL,**r} for r in evaluate(prep,_WORKER_CFG,_WORKER_MODEL)]}
def _iter_jsonl(path):
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',default=None);p.add_argument('--model',choices=['m0','m1rpc'],default=None);p.add_argument('--max-samples',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--no-resume',action='store_true');a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest']);base=resolve_path(cfg,cfg['paths']['results_dir']);model=a.model
 if model is None:
  selected=base/'blind'/'model.json'
  if not selected.exists():raise RuntimeError('no blind model selection exists; pass --model explicitly')
  model=json.loads(selected.read_text())['selected_model']
 records=list(_iter_jsonl(src));records=records[:a.max_samples] if a.max_samples else records;out=base/'contribution';out.mkdir(parents=True,exist_ok=True);checkpoint=out/f'{model}_checkpoint.jsonl';done={}
 if checkpoint.exists() and not a.no_resume:
  for row in _iter_jsonl(checkpoint):done[row['sample_id']]=row
 pending=[d for d in records if d['sample_id'] not in done];start=time.perf_counter();mode='w' if a.no_resume else 'a'
 with checkpoint.open(mode,encoding='utf-8') as handle:
  if int(a.workers)>1:
   with mp.get_context('spawn').Pool(int(a.workers),initializer=_init_worker,initargs=(cfg,model)) as pool:
    for row in pool.imap_unordered(_evaluate_record,pending,chunksize=1):
     done[row['sample_id']]=row;handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();print(f"contribution {len(done)}/{len(records)}, elapsed {time.perf_counter()-start:.1f}s",flush=True)
  else:
   _init_worker(cfg,model)
   for d in pending:
    row=_evaluate_record(d);done[row['sample_id']]=row;handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();print(f"contribution {len(done)}/{len(records)}, elapsed {time.perf_counter()-start:.1f}s",flush=True)
 missing=[d['sample_id'] for d in records if d['sample_id'] not in done]
 if missing:raise RuntimeError(f'contribution incomplete: {len(missing)} samples missing')
 rows=[]
 for d in records:rows.extend(done[d['sample_id']]['rows'])
 write_jsonl(out/'ring_contribution.jsonl',rows)
if __name__=='__main__':main()
