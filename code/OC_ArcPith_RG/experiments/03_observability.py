#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,multiprocessing as mp,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oc_arcpith_rg.config import load_config,resolve_path
from oc_arcpith_rg.io import read_jsonl,sample_from_dict,write_jsonl
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.observability import profile,classify
from oc_arcpith_rg.geometry import to_norm,h_from_point
from oc_arcpith_rg.objectives import objective

_WORKER_CFG=None
_WORKER_MODEL=None

def _init_worker(cfg,model):
 globals()['_WORKER_CFG']=cfg;globals()['_WORKER_MODEL']=model

def _iter_jsonl(path):
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)

def _tangent_diversity(prep,bins=18):
 angles=[];weights=[]
 for arc in prep.arcs:
  angle=np.mod(np.arctan2(arc.tangents[:,1],arc.tangents[:,0]),np.pi)
  weight=arc.omega*arc.ds/(np.sum(arc.ds)+1e-12);angles.extend(angle.tolist());weights.extend(weight.tolist())
 hist,_=np.histogram(angles,bins=bins,range=(0,np.pi),weights=weights)
 prob=hist/(np.sum(hist)+1e-12);nz=prob[prob>0]
 return float(-np.sum(nz*np.log(nz))/np.log(bins)),float(np.mean(hist>0))

def _observe(d):
 prep=prepare(sample_from_dict(d),_WORKER_CFG);pr=profile(prep,_WORKER_CFG,_WORKER_MODEL);entropy,occupied=_tangent_diversity(prep)
 gt=to_norm(prep.sample.pith_px,prep.center_px,prep.scale_px) if prep.sample.pith_px is not None else None
 gt_loss=None if gt is None else float(objective(h_from_point(gt),prep,_WORKER_CFG,_WORKER_MODEL))
 return {'sample_id':prep.sample.sample_id,'tree_id':prep.sample.tree_id,'section_id':d.get('section_id'),'model':_WORKER_MODEL,
         'state':classify(pr,_WORKER_CFG),'includes_axis':pr['includes_axis'],'phi_width_deg':pr['phi_width_deg'],'min_loss':pr['min_loss'],
         'gt_loss':gt_loss,'gt_loss_gap':None if gt_loss is None else float(gt_loss-pr['min_loss']),
         'gt_within_profile_support':None if gt_loss is None else bool(gt_loss<=pr['min_loss']+float(_WORKER_CFG['profile']['support_delta'])),
         'support_component_count':pr['support_component_count'],'support_component_sizes':pr['support_component_sizes'],'support_cells':pr['support_cells'],
         'support_kappa_min':pr['support_kappa_min'],'support_kappa_max':pr['support_kappa_max'],'crop_size_px':int(d['image_size'][0]),
         'true_pith_distance_norm':None if gt is None else float(np.linalg.norm(gt)),'pith_in_patch':bool(d.get('metadata',{}).get('pith_in_patch')),
         'visible_parent_ring_count':len(prep.points_by_ring),'visible_fragment_count':sum(len(v) for v in prep.points_by_ring.values()),
         'tangent_axis_entropy':entropy,'tangent_axis_occupied_fraction':occupied}

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',default=None);p.add_argument('--model',choices=['m0','m1rpc'],default=None);p.add_argument('--workers',type=int,default=1);p.add_argument('--progress-every',type=int,default=50);p.add_argument('--no-resume',action='store_true');a=p.parse_args();cfg=load_config(a.config);src=Path(a.manifest) if a.manifest else resolve_path(cfg,cfg['paths']['manifest']);base=resolve_path(cfg,cfg['paths']['results_dir']);model=a.model
 if model is None:
  selected=base/'blind'/'model.json'
  if not selected.exists():raise RuntimeError('no blind model selection exists; pass --model explicitly')
  model=json.loads(selected.read_text())['selected_model']
 out=base/'observability';out.mkdir(parents=True,exist_ok=True);checkpoint=out/f'{model}_checkpoint.jsonl';done={}
 if checkpoint.exists() and not a.no_resume:
  for row in _iter_jsonl(checkpoint):done[row['sample_id']]=row
 total=sum(1 for _ in _iter_jsonl(src));pending=(d for d in _iter_jsonl(src) if d['sample_id'] not in done);start=time.perf_counter();mode='w' if a.no_resume else 'a'
 with checkpoint.open(mode,encoding='utf-8') as handle:
  if int(a.workers)>1:
   with mp.get_context('spawn').Pool(int(a.workers),initializer=_init_worker,initargs=(cfg,model)) as pool:
    iterator=pool.imap(_observe,pending,chunksize=1)
    for row in iterator:
     done[row['sample_id']]=row;handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
     if len(done)%int(a.progress_every)==0:print(f'observability {len(done)}/{total}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
  else:
   _init_worker(cfg,model)
   for d in pending:
    row=_observe(d);done[row['sample_id']]=row;handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
    if len(done)%int(a.progress_every)==0:print(f'observability {len(done)}/{total}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
 order=[d['sample_id'] for d in _iter_jsonl(src)];missing=[sid for sid in order if sid not in done]
 if missing:raise RuntimeError(f'observability incomplete: {len(missing)} samples missing')
 write_jsonl(out/'states.jsonl',[done[sid] for sid in order])
if __name__=='__main__':main()
