from pathlib import Path
import json
import hashlib
import numpy as np
from .types import Sample,Ring

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def dump_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def _stable_hash(value):
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _attach_lineage_hashes(d):
    """Attach deterministic row-level hashes without trusting caller metadata.

    The protocol requires replayable input/observation/lineage identities.  A
    manifest may not contain them (older manifests do not), so compute them at
    the ingestion boundary while preserving any explicitly supplied values.
    Hashes deliberately exclude the hash fields themselves.
    """
    metadata=dict(d.get('metadata') or {})
    curves=d.get('rings',d.get('curves',[]))
    observation_payload={
        'curves':curves,
        'image_size':d.get('image_size'),
        'pith_px':d.get('pith_px'),
    }
    lineage_payload={
        'sample_id':d.get('sample_id'),
        'tree_id':d.get('tree_id'),
        'section_id':d.get('section_id'),
        'crop_id':d.get('crop_id'),
        'parent_ring_ids':[str(c.get('ring_id')) for c in curves],
        'fragment_ids':[str(c.get('fragment_id')) for c in curves],
    }
    input_payload={
        'image':d.get('image'),
        'image_size':d.get('image_size'),
        'observation_hash':_stable_hash(observation_payload),
        'lineage_hash':_stable_hash(lineage_payload),
    }
    expected={'observation_hash':_stable_hash(observation_payload),
              'lineage_hash':_stable_hash(lineage_payload),
              'input_hash':_stable_hash(input_payload)}
    for key,value in expected.items():
        supplied=metadata.get(key)
        if supplied is not None and supplied != value:
            raise ValueError(f'{key} does not match manifest content')
        metadata[key]=value
    return metadata

def sample_from_dict(d):
    rings=[]
    src=d.get('rings',d.get('curves',[]))
    for i,r in enumerate(src):
        pts=r.get('points_px',r.get('points'))
        rings.append(Ring(str(r.get('ring_id',i)),np.asarray(pts,float),r.get('order')))
    p=d.get('pith_px')
    return Sample(str(d['sample_id']),str(d['tree_id']),tuple(d['image_size']),rings,
                  None if p is None else np.asarray(p,float),d.get('mm_per_pixel'),_attach_lineage_hashes(d))
