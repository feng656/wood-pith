#!/usr/bin/env python3
"""Select a deterministic, tree-balanced exploratory Stage-1 sentinel."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

def stratum(row):
    meta=row.get('metadata') or {}
    size=int((row.get('image_size') or [0])[0])
    inside=bool(meta.get('pith_in_patch'))
    dist=float(meta.get('distance_to_pith_px', float('inf')))
    if inside: location='inside'
    elif dist < size: location='near_outside'
    elif dist < 2*size: location='mid_outside'
    else: location='far_outside'
    return (size,location)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--per-tree',type=int,default=8);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    by=defaultdict(list)
    for row in rows: by[str(row['tree_id'])].append(row)
    selected=[]
    for tree in sorted(by):
        strata=defaultdict(list)
        for row in sorted(by[tree],key=lambda r:str(r['sample_id'])): strata[stratum(row)].append(row)
        keys=sorted(strata)
        # Round-robin strata so each biological tree contributes varied
        # geometry/location cases without using any algorithm output.
        cursors={k:0 for k in keys}
        for j in range(min(a.per_tree,len(by[tree]))):
            k=keys[j % len(keys)]; selected.append(strata[k][cursors[k] % len(strata[k])]); cursors[k]+=1
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    print(json.dumps({'input_records':len(rows),'selected_records':len(selected),
                      'trees':sorted({r['tree_id'] for r in selected}),
                      'strata':{str(k):sum(stratum(r)==k for r in selected) for k in sorted({stratum(r) for r in selected})}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
