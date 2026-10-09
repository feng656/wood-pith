#!/usr/bin/env python3
from pathlib import Path
import argparse,json,re
from collections import Counter,defaultdict
import numpy as np

def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def summary(rows):
    bytree=defaultdict(list); hard=Counter(); typ=defaultdict(list); sizes=defaultdict(list)
    for r in rows:
        bytree[r['tree_id']].append(r)
        c=min((x for x in r['candidates'] if x['kind']!='truth'),key=lambda x:x['loss']);hard[c['kind']]+=1
        for k,v in r['candidate_types'].items():typ[k].append(v['min_gap'])
        m=re.search(r'_s(\d+)_',r['sample_id']) or re.search(r'_(\d+)px_',r['sample_id'])
        if m:sizes[int(m.group(1))].append(r)
    tg=[np.median([x['hard_gap'] for x in rr]) for rr in bytree.values()]
    tv=[np.median([x['truth_beats_fraction'] for x in rr]) for rr in bytree.values()]
    return {'samples':len(rows),'trees':len(bytree),'tree_median_truth_beats':float(np.median(tv)),'tree_positive_gap_fraction':float(np.mean(np.asarray(tg)>0)),'tree_median_hard_gap':float(np.median(tg)),'hardest_kind_counts':dict(hard),'type_min_gap_median':{k:float(np.median(v)) for k,v in typ.items()},'size_metrics':{str(s):{'n':len(rr),'median_hard_gap':float(np.median([x['hard_gap'] for x in rr])),'positive_gap_fraction':float(np.mean([x['hard_gap']>0 for x in rr]))} for s,rr in sorted(sizes.items())}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--strategy-a',required=True);p.add_argument('--strategy-b',required=True);p.add_argument('--strategy-b-m1');p.add_argument('--out',default='compare_step3.json');a=p.parse_args();out={'strategy_a_m0':summary(read(a.strategy_a)),'strategy_b_m0':summary(read(a.strategy_b))}
 if a.strategy_b_m1:out['strategy_b_m1']=summary(read(a.strategy_b_m1))
 Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
