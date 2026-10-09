#!/usr/bin/env python3
from pathlib import Path
import argparse,json,math
import numpy as np

def ring(pith,r0,a2,a3,n=180,span=(-1.5,1.5)):
    th=np.linspace(span[0],span[1],n); r=r0*(1+a2*np.cos(2*th)+a3*np.sin(3*th)); return np.column_stack([pith[0]+r*np.cos(th),pith[1]+r*np.sin(th)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='data/demo/manifest.jsonl');a=p.parse_args();path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);rows=[]
 for i in range(4):
    size=512; pith=np.array([120+40*i,260-20*i],float); rs=[]
    for k,r0 in enumerate([80,110,140,170,200]):rs.append({'ring_id':str(k),'order':k,'points_px':ring(pith,r0,0.05,0.025).tolist()})
    rows.append({'sample_id':f'demo{i}','tree_id':f'T{i}','image_size':[size,size],'pith_px':pith.tolist(),'rings':rs})
 with path.open('w') as f:
    for r in rows:f.write(json.dumps(r)+'\n')
 print(path)
if __name__=='__main__':main()
