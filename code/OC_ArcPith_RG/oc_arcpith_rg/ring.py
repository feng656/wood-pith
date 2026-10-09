import numpy as np
from scipy.interpolate import splprep,splev

def _dedupe(p):
    p=np.asarray(p,float)
    if len(p)<2:return p
    keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]
    return p[keep]

def resample(points,spacing=4.0,smoothing=0.8):
    p=_dedupe(points)
    if len(p)<4: raise ValueError('ring needs >=4 distinct points')
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1); u=np.r_[0,np.cumsum(seg)]
    if u[-1]<=0: raise ValueError('degenerate ring')
    total=u[-1]; u=u/total
    tck,_=splprep([p[:,0],p[:,1]],u=u,s=smoothing*smoothing*len(p),k=min(3,len(p)-1))
    n=max(8,int(np.ceil(total/max(spacing,1e-6)))+1)
    uu=np.linspace(0,1,n)
    pts=np.column_stack(splev(uu,tck)); der=np.column_stack(splev(uu,tck,der=1))
    tan=der/(np.linalg.norm(der,axis=1,keepdims=True)+1e-12)
    d=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    ds=np.empty(n); ds[0]=d[0]/2; ds[-1]=d[-1]/2; ds[1:-1]=(d[:-1]+d[1:])/2
    return pts,tan,ds
