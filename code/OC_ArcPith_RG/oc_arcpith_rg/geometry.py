from __future__ import annotations
import math
import numpy as np

def frame(image_size):
    w,h=image_size
    c=np.array([(w-1)/2.0,(h-1)/2.0],float)
    return c,max(w,h)/2.0

def to_norm(x,c,s): return (np.asarray(x,float)-c)/float(s)
def from_norm(x,c,s): return np.asarray(x,float)*float(s)+c

def canonical_h(h):
    h=np.asarray(h,float); n=np.linalg.norm(h)
    if n==0: raise ValueError('zero h')
    h=h/n
    if h[2]<0: h=-h
    return h

def h_from_point(p):
    p=np.asarray(p,float)
    return canonical_h(np.array([p[0],p[1],1.0]))

def h_from_phi_kappa(phi,kappa):
    # p = u/kappa; kappa=0 is point at infinity
    return canonical_h(np.array([math.cos(phi),math.sin(phi),max(float(kappa),0.0)]))

def point_from_h(h,axis_eps=1e-8):
    h=canonical_h(h)
    return None if h[2]<=axis_eps else h[:2]/h[2]

def phi_kappa_from_h(h):
    h=canonical_h(h); q=np.linalg.norm(h[:2])
    phi=math.atan2(h[1],h[0])
    return phi, (h[2]/q if q>1e-15 else float('inf'))

def projective_angle(h1,h2):
    a=canonical_h(h1); b=canonical_h(h2)
    return float(math.acos(np.clip(abs(float(a@b)),-1,1)))

def absolute_residual(h,x,t,eps=1e-9):
    h=canonical_h(h)
    v=h[:2][None,:]-h[2]*x
    return np.sum(t*v,axis=1)/np.sqrt(np.sum(v*v,axis=1)+eps*eps)

def radial_tangential_error(pred,gt):
    pred=np.asarray(pred,float); gt=np.asarray(gt,float)
    r=np.linalg.norm(gt)
    if r<1e-12: return float(np.linalg.norm(pred-gt)),0.0
    u=gt/r; up=np.array([-u[1],u[0]])
    e=pred-gt
    return float(e@u),float(e@up)
