#!/usr/bin/env python3
"""Stage 2A/2B: candidate-free preflight and six-source seed registry.

Seeds are deliberately not selected with GT and are not final coordinates.
All families are represented in one RP2-compatible homogeneous convention so
Stage 3 can rescore them with a single All-Arc objective.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def canon(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    h = h / max(float(np.linalg.norm(h)), 1e-300)
    if h[2] < 0:
        h = -h
    return h


def finite_h(p: np.ndarray) -> np.ndarray:
    return canon(np.r_[np.asarray(p, dtype=float), 1.0])


def far_h(phi: float, kappa: float) -> np.ndarray:
    return canon(np.array([math.cos(phi), math.sin(phi), max(0.0, float(kappa))]))


def rp_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.acos(np.clip(abs(float(canon(a) @ canon(b))), -1.0, 1.0)))


def ring_groups(arcs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arc in arcs:
        out[str(arc["ring_id"])].append(arc)
    return dict(out)


def weighted_rows(arcs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    groups = ring_groups(arcs)
    R = max(len(groups), 1)
    rows, weights = [], []
    for aa in groups.values():
        for arc in aa:
            pts = np.asarray(arc["points_norm"], dtype=float)
            tan = np.asarray(arc["tangents"], dtype=float)
            ds = np.asarray(arc["ds_norm"], dtype=float)
            q = ds / max(float(np.sum(ds)), 1e-12)
            for x, t, ww in zip(pts, tan, q):
                rows.append([float(t[0]), float(t[1]), -float(t @ x)])
                weights.append(float(ww) / R)
    return np.asarray(rows, dtype=float), np.asarray(weights, dtype=float)


def cs1_svd(arcs: list[dict[str, Any]]) -> tuple[np.ndarray | None, dict[str, Any]]:
    A, w = weighted_rows(arcs)
    if len(A) < 3:
        return None, {"rank": 0, "g_svd": None}
    Aw = A * np.sqrt(np.maximum(w, 1e-15))[:, None]
    _, s, vh = np.linalg.svd(Aw, full_matrices=False)
    eig = np.sort(s * s)
    gap = float((eig[1] - eig[0]) / max(eig[-1], 1e-15)) if len(eig) >= 3 else None
    return canon(vh[-1]), {"rank": int(np.linalg.matrix_rank(Aw)), "singular_values": s.tolist(), "g_svd": gap}


def group_quadratic(aa: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, float]:
    H = np.zeros((2, 2), dtype=float); b = np.zeros(2, dtype=float); length = 0.0
    for arc in aa:
        x = np.asarray(arc["points_norm"], dtype=float)
        t = np.asarray(arc["tangents"], dtype=float)
        ds = np.asarray(arc["ds_norm"], dtype=float)
        length += float(np.sum(ds))
        H += np.einsum("n,ni,nj->ij", ds, t, t)
        dot_tx = np.sum(t * x, axis=1)
        b += np.sum(ds[:, None] * t * dot_tx[:, None], axis=0)
    return H, b, length


def cs2_cs3(arcs: list[dict[str, Any]], max_combos: int = 128) -> tuple[list[dict[str, Any]], int]:
    groups = ring_groups(arcs); ids = sorted(groups)
    stats = {rid: group_quadratic(groups[rid]) for rid in ids}
    seeds: list[dict[str, Any]] = []
    combos = list(itertools.combinations(ids, 2)) + list(itertools.combinations(ids, 3))
    if len(combos) > max_combos:
        # Deterministic progressive sampling: extreme and regularly spaced ids.
        step = max(1, len(combos) // max_combos)
        combos = combos[::step][:max_combos]
    for combo in combos:
        H = np.zeros((2, 2)); b = np.zeros(2); total = 0.0
        for rid in combo:
            hi, bi, li = stats[rid]; wt = max(li, 1e-12); H += wt * hi; b += wt * bi; total += wt
        H /= max(total, 1e-12); b /= max(total, 1e-12)
        cond = float(np.linalg.cond(H)) if np.all(np.isfinite(H)) else float("inf")
        eig = np.linalg.eigvalsh(H)
        if eig[0] > 1e-7 and cond < 1e7:
            p = np.linalg.solve(H, b)
            if np.all(np.isfinite(p)):
                seeds.append({"h": finite_h(p).tolist(), "source": "CS2", "rings": list(combo), "condition": cond})
        else:
            # Weak intersections are retained as far-axis evidence by CS5.
            continue
    # CS3 uses the same analytic candidates but records parent-vote consensus.
    for row in seeds[:max_combos]:
        h = np.asarray(row["h"]); p = h[:2] / max(h[2], 1e-12)
        votes = 0
        for rid, aa in groups.items():
            vals=[]
            for arc in aa:
                x=np.asarray(arc["points_norm"]); t=np.asarray(arc["tangents"])
                vals.extend(np.abs(np.sum(t * (p[None,:]-x), axis=1)).tolist())
            if vals and float(np.median(vals)) <= 0.03: votes += 1
        seeds.append({"h": h.tolist(), "source": "CS3", "rings": row.get("rings", []),
                      "parent_vote_fraction": float(votes / max(len(groups), 1))})
    return seeds, len(combos)


def cs4_circles(arcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    # CS4 is a seed family, not an evidence reweighting step.  A deterministic
    # top-k gate keeps batch runtime bounded while All-Arc later sees every arc.
    ranked = sorted(arcs, key=lambda a: (float(a.get("theta_total_rad", 0.0)),
                                         float(a.get("length_norm", 0.0))), reverse=True)[:8]
    for arc in ranked:
        if float(arc.get("theta_total_rad", 0.0)) < math.radians(15.0):
            continue
        x=np.asarray(arc["points_norm"], dtype=float)
        if len(x)<6: continue
        M=np.column_stack([2*x[:,0],2*x[:,1],np.ones(len(x))]); y=np.sum(x*x,axis=1)
        try:
            sol,_,rank,_=np.linalg.lstsq(M,y,rcond=None)
            if rank<3 or not np.all(np.isfinite(sol)): continue
            p=sol[:2]; r=float(np.sqrt(max(sol[2]+p@p,0.0)))
            if r>0 and np.isfinite(r): out.append({"h":finite_h(p).tolist(),"source":"CS4","fragment_id":arc["fragment_id"],"radius_norm":r})
        except np.linalg.LinAlgError: continue
    return out


def cs5_far_axis(arcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    T=np.vstack([np.asarray(a["tangents"],float) for a in arcs]) if arcs else np.empty((0,2))
    if len(T)<2: return []
    C=T.T@T/max(len(T),1); vals,V=np.linalg.eigh(C); u=V[:,0]
    phi=math.atan2(float(u[1]),float(u[0])); out=[]
    for sign in (1,-1):
        ph=phi if sign==1 else phi+math.pi
        for radius in [1,2,4,8,16,32,64,128]:
            out.append({"h":far_h(ph,1.0/radius).tolist(),"source":"CS5","radius_over_fov":radius,"axis_sign":sign})
        out.append({"h":far_h(ph,0.0).tolist(),"source":"CS5","radius_over_fov":"inf","axis_sign":sign})
    return out


def cs6_mesh() -> list[dict[str, Any]]:
    out=[]
    # Cartesian finite chart: symmetric and bounded around the FOV center.
    for x in np.linspace(-4.0,4.0,9):
        for y in np.linspace(-4.0,4.0,9):
            out.append({"h":finite_h([x,y]).tolist(),"source":"CS6_FINITE_MESH"})
    for phi in np.linspace(-math.pi,math.pi,16,endpoint=False):
        for k in [0.0,0.01,0.03,0.1,0.3,1.0,3.0]:
            out.append({"h":far_h(float(phi),k).tolist(),"source":"CS6_FAR_MESH"})
    return out


def dedupe(rows: list[dict[str, Any]], tau_h: float = 0.01) -> list[dict[str, Any]]:
    kept=[]; kept_h=[]; cosine=math.cos(float(tau_h))
    for row in rows:
        h=np.asarray(row["h"],float)
        h=canon(h)
        if kept_h and float(np.max(np.abs(np.asarray(kept_h) @ h))) >= cosine:
            # Preserve seed-family provenance when two analytic families
            # discover the same RP2 basin.
            for old in kept:
                if rp_angle(h, np.asarray(old["h"])) <= tau_h:
                    aliases = old.setdefault("source_aliases", [old.get("source")])
                    if row.get("source") not in aliases:
                        aliases.append(row.get("source"))
                    break
            continue
        row["h"] = h.tolist(); row.setdefault("source_aliases", [row.get("source")]); kept.append(row); kept_h.append(h)
    return kept


def preflight(arcs: list[dict[str, Any]], crop: dict[str, Any]) -> dict[str, Any]:
    groups=ring_groups(arcs); R=len(groups)
    lengths={rid:sum(float(a["length_norm"]) for a in aa) for rid,aa in groups.items()}
    total=float(sum(lengths.values()))
    T=np.vstack([np.asarray(a["tangents"],float) for a in arcs]) if arcs else np.empty((0,2))
    if len(T):
        theta=np.mod(np.arctan2(T[:,1],T[:,0]),math.pi)
        hist,_=np.histogram(theta,bins=max(4,min(36,int(np.sqrt(len(theta))*2))),range=(0,math.pi))
        span=float(np.mean(hist>0)); M=T.T@T/len(T); eig=np.linalg.eigvalsh(M)
        lam_min=float(max(eig[0],0.0)); lam_max=float(max(eig[-1],0.0))
        tspan=float(np.ptp(np.sort(theta))) if len(theta)>1 else 0.0
    else: span=0.0; lam_min=lam_max=0.0; tspan=0.0
    A,w=weighted_rows(arcs)
    G=(A.T*w)@A if len(A) else np.zeros((3,3)); ge=np.linalg.eigvalsh(G) if len(A) else np.zeros(3)
    gsvd=float((ge[1]-ge[0])/max(ge[-1],1e-15)) if len(A)>=3 else None
    cent=[]
    for aa in groups.values():
        xx=np.vstack([np.asarray(a["points_norm"],float) for a in aa]); cent.append(np.mean(xx,axis=0))
    baseline=float(np.linalg.norm(np.asarray(cent)-np.mean(cent,axis=0),axis=1).max()) if len(cent)>1 else 0.0
    trunc=bool((crop.get("pith_px") is None) or crop.get("pith_full_px") is None)
    route="POINT_CANDIDATE" if R>=2 and total>=0.05 else "INSUFFICIENT_GEOMETRY"
    if R>=2 and (span<0.2 or lam_min<0.02): route="AXIS_ONLY"
    return {"sample_id":crop["sample_id"],"tree_id":crop["tree_id"],"section_id":crop["section_id"],"crop_id":crop["crop_id"],
            "eligible_parent_ring_count":R,"total_arc_norm":total,"Theta_total_rad":float(sum(float(a.get("theta_total_rad",0)) for a in arcs)),
            "B_spatial":baseline,"T_span_rad":tspan,"angular_coverage":span,"lambda_min_Mt":lam_min,"lambda_max_Mt":lam_max,
            "G_A_eigenvalues":ge.tolist(),"g_svd":gsvd,"truncation_ratio":float(trunc),
            "pith_in_crop":crop.get("pith_px") is not None,"target_domain":crop.get("target_domain","TARGET_UNKNOWN"),"route":route,
            "reason_codes":[] if route=="POINT_CANDIDATE" else ["INSUFFICIENT_GEOMETRY"]}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--stage1-dir",required=True); ap.add_argument("--output-dir",required=True); args=ap.parse_args()
    src,out=Path(args.stage1_dir),Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; seed_rows=[]; family_counts=defaultdict(int); mesh_seeds=cs6_mesh()
    # Stage 1 emits fragments grouped in the same crop order as crop_summary.
    # Stream both files so the 170 MB arc file never becomes a multi-GB object.
    fragment_file=(src/"fragments.jsonl").open("r",encoding="utf-8")
    pending=None
    def next_fragment():
        for raw in fragment_file:
            if raw.strip(): return json.loads(raw)
        return None
    pending=next_fragment()
    with (src/"crop_summary.jsonl").open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            crop=json.loads(line); cid=str(crop["crop_id"]); arcs=[]
            while pending is not None and str(pending["crop_id"]) == cid:
                arcs.append(pending); pending=next_fragment()
            pf=preflight(arcs,crop); rows.append(pf)
            seeds=[]; h,_=cs1_svd(arcs)
            if h is not None: seeds.append({"h":h.tolist(),"source":"CS1"})
            s23,ncombo=cs2_cs3(arcs); seeds.extend(s23); seeds.extend(cs4_circles(arcs)); seeds.extend(cs5_far_axis(arcs)); seeds.extend(mesh_seeds); seeds=dedupe(seeds)
            for s in seeds:
                for family in s.get("source_aliases", [s["source"]]): family_counts[family]+=1
            seed_rows.append({"sample_id":crop["sample_id"],"crop_id":crop["crop_id"],"tree_id":crop["tree_id"],"preflight_route":pf["route"],"seed_count":len(seeds),"cs2_combinations":ncombo,"seeds":seeds})
    fragment_file.close()
    def write(path, data):
        with path.open("w",encoding="utf-8") as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    write(out/"preflight.jsonl",rows); write(out/"seed_registry.jsonl",seed_rows)
    summary={"stage":2,"schema":"ArcPith-GT-v4-stage2","records":len(rows),"route_counts":dict(__import__('collections').Counter(r["route"] for r in rows)),"seed_family_counts":dict(family_counts),"mean_seeds":float(np.mean([r["seed_count"] for r in seed_rows])) if seed_rows else 0.0}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"README.md").write_text("Stage 2 stores candidate-free descriptors and a six-family RP2 cold-start registry. Seeds are not final estimates and do not use GT.\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if rows else 2


if __name__=="__main__": raise SystemExit(main())
