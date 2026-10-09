import os
import cv2
import math
import json
import time
import argparse
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

from skimage import measure
from sklearn.mixture import GaussianMixture
from scipy.stats import kendalltau



PICSPACE = 3072

# -----------------------------
# Utilities: IO & Geometry
# -----------------------------
# (保持不变)
def read_train_samples(train_txt):
    """Parse train.txt (space-separated): rel_path cx cy

    Returns [{'dir': dirname, 'tag': 'B'|'D', 'center': [cx, cy]}, ...]
    """
    samples = []
    if not os.path.exists(train_txt):
        print(f"[Warn] {train_txt} not found.")
        return samples
    with open(train_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: dataset\\dirname\\ann.png cx cy
            parts = line.rsplit(maxsplit=2)
            if len(parts) < 3:
                continue
            rel_path = parts[0].replace('\\', '/')
            path_parts = rel_path.split('/')
            # path_parts = ['dataset', 'T0_B1_N27_B_s1000_n0', 'B_ann.png']
            dirname = path_parts[-2]           # directory name
            filename = path_parts[-1]          # B_ann.png or D_ann.png
            tag = filename.split('_')[0]       # 'B' or 'D'
            cx = float(parts[1])
            cy = float(parts[2])
            samples.append({'dir': dirname, 'tag': tag, 'center': [cx, cy]})
    return samples

def load_ann_png(path):
    ann = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if ann is None: raise FileNotFoundError(path)
    edge_mask = (ann < 255)
    return ann, edge_mask

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def sample_edge_points(edge_mask, max_points=10000, seed=0):
    H, W = edge_mask.shape
    ys, xs = np.where(edge_mask)
    n = len(xs)
    if n == 0: return np.zeros((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs = xs.astype(np.float32) - (W - 1) / 2.0
    ys = ys.astype(np.float32) - (H - 1) / 2.0
    return np.stack([xs, ys], axis=1)

def connected_components(edge_mask):
    H, W = edge_mask.shape
    num, lab = cv2.connectedComponents(edge_mask.astype(np.uint8), connectivity=8)
    comps = []
    for k in range(1, num):
        ys, xs = np.where(lab == k)
        if len(xs) < 8: continue 
        xs = xs.astype(np.float32) - (W - 1) / 2.0
        ys = ys.astype(np.float32) - (H - 1) / 2.0
        comps.append(np.stack([xs, ys], axis=1))
    return comps

def extract_tau_values_from_ann(ann_u8, edge_mask):
    vals = ann_u8[edge_mask]
    if vals.size == 0: return []
    uniq = np.unique(vals)
    uniq = uniq[uniq < 255]
    return np.sort(uniq)[::-1].tolist()

# -----------------------------
# Memory Optimized Stage 1
# -----------------------------
def vectorized_cold_start(edge_mask, topN=10, grid_step=16, max_points=15000, device="cuda"):
    torch.cuda.empty_cache()
    H, W = edge_mask.shape
    pts_np = sample_edge_points(edge_mask, max_points=max_points, seed=42)
    if len(pts_np) < 10: return [], []
    
    comps = connected_components(edge_mask)
    valid_comps = [c for c in comps if len(c) > 15] 
    if not valid_comps: return [], []

    half = PICSPACE / 2.0
    line_coords = torch.arange(-half, half, grid_step, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(line_coords, line_coords, indexing="ij")
    candidates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1) 
    
    scores = torch.zeros(candidates.shape[0], device=device)
    batch_size = 1024 
    
    gpu_comps = []
    try:
        for c in valid_comps:
            gpu_comps.append(torch.from_numpy(c).to(device))
    except RuntimeError:
        gpu_comps = valid_comps

    for i in range(0, candidates.shape[0], batch_size):
        end = min(i + batch_size, candidates.shape[0])
        cands_batch = candidates[i:end].unsqueeze(1)
        batch_score = torch.zeros(cands_batch.shape[0], device=device)
        
        for comp in gpu_comps:
            if isinstance(comp, torch.Tensor):
                pts = comp.unsqueeze(0)
            else:
                pts = torch.from_numpy(comp).to(device).unsqueeze(0)
            
            diff = pts - cands_batch 
            dists = torch.norm(diff, dim=-1)
            
            medians = dists.median(dim=1).values.unsqueeze(1)
            dists.sub_(medians).abs_() 
            mads = dists.median(dim=1).values
            
            weight = math.sqrt(pts.shape[1])
            batch_score += weight / (mads + 1e-6)
            
        scores[i:end] = batch_score
        
        if i % (batch_size * 5) == 0:
            del diff, dists, mads
            torch.cuda.empty_cache()

    top_scores, idx = torch.topk(scores, k=topN)
    top_centers = candidates[idx]
    
    del candidates, scores
    torch.cuda.empty_cache()
    
    return top_centers.cpu().numpy(), top_scores.cpu().numpy()

# -----------------------------
# Models
# -----------------------------
class RBFDelta(nn.Module):
    def __init__(self, mus_xy, sigma=160.0):
        super().__init__()
        self.register_buffer("mus", mus_xy)
        self.sigma = float(sigma)
        self.a = nn.Parameter(torch.zeros(mus_xy.shape[0], dtype=torch.float32))

    def forward(self, x):
        x_norm = (x**2).sum(-1, keepdim=True)
        mus_norm = (self.mus**2).sum(-1).unsqueeze(0)
        dist_sq = x_norm + mus_norm - 2 * (x @ self.mus.t())
        return (torch.exp(-dist_sq / (2 * self.sigma**2)) @ self.a)

def build_rbf_centers(grid_step=128, device="cpu"):
    half = PICSPACE / 2.0
    coords = np.arange(-half - 200, half + 200, grid_step, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    mus = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], 1)).to(device)
    return mus

class RadialConsistencyLoss(nn.Module):
    def __init__(self, num_rays=64, samples_per_ray=16):
        super().__init__()
        self.num_rays = num_rays
        self.samples_per_ray = samples_per_ray

    def forward(self, phi_func, c, max_r=200.0):
        """
        Enforce radial consistency of gradient directions.
        Distortions (knots) should propagate radially, meaning the gradient direction 
        along a ray from the pith should change smoothly, not chaotically.
        """
        device = c.device
        
        # 1. Construct Rays
        # Random starting angles to prevent overfitting to fixed grid
        angles = torch.rand(self.num_rays, device=device) * 2 * math.pi
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        dirs = torch.stack([cos_a, sin_a], dim=1) # (N_rays, 2)
        
        # 2. Sample points along rays (Linear spacing from near center to edge)
        # r range: [10, max_r]
        rs = torch.linspace(10.0, max_r, self.samples_per_ray, device=device)
        
        # (samples, rays) mesh
        # P = C + r * dir
        # shape: (samples, rays, 2)
        ray_pts = c.view(1, 1, 2) + rs.view(-1, 1, 1) * dirs.view(1, -1, 2)
        
        # Flatten for batch processing: (samples*rays, 2)
        flat_pts = ray_pts.view(-1, 2).requires_grad_(True)
        
        # 3. Compute Gradients
        vals = phi_func(flat_pts, c)
        grads = torch.autograd.grad(vals.sum(), flat_pts, create_graph=True)[0]
        
        # Normalize gradients to focus purely on DIRECTION (geometry consistency)
        # Avoid div by zero
        gnorms = torch.norm(grads, dim=1, keepdim=True) + 1e-6
        g_dirs = grads / gnorms
        
        # Reshape back to (samples, rays, 2)
        g_dirs_reshaped = g_dirs.view(self.samples_per_ray, self.num_rays, 2)
        
        # 4. Compute Consistency Loss
        # Compare gradient at r[i] with r[i+1]
        # We want Cosine Similarity to be 1.0 -> Loss = 1 - CosSim
        # neighboring points along the ray should have similar normals
        g_inner = g_dirs_reshaped[:-1, :, :]
        g_outer = g_dirs_reshaped[1:, :, :]
        
        dot_product = (g_inner * g_outer).sum(dim=2) # (samples-1, rays)
        
        # If there is a knot, the normal usually deflects. 
        # But physics say: deflection at r should correlate with deflection at r+dr.
        # This Smoothness constraint allows deflection, but penalizes jerky high-freq noise.
        loss_radial = (1.0 - dot_product).mean()
        
        return loss_radial

# -----------------------------
# Revised C4 Symmetry Logic
# -----------------------------

def rotate_points_matrix(pts, angle_deg):
    r"""
    Rotates points counter-clockwise around the origin (0,0).
    Group Action g \cdot x
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    # Standard Rotation Matrix for column vectors: [cos -sin; sin cos]
    # For row vectors (N,2): [x y] @ R.T
    R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=pts.device)
    return pts @ R.t()

class ConsistencyLossC4(nn.Module):
    def __init__(self): 
        super().__init__()
        self.angles = [90.0, 180.0, 270.0]

    def forward(self, phi_func_closure, c_curr, pts_edge):
        r"""
        Implements self-supervised consistency under C4 Group Action.

        Mathematical Basis:
        scalar field equivariance: Phi(g \cdot x, g \cdot c) == Phi(x, c)

        Rationale:
        If a point 'x' has a growth value 'v' relative to pith 'c',
        then if we rotate the entire world (point x becomes x', pith c becomes c') by 90 degrees,
        the growth value at x' relative to c' must still be 'v'.
        """
        loss = 0.0
        
        # 1. Calculate the canonical field values (Identity Element)
        # Phi_identity = Phi(x, c)
        phi_original = phi_func_closure(pts_edge, c_curr)
        
        for ang in self.angles:
            # 2. Apply Group Action to the Input Space (Covariance of coordinates)
            # x' = R * x
            pts_rot = rotate_points_matrix(pts_edge, ang)
            
            # 3. Apply Group Action to the Pith Parameter (Covariance of parameters)
            # c' = R * c
            # We must detach c_rot to avoid trying to optimize c via the rotation logic weirdly,
            # or keep it attached if we want c to satisfy geometric symmetry. 
            # Keeping it attached creates a strong constraint on c position.
            c_rot = rotate_points_matrix(c_curr.unsqueeze(0), ang).squeeze(0)
            
            # 4. Calculate Field in Transformed Domain
            # Phi_rotated = Phi(x', c')
            phi_transformed = phi_func_closure(pts_rot, c_rot)
            
            # 5. Enforce Invariance constraint: Phi(x) - Phi(g.x) = 0
            # We use L1 loss/Huber loss style
            loss += (phi_original - phi_transformed).abs().mean()
            
        return loss / len(self.angles)

# -----------------------------
# Evaluation Metrics & Analysis Helpers
# -----------------------------
class EvaluationMetrics:
    @staticmethod
    def compute_coverage_accuracy(pred_contours, edge_mask, distance_threshold=5.0):
        """
        Compute coverage and accuracy.
        """
        gt_pts = sample_edge_points(edge_mask, max_points=20000, seed=0)
        if len(gt_pts) == 0 or len(pred_contours) == 0:
            return 0.0, 0.0
        
        all_pred_pts = np.vstack(pred_contours)
        if len(all_pred_pts) > 20000:
            idx = np.random.choice(len(all_pred_pts), 20000, replace=False)
            all_pred_pts = all_pred_pts[idx]
        
        d_gt2pred = cdist(gt_pts, all_pred_pts, metric='euclidean')
        min_dists_gt = d_gt2pred.min(axis=1)
        coverage = np.mean(min_dists_gt <= distance_threshold)
        
        d_pred2gt = cdist(all_pred_pts, gt_pts, metric='euclidean')
        min_dists_pred = d_pred2gt.min(axis=1)
        accuracy = np.mean(min_dists_pred <= distance_threshold)
        
        return float(coverage), float(accuracy)

    @staticmethod
    def compute_instance_metrics(phi_func, c, edge_mask, device, n_gt_map):
        """Calculates Phase and Geometric Concentricity Potential per pixel on edges"""
        pts_np = sample_edge_points(edge_mask, max_points=50000, seed=0) 
        if len(pts_np) == 0: return None, None, None
        
        H, W = edge_mask.shape
        t_pts = torch.from_numpy(pts_np).to(device).requires_grad_(True)
        
        # 1. Phase (Phi)
        phi_val = phi_func(t_pts, c)
        
        # 2. Gradient of Field
        grad = torch.autograd.grad(phi_val.sum(), t_pts, create_graph=False)[0]
        grad_norm = torch.sqrt((grad**2).sum(dim=1) + 1e-8)
        n_pred = grad / grad_norm.unsqueeze(1)
        
        # 3. Ground Truth Normals lookup
        img_xs = (pts_np[:, 0] + (W - 1) / 2.0).astype(int)
        img_ys = (pts_np[:, 1] + (H - 1) / 2.0).astype(int)
        img_xs = np.clip(img_xs, 0, W-1)
        img_ys = np.clip(img_ys, 0, H-1)
        
        n_gt_samples = n_gt_map[img_ys, img_xs] # (N, 2) numpy
        n_gt_tensor = torch.from_numpy(n_gt_samples).to(device)
        
        # 4. Concentricity Potential
        alignment = (n_pred * n_gt_tensor).sum(dim=1).abs()
        
        return pts_np, phi_val.detach().cpu().numpy(), alignment.detach().cpu().numpy()

def compute_dt_normal(edge_mask, sigma=1.2):
    inv = (~edge_mask).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    k = int(2 * round(3 * sigma) + 1)
    Ds = cv2.GaussianBlur(dt, (k, k), sigmaX=sigma, sigmaY=sigma)
    gx = cv2.Sobel(Ds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ds, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx**2 + gy**2) + 1e-6
    n = np.stack([gx/norm, gy/norm], axis=-1)
    Wmask = ((Ds > 0.5) & (Ds < 6.0)).astype(np.float32)
    return n, Wmask, Ds

def marching_contours_chunked(phi_eval_fn, c_curr, taus, step=2, extent=PICSPACE/2, device="cpu"):
    coords = np.arange(-extent, extent, step, dtype=np.float32)
    H, W = len(coords), len(coords)
    
    phi_map = torch.zeros((H, W), dtype=torch.float32)
    row_block = 64
    
    with torch.no_grad():
        c_gpu = c_curr 
        for i in range(0, H, row_block):
            end_i = min(i + row_block, H)
            y_strip = coords[i:end_i]
            x_full = coords
            
            xx, yy = np.meshgrid(x_full, y_strip, indexing='xy')
            pts_strip = np.stack([xx.ravel(), yy.ravel()], axis=1)
            
            tpts = torch.from_numpy(pts_strip).to(device)
            val_strip = phi_eval_fn(tpts, c_gpu)
            phi_map[i:end_i, :] = val_strip.view(end_i - i, W).cpu()
            del tpts, val_strip
            
    phi_numpy = phi_map.numpy()
    contours_out = []
    for tau in taus:
        cs = measure.find_contours(phi_numpy, float(tau))
        for c_pts in cs:
            xw = coords[0] + c_pts[:, 1] * step
            yw = coords[0] + c_pts[:, 0] * step
            contours_out.append(np.stack([xw, yw], axis=1))
    return contours_out

# -----------------------------
# Visualization Functions
# -----------------------------
def visualize_growth_field_final(prefix, out_path, ann, edge_mask, c_xy, contours, arrow_labels):
    """
    Visualization 1 (Final): 
    Left: Input Centric (256x256) - Original Image + Local Fit
    Right: Pith Centric(Rescaled) - Global Fit + Transformed Input Image Overlay
    """
    H, W = ann.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=150)
    
    # --- LEFT PANEL: Input Centric ---
    ax1.set_aspect("equal")
    ax1.invert_yaxis() 
    
    half_w = (W - 1) / 2.0
    half_h = (H - 1) / 2.0
    local_extent = [-half_w, half_w, half_h, -half_h]
    
    from matplotlib.colors import ListedColormap
    edge_overlay = np.zeros((H, W, 4), dtype=np.float32)
    edge_overlay[edge_mask, :] = [0, 0, 0, 0.6] 
    
    ax1.imshow(edge_overlay, extent=local_extent, zorder=1)
    
    margin = 10
    for c in contours:
        xs, ys = c[:, 0], c[:, 1]
        if (xs.max() > -half_w - margin) and (xs.min() < half_w + margin) and \
           (ys.max() > -half_h - margin) and (ys.min() < half_h + margin):
            ax1.plot(xs, ys, color='red', linewidth=1.2, alpha=0.9, zorder=2)

    cx, cy = c_xy 
    for lab in arrow_labels:
        mask = (ann == lab)
        ys, xs = np.where(mask)
        if len(xs) < 10: continue
        
        mx = xs.mean() - half_w
        my = ys.mean() - half_h
        
        ax1.text(mx, my, str(lab), color='lime', fontsize=10, 
                 fontweight='bold', ha='center', va='center', zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        
        dx, dy = cx - mx, cy - my
        length = math.sqrt(dx*dx + dy*dy) + 1e-6
        arrow_len = 30.0
        ax1.arrow(mx, my, (dx/length)*arrow_len, (dy/length)*arrow_len, 
                  color='lime', head_width=6, length_includes_head=True, alpha=0.9, zorder=5)

    ax1.set_xlim(local_extent[0], local_extent[1])
    ax1.set_ylim(local_extent[2], local_extent[3]) 
    ax1.set_title(f"Input Image (256x256)\nRed: Fitted Field | Arrow: To Pith", fontsize=14)
    
    # --- RIGHT PANEL: Pith Centric ---
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    
    ax2.plot(0, 0, 'b+', markersize=20, markeredgewidth=2, zorder=10, label='Pith')
    
    max_dist = 0
    for c in contours:
        c_trans = c - np.array([cx, cy])
        curr_max = np.max(np.abs(c_trans))
        if curr_max > max_dist: max_dist = curr_max
        ax2.plot(c_trans[:, 0], c_trans[:, 1], color='red', linewidth=0.8, alpha=0.5, zorder=1)
        
    trans_extent = [
        local_extent[0] - cx, local_extent[1] - cx,
        local_extent[2] - cy, local_extent[3] - cy
    ]
    ax2.imshow(edge_overlay, extent=trans_extent, zorder=5)
    
    box_x = [trans_extent[0], trans_extent[1], trans_extent[1], trans_extent[0], trans_extent[0]]
    box_y = [trans_extent[3], trans_extent[3], trans_extent[2], trans_extent[2], trans_extent[3]]
    ax2.plot(box_x, box_y, color='cyan', linestyle='--', linewidth=2, zorder=6, label='Input Area')
    
    center_x_trans = -cx
    center_y_trans = -cy
    ax2.annotate("Input Instance", 
                 xy=(center_x_trans, center_y_trans),
                 xytext=(center_x_trans, center_y_trans + max(H, W)*0.6),
                 color='cyan', arrowprops=dict(arrowstyle="->", color='cyan', lw=1.5),
                 ha='center', fontweight='bold', zorder=20)

    img_dist = np.max(np.abs(trans_extent))
    view_lim = max(max_dist * 1.1, img_dist * 1.3)
    
    ax2.set_xlim(-view_lim, view_lim)
    ax2.set_ylim(view_lim, -view_lim)
    ax2.set_title(f"Global Field (Pith Centered)\nCyan: Transformed Input Position", fontsize=14)
    ax2.legend(loc='lower right')
    
    fig.suptitle(f'{prefix} Growth Field Reconstruction', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_metrics_panel_final(prefix, out_path, logs, trajectories, metric_history, analysis_data):
    """
    Visualization 2 (Metrics):
    Layout 3x3:
    [1] Total Loss      [2] Trajectory Map    [3] Coverage/Accuracy
    [4] Iso/Dir Loss    [5] Phase Map         [6] Concentricity Map
    [7] Reg Losses      [8] Phase Hist        [9] Concentricity Hist
    """
    if not logs: return

    iters = [log['iter'] for log in logs]
    losses = [log.get('loss', 0) for log in logs]
    stages = [log['stage'] for log in logs]
    
    # Data extraction
    l_iso = [log.get('L_iso', 0) for log in logs]
    l_dir = [log.get('L_dir', 0) for log in logs]
    l_c4  = [log.get('L_c4', 0) for log in logs]
    l_gap = [log.get('L_snap', 0) + log.get('L_d0', 0) for log in logs]
    
    pts, phase_vals, align_vals = analysis_data
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3) # 3 Rows, 3 Columns

    s2_end = next((i for i, s in enumerate(stages) if s==3), len(iters))

    # --- Row 1 ---
    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, losses, 'k-', lw=2, label='Total Loss')
    ax1.axvspan(0, s2_end, color='blue', alpha=0.1, label='Stage 2 (Geo)')
    ax1.axvspan(s2_end, len(iters), color='red', alpha=0.1, label='Stage 3 (Ring)')
    ax1.set_title("1. Total Optimization Loss")
    ax1.set_xlabel("Iter")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Pith Trajectory (NEW)
    ax2 = fig.add_subplot(gs[0, 1])
    if trajectories:
        # Separate by stage
        traj_s2 = trajectories[:s2_end]
        traj_s3 = trajectories[s2_end:]
        
        if traj_s2:
            tx2, ty2 = zip(*traj_s2)
            ax2.plot(tx2, ty2, 'b.-', alpha=0.6, markersize=3, label='Stage 2 Path')
        if traj_s3:
            tx3, ty3 = zip(*traj_s3)
            ax2.plot(tx3, ty3, 'r.-', alpha=0.6, markersize=3, label='Stage 3 Path')
            
        # Start/End
        start_pt = trajectories[0]
        end_pt = trajectories[-1]
        ax2.plot(start_pt[0], start_pt[1], 'go', markersize=10, label='Start')
        ax2.plot(end_pt[0], end_pt[1], 'k*', markersize=15, label='Final')
        
        ax2.set_aspect('equal')
        ax2.invert_yaxis() # Image coordinates match
        ax2.set_title("2. Pith Center Trajectory")
        ax2.set_xlabel("X (Image Space)")
        ax2.set_ylabel("Y (Image Space)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No Trajectory Data", ha='center')

    # 3. Coverage & Accuracy Evolution (NEW)
    ax3 = fig.add_subplot(gs[0, 2])
    if metric_history:
        m_iters = [m['iter'] for m in metric_history]
        m_cov = [m['cov'] for m in metric_history]
        m_acc = [m['acc'] for m in metric_history]
        
        ax3.plot(m_iters, m_cov, 'g-o', markersize=4, label='Coverage')
        ax3.plot(m_iters, m_acc, 'c-s', markersize=4, label='Accuracy')
        ax3.set_ylim(0, 1.05)
        ax3.axvspan(0, s2_end, color='blue', alpha=0.1)
        ax3.axvspan(s2_end, len(iters), color='red', alpha=0.1)
        ax3.set_title("3. Fit Quality Evolution")
        ax3.set_xlabel("Iter")
        ax3.set_ylabel("Score (0-1)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No Metric History", ha='center')

    # --- Row 2 ---
    # 4. Primary Loss Components
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(iters, l_iso, 'r-', alpha=0.7, label='L_iso (Shape)')
    ax4.plot(iters, l_dir, 'b-', alpha=0.7, label='L_dir (Normal)')
    ax4.plot(iters, l_c4, 'g-', alpha=0.7, label='L_c4 (Symm)')
    ax4.set_title("4. Shape & Geometry Losses")
    ax4.set_xlabel("Iter")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Phase Map
    ax5 = fig.add_subplot(gs[1, 1])
    if pts is not None:
        sc2 = ax5.scatter(pts[:,0], pts[:,1], c=phase_vals, cmap='plasma', s=2, alpha=0.8)
        plt.colorbar(sc2, ax=ax5, label='Phase')
        ax5.set_aspect('equal')
        ax5.invert_yaxis()
    ax5.set_title("5. Input Instance Phase Field")

    # 6. Concentricity Map
    ax6 = fig.add_subplot(gs[1, 2])
    if pts is not None:
        sc = ax6.scatter(pts[:,0], pts[:,1], c=align_vals, cmap='viridis', s=2, alpha=0.8)
        plt.colorbar(sc, ax=ax6, label='Dot Product')
        ax6.set_aspect('equal')
        ax6.invert_yaxis()
    ax6.set_title("6. Geometric Concentricity Potential")

    # --- Row 3 ---
    # 7. Regularization Losses
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(iters, l_gap, 'y-', alpha=0.8, label='L_reg (Gap+Reg)')
    ax7.set_title("7. Regularization Terms")
    ax7.set_xlabel("Iter")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Phase Histogram
    ax8 = fig.add_subplot(gs[2, 1])
    if pts is not None:
        ax8.hist(phase_vals, bins=50, color='purple', alpha=0.7)
    ax8.set_title("8. Phase Value Distribution")

    # 9. Concentricity Histogram
    ax9 = fig.add_subplot(gs[2, 2])
    if pts is not None:
        ax9.hist(align_vals, bins=50, color='teal', alpha=0.7)
    ax9.set_title("9. Concentricity Score Distribution")

    fig.suptitle(f"{prefix} Metrics Dashboard\nTotal Iters: {len(iters)} | Final Acc: {metric_history[-1]['acc'] if metric_history else 0:.3f}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=100)
    plt.close()

def check_ring_order_consistency(ann_u8, edge_mask, phi_func, c_final, device):
    """
    Checks if the order of Ground Truth rings matches the order of the Predicted Field.
    Returns:
        spearman_corr: Correlation coefficient (-1 to 1). 
                       1.0 = Perfect match, -1.0 = Perfect inverse (also fine), ~0 = Random.
        is_consistent: Boolean, True if |corr| > 0.8 (allows for some noise).
        details: List of (gt_label, pred_phi_mean).
    """
    # 1. Get unique existing labels in the annotation
    # We only care about ring labels (usually pixel values > 0 and < 255)
    # The background is usually 255 or 0 depending on preprocessing, assuming 255 is background based on code.
    unique_labels = np.unique(ann_u8[edge_mask])
    unique_labels = unique_labels[unique_labels < 255] # Exclude background
    unique_labels = np.sort(unique_labels)
    
    if len(unique_labels) < 2:
        return 1.0, True, [] # Only 1 ring, order is trivially consistent

    gt_vals = []
    pred_vals = []
    
    H, W = ann_u8.shape
    
    with torch.no_grad():
        for lab in unique_labels:
            ys, xs = np.where((ann_u8 == lab) & edge_mask)
            if len(xs) < 10: continue # Skip too small fragments
            
            # Sample some points to estimate the field value for this ring
            # Downsample if too many
            if len(xs) > 500:
                idx = np.random.choice(len(xs), 500, replace=False)
                ys, xs = ys[idx], xs[idx]
            
            # Convert to coordinates centered like the model expects
            xx = xs.astype(np.float32) - (W - 1) / 2.0
            yy = ys.astype(np.float32) - (H - 1) / 2.0
            
            pts = torch.from_numpy(np.stack([xx, yy], axis=1)).to(device)
            phi_val = phi_func(pts, c_final).mean().item()
            
            gt_vals.append(lab)
            pred_vals.append(phi_val)
            
    if len(gt_vals) < 2:
        return 1.0, True, list(zip(gt_vals, pred_vals))

    # Calculate Kendall's Tau or Spearman Correlation
    # We use simple correlation because linear relationship suggests monotonic mapping
    # Actually, Kendall-Tau is better for rank checking.
    tau, p_value = kendalltau(gt_vals, pred_vals)
    
    # We accept both positive (growing out) and negative (growing in) correlation as "Consistent" geometry
    is_consistent = abs(tau) > 0.8
    
    return tau, is_consistent, list(zip(gt_vals, pred_vals))

# -----------------------------
# Optimization Core
# -----------------------------
def optimize_growth_field(edge_mask, ann_u8, init_center, out_prefix, out_dir, device="cuda"):
    H, W = edge_mask.shape
    
    logs = []
    # NEW: History trackers
    traj_history = []
    metric_history = []
    
    # Sample points
    pts_edge_np = sample_edge_points(edge_mask, max_points=12000) 
    pts_edge = torch.from_numpy(pts_edge_np).to(device)
    
    # Connectivity
    comps_np = connected_components(edge_mask)
    comps = [torch.from_numpy(c).to(device) for c in comps_np]
    
    n_gt_np, Wmask_np, _ = compute_dt_normal(edge_mask)
    n_gt = torch.from_numpy(n_gt_np).to(device)
    Wmask = torch.from_numpy(Wmask_np).to(device)
    
    # Models
    c = torch.tensor(init_center, device=device, dtype=torch.float32, requires_grad=True)
    mus = build_rbf_centers(128, device)
    delta_model = RBFDelta(mus, 160.0).to(device)
    c4_loss = ConsistencyLossC4()
    radial_loss_fn = RadialConsistencyLoss(num_rays=48, samples_per_ray=12) 

    # Opt Stage 2
    opt = torch.optim.Adam([
        {'params': [c], 'lr': 1.5e-2}, 
        {'params': delta_model.parameters(), 'lr': 2e-2}
    ])

    def phi(x, center_v):
        r = torch.norm(x - center_v, dim=-1)
        return r + delta_model(x)

    def get_stochastic_patch_grad(center_val, batch_size=4000):
        if not hasattr(get_stochastic_patch_grad, "active_indices"):
            ys, xs = torch.where(Wmask > 0.001)
            get_stochastic_patch_grad.active_indices = (ys, xs)
            get_stochastic_patch_grad.num_active = len(ys)

        ys_all, xs_all = get_stochastic_patch_grad.active_indices
        if get_stochastic_patch_grad.num_active == 0:
            return torch.zeros((1, 2), device=device), torch.zeros((1, 2), device=device), torch.zeros(1, device=device)

        idx = torch.randint(0, get_stochastic_patch_grad.num_active, (batch_size,), device=device)
        y_sel = ys_all[idx]
        x_sel = xs_all[idx]
        
        xx = x_sel.float() - (W-1)/2.0
        yy = y_sel.float() - (H-1)/2.0
        pts = torch.stack([xx, yy], -1).requires_grad_(True)
        
        val = phi(pts, center_val)
        g_pts = torch.autograd.grad(val.sum(), pts, create_graph=True)[0]
        
        n_gt_sampled = n_gt[y_sel, x_sel]
        w_sampled = Wmask[y_sel, x_sel]
        
        return g_pts, n_gt_sampled, w_sampled

    # --- STAGE 2: Geometric Alignment ---
    iters_s2 = 400
    for it in range(iters_s2):
        opt.zero_grad()
        
        L_c4 = c4_loss(phi, c, pts_edge)
        
        L_iso = 0.0
        cur_comps = comps if len(comps)<40 else [comps[i] for i in np.random.choice(len(comps), 40)]
        for cp in cur_comps:
            ph = phi(cp, c)
            L_iso += (ph - ph.median()).abs().mean()
        L_iso /= max(1, len(cur_comps))

        g_pred, n_target, w_target = get_stochastic_patch_grad(c, batch_size=2048)
        g_dir = F.normalize(g_pred, dim=-1)
        align = (g_dir * n_target).sum(-1).abs()
        L_dir = (w_target * (1 - align)).sum() / (w_target.sum() + 1e-6)
        
        L_reg = (delta_model.a ** 2).mean()

        # Too much weight here prevents forming the knot shape.
        L_rad_grad = radial_loss_fn(phi, c, max_r=PICSPACE/2.0)

        # loss = 2.0*L_iso + 0.8*L_dir + 0.5*L_c4 + 0.05*L_reg
        loss = 2.0*L_iso + 0.8*L_dir + 0.5*L_c4 + 0.05*L_reg + 0.1*L_rad_grad
        loss.backward()
        opt.step()
        
        # LOGGING
        logs.append({
            'stage': 2, 'iter': it, 'loss': loss.item(),
            'L_rad': 0.0, 'L_iso': L_iso.item(), 'L_dir': L_dir.item(),
            'L_c4': L_c4.item(), 'L_d0': L_reg.item()
        })
        traj_history.append(c.detach().cpu().numpy().tolist())

        # NEW: Compute Metrics periodically (using estimated implicit rings for S2)
        if it % 50 == 0:
            with torch.no_grad():
                # Estimate temporary rings by percentile for checking alignment
                vals = phi(pts_edge, c).cpu().numpy()
                temp_taus = np.percentile(vals, np.linspace(10, 90, 5))
                contours_tmp = marching_contours_chunked(phi, c, temp_taus, device=device)
                cov, acc = EvaluationMetrics.compute_coverage_accuracy(contours_tmp, edge_mask)
                metric_history.append({'iter': it, 'cov': cov, 'acc': acc})

    if hasattr(get_stochastic_patch_grad, "active_indices"):
        del get_stochastic_patch_grad.active_indices
        del get_stochastic_patch_grad.num_active

    # --- STAGE 3: GMM & Coverage ---
    # Init GMM
        # ... (Stage 2 结束) ...

    # --- STAGE 3 INIT (Improved) ---
    with torch.no_grad():
        vals = phi(pts_edge, c).cpu().numpy().reshape(-1, 1)
        
        # 使用简单的 Kernel Density Estimation 思想找波峰，比 GMM 更适合从连续场找离散层级
        from sklearn.neighbors import KernelDensity
        from scipy.signal import find_peaks
        
        # 带宽决定了分辨年轮的能力，2.0 约等于 phi 值差 2 的单位
        kde = KernelDensity(kernel='gaussian', bandwidth=2.0).fit(vals)
        s_grid = np.linspace(vals.min(), vals.max(), 500)[:, None]
        log_pdf = kde.score_samples(s_grid)
        peaks, _ = find_peaks(log_pdf, distance=10) # distance 防止过密
        
        if len(peaks) > 3:
            taus_init = s_grid[peaks].flatten()
        else:
            # Fallback
            gmm = GaussianMixture(n_components=10, covariance_type='spherical').fit(vals)
            taus_init = gmm.means_.flatten()
            
        print(f"[{out_prefix}] Stage 3 Init: {len(taus_init)} rings.")

    taus_param = nn.Parameter(torch.tensor(taus_init, device=device, dtype=torch.float32))
    
    # 优化策略：
    # Stage 2 已经把几何形状拟合得很好，Stage 3 主要是为了定 Taus（年轮位置）。
    # 因此，大幅降低 c 和 delta 的 LR，大幅提高 taus 的 LR。
    opt = torch.optim.Adam([
        {'params': [c], 'lr': 1e-4},              # 极低的学习率，保持圆心
        {'params': delta_model.parameters(), 'lr': 1e-3}, # 低学习率，微调形状
        {'params': [taus_param], 'lr': 5e-2}      # 高学习率，快速对齐
    ])

    iters_s3 = 350
    for it in range(iters_s3):
        cur_total_iter = it + iters_s2
        opt.zero_grad()
        
        # 1. 动态对齐 Loss (Soft Alignment)
        # 初始阶段允许较大的吸附范围，后期收紧
        beta = 0.3 + 0.7 * (it / iters_s3)
        
        ph_pts = phi(pts_edge, c).view(-1, 1)
        t_sorted, _ = torch.sort(taus_param) 
        
        dist_mat = (ph_pts - t_sorted.view(1, -1)).abs()
        probs = F.softmax(-beta * dist_mat, dim=1) 
        L_fit = (probs * dist_mat).sum(1).mean()
        
        # 2. 覆盖率 Loss (Coverage)
        # 仅仅采样部分连通域，防止过拟合
        L_cover = 0.0
        # 增加采样数量以获得更稳定的梯度
        sub_comps = comps if len(comps)<50 else [comps[i] for i in np.random.choice(len(comps), 50)]
        for cp in sub_comps:
            ph_cp = phi(cp, c).mean()
            # 使用 Smooth L1 Loss 防止梯度爆炸
            min_dist = (ph_cp - t_sorted).abs().min()
            L_cover += min_dist
        L_cover /= len(sub_comps)

        # 3. 间隔正则化 (Gap Regularization)
        gaps = t_sorted[1:] - t_sorted[:-1]
        # 放宽最小间距限制，防止强行推开本来就很密的年轮
        L_gap_collision = F.relu(2.5 - gaps).sum() 
        L_gap_reg = gaps.std() # 使用 std 替代 var，量级更可控
        
        # 4. 几何约束 (Geometry)
        L_c4 = c4_loss(phi, c, pts_edge)
        
        # 5. 为了保持对齐，保留一定的方向 Loss，但权重降低
        if it % 2 == 0: # 隔代计算以加速
            g_pred, n_target, w_target = get_stochastic_patch_grad(c, batch_size=2048)
            g_dir = F.normalize(g_pred, dim=-1)
            L_dir = (w_target * (1 - (g_dir * n_target).sum(-1).abs())).sum() / (w_target.sum() + 1e-6)
        else:
            L_dir = torch.tensor(0.0, device=device)

        L_rad_grad = radial_loss_fn(phi, c, max_r=PICSPACE/2.0)

        # 权重调整：突出 Fitting，弱化 C4 和 Dir（因为 Stage 2 已经做好了）
        # loss = L_fit * 2.0 + L_cover * 0.5 + L_gap_collision * 1.0 + L_gap_reg * 0.1 + L_c4 * 0.1 + L_dir * 0.1
        loss = L_fit * 2.0 + L_cover * 0.5 + L_gap_collision * 1.0 + \
           L_gap_reg * 0.1 + L_c4 * 0.1 + L_dir * 0.1 + \
           0.5 * L_rad_grad 
        
        loss.backward()
        
        # 梯度裁剪，防止 taus 飞出天际
        torch.nn.utils.clip_grad_norm_([taus_param], max_norm=2.0)
        
        opt.step()

        
        # LOGGING
        logs.append({
            'stage': 3, 'iter': cur_total_iter, 'loss': loss.item(),
            'L_rad': L_cover.item(), 'L_iso': L_fit.item(), 'L_dir': L_dir.item(),
            'L_c4': L_c4.item(), 'L_snap': L_gap_collision.item(), 'L_d0': L_gap_reg.item()
        })
        traj_history.append(c.detach().cpu().numpy().tolist())

        # NEW: Compute Metrics periodically using actual Taus
        if it % 50 == 0:
            with torch.no_grad():
                taus_curr = t_sorted.detach().cpu().numpy()
                contours_tmp = marching_contours_chunked(phi, c, taus_curr, device=device)
                cov, acc = EvaluationMetrics.compute_coverage_accuracy(contours_tmp, edge_mask)
                metric_history.append({'iter': cur_total_iter, 'cov': cov, 'acc': acc})
    
    # Cleanup memory
    del g_pred, n_target, w_target, n_gt, Wmask, comps
    if hasattr(get_stochastic_patch_grad, "active_indices"):
        del get_stochastic_patch_grad.active_indices
    torch.cuda.empty_cache()

    # --- Analysis Data Generation ---
    analysis_data = EvaluationMetrics.compute_instance_metrics(phi, c, edge_mask, device, n_gt_np)
    ord_corr, ord_consistent, ord_details = check_ring_order_consistency(
        ann_u8, edge_mask, phi, c, device
    )

    # --- Final Output ---
    c_final = c.detach().cpu().numpy()
    taus_final = taus_param.detach().cpu().numpy()
    taus_final.sort()
    
    # Compute final contours
    contours = marching_contours_chunked(phi, c, taus_final, device=device)
    arrow_labels = extract_tau_values_from_ann(ann_u8, edge_mask)
    
    # Final coverage/accuracy
    final_coverage, final_accuracy = EvaluationMetrics.compute_coverage_accuracy(contours, edge_mask)
    # Add final point to history
    metric_history.append({'iter': iters_s2 + iters_s3, 'cov': final_coverage, 'acc': final_accuracy})
    
    # Visualization 1: Growth Field Split View
    growth_field_path = os.path.join(out_dir, f"{out_prefix}growth_field.png")
    visualize_growth_field_final(
        f"{out_prefix}", growth_field_path,
        ann_u8, edge_mask, c_final, contours, arrow_labels
    )
    
    # Visualization 2: Analysis Dashboard (UPDATED)
    metrics_panel_path = os.path.join(out_dir, f"{out_prefix}metrics_panel.png")
    visualize_metrics_panel_final(
        f"{out_prefix}", metrics_panel_path,
        logs, traj_history, metric_history, analysis_data
    )
    
    # Save metrics
    metrics = {
        "id": out_prefix,                             # ID
        "coverage": float(final_coverage),            # 覆盖率
        "accuracy": float(final_accuracy),            # 准确率
        "order_consistency_tau": float(ord_corr),     # 环序相关系数
        "is_order_consistent": bool(ord_consistent),  # 是否一致 (T/F)
        "num_rings_gt": len(ord_details),             # GT 识别到的环数
        "num_rings_pred": len(taus_final),            # 预测生成的环数
        "final_center": c_final.tolist(),
        "final_taus": taus_final.tolist()
    }
    with open(os.path.join(out_dir, f"{out_prefix}metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

# -----------------------------
# Main
# -----------------------------
def process_dataset(train_txt, root, out_dir, device="cuda"):
    samples = read_train_samples(train_txt)
    ensure_dir(out_dir)
    all_metrics = []

    # Prepare CSV headers
    csv_columns = ['id', 'coverage', 'accuracy', 'order_consistency_tau', 'is_order_consistent', 'num_rings_gt', 'num_rings_pred']

    for sample in tqdm(samples, desc="Samples"):
        torch.cuda.empty_cache()
        gc.collect()

        dirname = sample['dir']
        tag = sample['tag']
        init_center = sample['center']

        path = os.path.join(root, dirname, f"{tag}_ann.png")
        if not os.path.exists(path):
            tqdm.write(f"  [Skip] {path} not found")
            continue

        prefix = f"{dirname}_{tag}_"
        ann_u8, edge_mask = load_ann_png(path)

        # 直接使用 train.txt 中的真实髓心坐标，跳过 cold start
        m = optimize_growth_field(edge_mask, ann_u8, init_center, prefix, out_dir, device)

        # Simple print for immediate feedback
        status = "OK" if m['is_order_consistent'] else "BAD ORDER"
        tqdm.write(f"  -> {m['id']}: Acc={m['accuracy']:.3f}, Cov={m['coverage']:.3f}, Order={m['order_consistency_tau']:.2f} ({status})")

        all_metrics.append(m)
            
    if all_metrics:
        # Save detailed CSV
        df = pd.DataFrame(all_metrics)
        # Reorder columns for better readability
        cols = [c for c in csv_columns if c in df.columns] + [c for c in df.columns if c not in csv_columns]
        df = df[cols]
        
        df.to_csv(os.path.join(out_dir, "summary_report.csv"), index=False)
        
        print("\n" + "="*50)
        print("FINAL BATCH SUMMARY")
        print("="*50)
        # Print a clean table to console
        print(df[['id', 'accuracy', 'coverage', 'is_order_consistent', 'order_consistency_tau']].to_string(index=False))
        print("="*50)
        print(f"Mean Accuracy: {df['accuracy'].mean():.4f}")
        print(f"Mean Coverage: {df['coverage'].mean():.4f}")
        print(f"Consistency Rate: {df['is_order_consistent'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default=r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\train.txt")
    parser.add_argument("--root", default=r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\dataset")
    parser.add_argument("--out_dir", default=r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\output")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available(): args.device = "cpu"

    process_dataset(args.train_txt, args.root, args.out_dir, args.device)
