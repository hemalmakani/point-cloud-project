#!/usr/bin/env python3
"""
scan.py - Identify defects in a 3D scan by comparing to a reference model.
"""

import json
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree, ConvexHull
from sklearn.cluster import DBSCAN

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh

def mesh_to_point_cloud(mesh, n_points=20000):
    try:
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    except Exception:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.estimate_normals()
    return pcd

def bbox_diagonal(geom):
    aabb = geom.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()))

def icp_align(good_pcd, bad_pcd, D):
    # Coarse alignment
    voxel_coarse = 0.01 * D
    good_c = good_pcd.voxel_down_sample(voxel_coarse)
    bad_c = bad_pcd.voxel_down_sample(voxel_coarse)
    good_c.estimate_normals(); bad_c.estimate_normals()
    
    reg1 = o3d.pipelines.registration.registration_icp(
        bad_c, good_c, 2 * voxel_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    )
    
    # Fine alignment
    voxel_fine = 0.002 * D
    bad_full = o3d.geometry.PointCloud(bad_pcd)
    bad_full.transform(reg1.transformation)
    bad_full.estimate_normals()
    
    reg2 = o3d.pipelines.registration.registration_icp(
        bad_full, good_pcd, 2 * voxel_fine,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    )
    return reg2.transformation @ reg1.transformation

def analyze(good_path, bad_path, out_path, n_points=20000, alpha=0.005):
    print(f"Scanning {bad_path} against {good_path}...")
    
    good_mesh = load_mesh(good_path)
    bad_mesh = load_mesh(bad_path)
    
    good_pcd = mesh_to_point_cloud(good_mesh, n_points)
    bad_pcd = mesh_to_point_cloud(bad_mesh, n_points)
    
    D = bbox_diagonal(good_pcd)
    
    # Align
    T = icp_align(good_pcd, bad_pcd, D)
    bad_mesh.transform(T)
    bad_pcd = mesh_to_point_cloud(bad_mesh, n_points)
    
    # Compare
    good_pts = np.asarray(good_pcd.points)
    good_nrm = np.asarray(good_pcd.normals)
    bad_pts = np.asarray(bad_pcd.points)
    
    tree = cKDTree(bad_pts)
    dists, idx = tree.query(good_pts, k=1, workers=-1)
    disp = bad_pts[idx] - good_pts
    signed_d = np.sign(np.einsum('ij,ij->i', disp, good_nrm)) * dists
    
    # Cluster defects
    t = max(alpha * D, 1e-9)
    mask = np.abs(signed_d) > t
    
    defects = []
    if np.any(mask):
        seeds_idx = np.where(mask)[0]
        eps = max(0.01 * D, 1e-9)
        labels = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit(good_pts[seeds_idx]).labels_
        
        full_labels = np.full(len(good_pts), -1)
        full_labels[seeds_idx] = labels
        
        for lab in set(labels):
            if lab == -1: continue
            idx = np.where(full_labels == lab)[0]
            pts = good_pts[idx]
            depths = signed_d[idx]
            
            # PCA for dimensions
            c = pts.mean(axis=0)
            X = pts - c
            eigvals, eigvecs = np.linalg.eigh((X.T @ X) / len(pts))
            dims = np.ptp(X @ eigvecs, axis=0) # range of projections
            L, W, H = sorted(dims)[::-1]
            
            # Type guess
            aspect = L / (W + 1e-9)
            dtype = "scratch" if aspect > 3.0 else "pit_or_chip"
            
            defects.append({
                "type_guess": dtype,
                "centroid": c.tolist(),
                "length_width_height": [float(L), float(W), float(H)],
                "aspect_LW": float(aspect),
                "depth_abs_stats": {
                    "mean": float(np.abs(depths).mean()),
                    "max": float(np.abs(depths).max())
                }
            })
            
    output = {
        "good_mesh": str(good_path),
        "bad_mesh": str(bad_path),
        "D": D,
        "defects": defects
    }
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved analysis to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", required=True)
    parser.add_argument("--bad", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--points", type=int, default=20000)
    args = parser.parse_args()
    analyze(args.good, args.bad, args.out, n_points=args.points, alpha=args.alpha)
