import os
import math
import numpy as np
import open3d as o3d

def load_mesh(path):
    """Load a mesh file (STL, OBJ, PLY)."""
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to read mesh: {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh

def save_mesh(path, mesh):
    """Save a mesh file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)

def compute_scale_metrics(mesh):
    """Compute bounding box diagonal and mean edge length."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    if len(tris) == 0:
        return {"bbox_diag": diag, "mean_edge": max(diag * 0.002, 1e-6)}
    # Sample edges for mean length
    edges = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
    # Take a subset for speed if too large
    if len(edges) > 10000:
        idx = np.random.choice(len(edges), 10000, replace=False)
        edges = edges[idx]
    e_len = np.linalg.norm(verts[edges[:,0]] - verts[edges[:,1]], axis=1)
    L = float(np.median(e_len)) if len(e_len) else max(diag * 0.002, 1e-6)
    return {"bbox_diag": diag, "mean_edge": L}

def build_vertex_kdtree(mesh):
    """Build a KDTree for mesh vertices."""
    verts = np.asarray(mesh.vertices)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    return kdt, verts

def project_point_to_surface(point, kdt, verts, normals):
    """Find nearest vertex to a point."""
    q = np.asarray(point, dtype=float)
    _, idxs, _ = kdt.search_knn_vector_3d(q, 1)
    vidx = int(idxs[0])
    return verts[vidx], normals[vidx], vidx

def falloff_weight(dist, radius, mode="smooth"):
    """Calculate displacement weight based on distance."""
    x = np.clip(dist / max(radius, 1e-9), 0.0, 1.0)
    if mode == "linear":   return 1.0 - x
    if mode == "gaussian": return np.exp(-0.5 * (dist/(radius/2.355+1e-9))**2)
    # smoothstep-like
    return 1.0 - (3*x**2 - 2*x**3)

def find_nearest_vertex(mesh, xyz):
    """Simple nearest vertex search (brute force or via KDTree if cached)."""
    pts = np.asarray(mesh.vertices)
    # Brute force for single point is fine, or use KDTree if performance needed
    idx = np.argmin(np.linalg.norm(pts - xyz[None, :], axis=1))
    return idx

def local_tangent_basis(mesh, vidx):
    """Get normal and two tangent vectors at a vertex."""
    norms = np.asarray(mesh.vertex_normals)
    n = norms[vidx]
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.8:
        a = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2) + 1e-12)
    return n, t1, t2

def apply_pit(mesh, center, radius, depth, falloff="smooth"):
    """Apply a pit/dent at the specified center."""
    m = o3d.geometry.TriangleMesh(mesh)
    v = np.asarray(m.vertices)
    n = np.asarray(m.vertex_normals)
    
    kdt, verts = build_vertex_kdtree(m)
    c_surf, _, _ = project_point_to_surface(center, kdt, verts, n)
    
    dists = np.linalg.norm(v - c_surf, axis=1)
    mask = dists <= radius
    
    if not np.any(mask):
        return m
        
    w = falloff_weight(dists[mask], radius, mode=falloff) * depth
    v[mask] -= n[mask] * w[:, None]
    
    m.vertices = o3d.utility.Vector3dVector(v)
    m.compute_vertex_normals()
    return m

def apply_scratch(mesh, path_world, width, depth, profile="gaussian"):
    """Apply a scratch along a path of points."""
    m = o3d.geometry.TriangleMesh(mesh)
    v = np.asarray(m.vertices)
    n = np.asarray(m.vertex_normals)
    
    kdt, verts = build_vertex_kdtree(m)
    
    # Project path points to surface
    P = []
    for p in np.asarray(path_world, float):
        ps, _, _ = project_point_to_surface(p, kdt, verts, n)
        P.append(ps)
    P = np.asarray(P)
    
    if len(P) < 2: return m
    
    # Distance to segment function
    def dist_to_segment(points, a, b):
        ab = b - a
        ab2 = np.dot(ab, ab) + 1e-12
        t = np.clip(((points - a) @ ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.linalg.norm(points - proj, axis=1)
    
    # Compute min distance to any segment in path
    dist = np.full((len(v),), np.inf, dtype=float)
    for i in range(len(P)-1):
        dist = np.minimum(dist, dist_to_segment(v, P[i], P[i+1]))
        
    mask = dist <= (1.5 * width)
    if not np.any(mask):
        return m
        
    if profile == "gaussian":
        sigma = width / 2.355
        groove = np.exp(-0.5 * (dist[mask] / (sigma + 1e-9))**2) * depth
    else:
        groove = np.clip(1.0 - dist[mask] / width, 0.0, 1.0) * depth
        
    v[mask] -= n[mask] * groove[:, None]
    
    m.vertices = o3d.utility.Vector3dVector(v)
    m.compute_vertex_normals()
    return m
