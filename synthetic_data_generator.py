#!/usr/bin/env python3
"""
Synthetic STL Data Generator for ML Training

Generates synthetic "OK" and "NG" STL files by applying learned deviation patterns
to a nominal CAD model. Uses per-face statistics from real scans to create realistic
manufacturing defects for training classification models.

Key Features:
- Captures ALL face deviations (not just worst faces) for realistic patterns
- Preserves the natural deviation distribution across the part
- Faces with low deviation stay good; only problem faces show defects
"""
import os
import json
import numpy as np
import trimesh
import argparse
from pathlib import Path
from scipy.spatial import cKDTree
import gmsh
from datetime import datetime

PROJECT_ROOT = "/Users/hemal/Desktop/point-cloud-project"


class DeviationProfile:
    """Statistical profile of deviations for a quality class (OK or NG)"""
    
    def __init__(self, name="unnamed"):
        self.name = name
        self.face_stats = {}  # {face_tag: {'mean': .., 'std': .., 'p95': ..}}
        self.global_stats = {}
        self.total_faces_in_cad = 0  # Track how many faces the CAD has
        
    def add_face_stats(self, face_tag, mean, std, p95, max_dev):
        """Add statistics for a specific face"""
        self.face_stats[str(face_tag)] = {
            'mean': mean,
            'std': std,
            'p95': p95,
            'max': max_dev
        }
    
    def set_global_stats(self, mean_of_means, mean_of_p95, mean_of_max, 
                         total_faces=0, faces_with_stats=0):
        """Set aggregate statistics across all faces"""
        self.global_stats = {
            'mean_of_means': mean_of_means,
            'mean_of_p95': mean_of_p95,
            'mean_of_max': mean_of_max,
            'total_faces_in_cad': total_faces,
            'faces_with_stats': faces_with_stats,
            'coverage_percent': (faces_with_stats / total_faces * 100) if total_faces > 0 else 0
        }
        self.total_faces_in_cad = total_faces
    
    def get_face_stats(self, face_tag, default_mean=0.01, default_std=0.005):
        """
        Get stats for a face, with fallback for faces not in profile.
        
        If face not in profile, returns minimal deviation (near-zero).
        This ensures faces that weren't analyzed (because they're fine) 
        get realistic low deviations.
        """
        face_tag_str = str(face_tag)
        if face_tag_str in self.face_stats:
            return self.face_stats[face_tag_str]
        else:
            # Face not in profile - assume it's a "good" face with minimal deviation
            return {
                'mean': default_mean,
                'std': default_std,
                'p95': default_mean * 2,
                'max': default_mean * 3
            }
    
    def save(self, path):
        """Save profile to JSON"""
        data = {
            'name': self.name,
            'face_stats': self.face_stats,
            'global_stats': self.global_stats,
            'total_faces_in_cad': self.total_faces_in_cad
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved profile: {path}")
    
    @classmethod
    def load(cls, path):
        """Load profile from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        profile = cls(name=data.get('name', 'loaded'))
        profile.face_stats = data['face_stats']
        profile.global_stats = data.get('global_stats', {})
        profile.total_faces_in_cad = data.get('total_faces_in_cad', 0)
        return profile
    
    def summary(self):
        """Print profile summary"""
        n_faces = len(self.face_stats)
        if n_faces == 0:
            return "Empty profile"
        
        means = [s['mean'] for s in self.face_stats.values()]
        p95s = [s['p95'] for s in self.face_stats.values()]
        
        lines = [
            f"Profile: {self.name}",
            f"  Faces with stats: {n_faces}",
            f"  Mean deviation range: {min(means):.4f} - {max(means):.4f} mm",
            f"  P95 deviation range:  {min(p95s):.4f} - {max(p95s):.4f} mm",
        ]
        if self.global_stats:
            lines.append(f"  Coverage: {self.global_stats.get('coverage_percent', 0):.1f}%")
        return "\n".join(lines)


def analyze_all_faces_from_step(step_path, scan_stl_path, nominal_stl_path,
                                 samples_per_face=5000, mesh_min=0.3, mesh_max=0.8):
    """
    Analyze ALL faces of a CAD model against a scan to build complete deviation profile.
    
    This is different from the main analysis script which only does detailed analysis
    on the worst K faces. Here we need complete coverage for realistic synthetic generation.
    
    Args:
        step_path: Path to STEP file
        scan_stl_path: Path to scanned STL
        nominal_stl_path: Path to nominal STL (for ICP alignment)
        samples_per_face: Points to sample per face
        mesh_min/max: Mesh density parameters
    
    Returns:
        DeviationProfile with stats for ALL faces
    """
    from measure_479_dm import DirectMeshMeasurer
    
    print("\n" + "="*70)
    print("COMPLETE FACE ANALYSIS (All Faces)")
    print("="*70)
    print(f"STEP file: {step_path}")
    print(f"Scan STL:  {scan_stl_path}")
    print(f"Samples per face: {samples_per_face}")
    print("="*70)
    
    # Step 1: Align scan to nominal
    print("\n[1/3] Aligning scan to nominal with ICP...")
    meas = DirectMeshMeasurer(project_root=PROJECT_ROOT)
    meas.load_stl_mesh_direct(scan_stl_path)
    meas.load_reference_model(nominal_stl_path)
    ok = meas.align_to_reference_with_icp()
    if not ok:
        raise RuntimeError("ICP alignment failed")
    print(f"  ICP Fitness: {meas.icp_fitness:.4f}")
    print(f"  ICP RMSE:    {meas.icp_rmse:.4f} mm")
    
    # Build KD-tree from aligned scan
    scan_vertices = np.asarray(meas.trimesh_object.vertices)
    # Use more points for accurate analysis
    target_pts = min(500000, scan_vertices.shape[0])
    if scan_vertices.shape[0] > target_pts:
        sel = np.random.choice(scan_vertices.shape[0], size=target_pts, replace=False)
        ref_points = scan_vertices[sel]
    else:
        ref_points = scan_vertices
    ref_tree = cKDTree(ref_points)
    print(f"  Built KD-tree with {len(ref_points)} points")
    
    # Step 2: Load STEP and mesh all faces
    print("\n[2/3] Loading STEP and extracting face geometry...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cad_analysis")
    
    try:
        try:
            gmsh.open(step_path)
        except:
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_max)
        gmsh.model.mesh.generate(2)
        
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        node_map = {int(tag): node_coords[i] for i, tag in enumerate(node_tags)}
        
        # Get ALL surface faces
        faces = gmsh.model.getEntities(2)
        total_faces = len(faces)
        print(f"  Found {total_faces} CAD faces")
        
        # Step 3: Analyze each face
        print(f"\n[3/3] Analyzing all {total_faces} faces ({samples_per_face} pts/face)...")
        
        profile = DeviationProfile(name="complete_analysis")
        all_means = []
        all_p95s = []
        all_maxs = []
        
        for i, (dim, tag) in enumerate(faces, start=1):
            # Get triangles for this face
            tris = _triangles_for_entity(dim, tag, node_map)
            
            if tris.shape[0] == 0:
                # Empty face - use zero deviation
                profile.add_face_stats(tag, 0.0, 0.0, 0.0, 0.0)
                continue
            
            # Sample points on face
            pts = _sample_points_from_triangles(tris, samples_per_face)
            
            if len(pts) == 0:
                profile.add_face_stats(tag, 0.0, 0.0, 0.0, 0.0)
                continue
            
            # Measure deviations to scan
            distances, _ = ref_tree.query(pts, k=1, workers=-1)
            
            mean = float(np.mean(distances))
            std = float(np.std(distances))
            p95 = float(np.percentile(distances, 95))
            max_dev = float(np.max(distances))
            
            profile.add_face_stats(tag, mean, std, p95, max_dev)
            
            all_means.append(mean)
            all_p95s.append(p95)
            all_maxs.append(max_dev)
            
            if i % 50 == 0 or i == total_faces:
                print(f"  Analyzed {i}/{total_faces} faces...")
        
        # Set global stats
        profile.set_global_stats(
            mean_of_means=float(np.mean(all_means)) if all_means else 0,
            mean_of_p95=float(np.mean(all_p95s)) if all_p95s else 0,
            mean_of_max=float(np.mean(all_maxs)) if all_maxs else 0,
            total_faces=total_faces,
            faces_with_stats=len(all_means)
        )
        
        print(f"\n✓ Analysis complete!")
        print(f"  Total faces analyzed: {len(all_means)}/{total_faces}")
        print(f"  Average mean deviation: {profile.global_stats['mean_of_means']:.4f} mm")
        print(f"  Average P95 deviation:  {profile.global_stats['mean_of_p95']:.4f} mm")
        
        return profile
        
    finally:
        gmsh.finalize()


def _triangles_for_entity(dim, tag, node_map):
    """Extract triangles from a Gmsh entity"""
    types, _, node_tags_per_type = gmsh.model.mesh.getElements(dim, tag)
    tris = []
    
    for etype, nlist in zip(types, node_tags_per_type):
        arr = np.array(nlist, dtype=np.int64)
        
        if etype == 2:  # tri3
            if arr.size % 3 != 0:
                continue
            arr = arr.reshape(-1, 3)
            for tri in arr:
                tris.append(np.stack([node_map[int(n)] for n in tri], axis=0))
        elif etype == 9:  # tri6
            if arr.size % 6 != 0:
                continue
            arr = arr.reshape(-1, 6)[:, :3]
            for tri in arr:
                tris.append(np.stack([node_map[int(n)] for n in tri], axis=0))
        elif etype == 3:  # quad4
            if arr.size % 4 != 0:
                continue
            arr = arr.reshape(-1, 4)
            for q in arr:
                tris.append(np.stack([node_map[int(n)] for n in [q[0], q[1], q[2]]], axis=0))
                tris.append(np.stack([node_map[int(n)] for n in [q[0], q[2], q[3]]], axis=0))
        elif etype == 10:  # quad9
            if arr.size % 9 != 0:
                continue
            arr = arr.reshape(-1, 9)[:, :4]
            for q in arr:
                tris.append(np.stack([node_map[int(n)] for n in [q[0], q[1], q[2]]], axis=0))
                tris.append(np.stack([node_map[int(n)] for n in [q[0], q[2], q[3]]], axis=0))
    
    if not tris:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.stack(tris, axis=0)


def _sample_points_from_triangles(tris, n_samples):
    """Sample random points uniformly on triangle surfaces"""
    if tris.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]
    
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = np.sum(areas)
    
    if total <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    probs = areas / total
    idx = np.random.choice(tris.shape[0], size=n_samples, p=probs)
    
    u = np.sqrt(np.random.rand(n_samples, 1))
    v = np.random.rand(n_samples, 1)
    a = 1 - u
    b = u * (1 - v)
    c = u * v
    
    tri_sel = tris[idx]
    pts = a * tri_sel[:, 0, :] + b * tri_sel[:, 1, :] + c * tri_sel[:, 2, :]
    return pts


def learn_profile_from_analysis_json(analysis_json_path, profile_name="profile"):
    """
    Create deviation profile from existing analysis JSON.
    
    NOTE: This only works well if the analysis covered many/all faces.
    For best results, use analyze_all_faces_from_step() instead.
    """
    with open(analysis_json_path, 'r') as f:
        data = json.load(f)
    
    profile = DeviationProfile(name=profile_name)
    faces = data.get('faces', [])
    
    all_means = []
    all_p95s = []
    all_maxs = []
    
    for face in faces:
        tag = face['tag']
        metrics = face['metrics']
        mean = metrics['mean']
        p95 = metrics['p95']
        max_dev = metrics['max']
        rmse = metrics['rmse']
        
        variance = max(rmse**2 - mean**2, 0)
        std = np.sqrt(variance)
        
        profile.add_face_stats(tag, mean, std, p95, max_dev)
        all_means.append(mean)
        all_p95s.append(p95)
        all_maxs.append(max_dev)
    
    profile.set_global_stats(
        mean_of_means=float(np.mean(all_means)) if all_means else 0,
        mean_of_p95=float(np.mean(all_p95s)) if all_p95s else 0,
        mean_of_max=float(np.mean(all_maxs)) if all_maxs else 0,
        total_faces=len(faces),  # Only know faces we have
        faces_with_stats=len(faces)
    )
    
    print(f"\n{'='*60}")
    print(f"Learned Profile: {profile_name}")
    print(f"{'='*60}")
    print(f"Faces in profile: {len(faces)}")
    print(f"Average mean deviation: {profile.global_stats['mean_of_means']:.4f} mm")
    print(f"Average P95 deviation:  {profile.global_stats['mean_of_p95']:.4f} mm")
    print(f"{'='*60}")
    
    if len(faces) < 100:
        print(f"\n⚠️  WARNING: Only {len(faces)} faces in profile!")
        print(f"   For realistic synthetic data, use --analyze-all to capture ALL faces.")
        print(f"   Faces not in profile will use near-zero deviation (may be unrealistic).")
    
    return profile


def load_step_with_face_mapping(step_path, size_min=0.3, size_max=0.8):
    """
    Load STEP file and create mapping from face tags to mesh vertices.
    """
    print(f"\nLoading STEP file: {step_path}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("nominal")
    
    try:
        try:
            gmsh.open(step_path)
        except:
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_max)
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        node_map = {int(tag): node_coords[i] for i, tag in enumerate(node_tags)}
        
        faces = gmsh.model.getEntities(2)
        print(f"  Found {len(faces)} CAD faces")
        
        face_mesh_map = {}
        all_vertices = []
        all_faces_tri = []
        vertex_offset = 0
        
        for dim, tag in faces:
            elem_types, _, node_tags_per_type = gmsh.model.mesh.getElements(dim, tag)
            
            face_vertices = []
            face_triangles = []
            local_node_map = {}
            
            for etype, nlist in zip(elem_types, node_tags_per_type):
                arr = np.array(nlist, dtype=np.int64)
                
                if etype == 2:
                    arr = arr.reshape(-1, 3)
                elif etype == 9:
                    arr = arr.reshape(-1, 6)[:, :3]
                elif etype == 3:
                    arr = arr.reshape(-1, 4)
                    tri_arr = []
                    for q in arr:
                        tri_arr.append([q[0], q[1], q[2]])
                        tri_arr.append([q[0], q[2], q[3]])
                    arr = np.array(tri_arr)
                elif etype == 10:
                    arr = arr.reshape(-1, 9)[:, :4]
                    tri_arr = []
                    for q in arr:
                        tri_arr.append([q[0], q[1], q[2]])
                        tri_arr.append([q[0], q[2], q[3]])
                    arr = np.array(tri_arr)
                else:
                    continue
                
                for tri in arr:
                    tri_indices = []
                    for node_tag in tri:
                        node_tag = int(node_tag)
                        if node_tag not in local_node_map:
                            local_node_map[node_tag] = len(face_vertices)
                            face_vertices.append(node_map[node_tag])
                        tri_indices.append(local_node_map[node_tag])
                    face_triangles.append(tri_indices)
            
            if face_vertices:
                face_vertices = np.array(face_vertices)
                face_triangles = np.array(face_triangles)
                
                face_mesh_map[tag] = {
                    'vertices': face_vertices,
                    'faces': face_triangles,
                    'vertex_offset': vertex_offset,
                    'vertex_count': len(face_vertices)
                }
                
                all_vertices.append(face_vertices)
                all_faces_tri.append(face_triangles + vertex_offset)
                vertex_offset += len(face_vertices)
        
        all_vertices = np.vstack(all_vertices)
        all_faces_tri = np.vstack(all_faces_tri)
        full_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces_tri)
        
        print(f"  Total vertices: {len(all_vertices)}")
        print(f"  Total triangles: {len(all_faces_tri)}")
        
        return face_mesh_map, full_mesh
        
    finally:
        gmsh.finalize()


def apply_deviation_to_mesh(mesh, face_mesh_map, profile, 
                            ok_profile=None,
                            scale_factor=1.0, 
                            random_seed=None,
                            default_deviation=0.01,
                            smoothness='face'):
    """
    Apply SMOOTH, REALISTIC deviations to mesh based on profile.
    
    Key improvements for realism:
    1. Deviations are applied per-FACE (not per-vertex) for smooth surfaces
    2. Deviations are BIDIRECTIONAL (can be + or -, centered at 0)
    3. Small random variation within each face for natural look
    
    Args:
        mesh: Original trimesh
        face_mesh_map: Mapping of face tags to vertex indices
        profile: Deviation profile with per-face statistics
        ok_profile: Optional fallback profile
        scale_factor: Multiplier for deviation magnitude
        random_seed: For reproducibility
        default_deviation: Fallback if face not in profile
        smoothness: 'face' (uniform per face) or 'smooth' (gradual transitions)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    new_vertices = mesh.vertices.copy()
    
    # Compute vertex normals properly
    vertex_normals = np.zeros_like(new_vertices)
    face_normals = mesh.face_normals
    
    for i, face in enumerate(mesh.faces):
        for vertex_idx in face:
            vertex_normals[vertex_idx] += face_normals[i]
    
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vertex_normals = vertex_normals / norms
    
    # Pre-compute a single deviation value per CAD face (for smoothness)
    face_deviations = {}
    
    for face_tag, face_info in face_mesh_map.items():
        face_tag_str = str(face_tag)
        
        # Get deviation parameters for this face
        if face_tag_str in profile.face_stats:
            stats = profile.face_stats[face_tag_str]
            magnitude = stats['mean'] * scale_factor
        elif ok_profile and face_tag_str in ok_profile.face_stats:
            stats = ok_profile.face_stats[face_tag_str]
            magnitude = stats['mean'] * scale_factor
        else:
            magnitude = default_deviation * scale_factor
        
        # BIDIRECTIONAL: Sample from normal distribution centered at 0
        # The magnitude determines the spread, direction is random
        # This gives realistic +/- deviations (some areas bigger, some smaller)
        base_deviation = np.random.normal(0, magnitude * 0.7)
        
        face_deviations[face_tag] = base_deviation
    
    # Apply deviations to vertices
    for face_tag, face_info in face_mesh_map.items():
        v_offset = face_info['vertex_offset']
        v_count = face_info['vertex_count']
        vertex_indices = np.arange(v_offset, v_offset + v_count)
        
        # Get the base deviation for this face
        base_dev = face_deviations.get(face_tag, 0.0)
        
        # Add tiny per-vertex variation (5% of base) for natural look without bumpiness
        # This is much smaller than before - just micro-texture, not noise
        micro_variation = np.random.normal(0, abs(base_dev) * 0.05 + 0.001, 
                                           size=len(vertex_indices))
        
        # Apply smooth deviation + micro variation along normals
        for i, idx in enumerate(vertex_indices):
            if idx < len(vertex_normals):
                total_deviation = base_dev + micro_variation[i]
                new_vertices[idx] += vertex_normals[idx] * total_deviation
    
    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)


def generate_synthetic_dataset(step_path, ng_profile, output_dir,
                               ok_profile=None,
                               n_ok=50, n_ng=50, 
                               scale_variation=0.2,
                               ok_scale=0.15,
                               ng_scale=0.5,
                               ok_default_deviation=0.005,
                               ng_default_deviation=0.01):
    """
    Generate synthetic dataset of OK and NG STL files.
    
    For realistic data:
    - OK samples: Use ok_profile if available, else minimal deviation on all faces
    - NG samples: Use ng_profile for problem faces, minimal for others
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Generating {n_ok} OK + {n_ng} NG samples")
    print(f"Scale variation: ±{scale_variation*100:.0f}%")
    print("="*70)
    
    # Check profile coverage
    n_ng_faces = len(ng_profile.face_stats)
    print(f"\nNG Profile: {n_ng_faces} faces with deviation data")
    if ok_profile:
        n_ok_faces = len(ok_profile.face_stats)
        print(f"OK Profile: {n_ok_faces} faces with deviation data")
    else:
        print("OK Profile: None (using minimal deviation)")
    
    # Load nominal mesh
    face_mesh_map, nominal_mesh = load_step_with_face_mapping(step_path)
    total_cad_faces = len(face_mesh_map)
    
    print(f"\nCAD model has {total_cad_faces} faces")
    coverage = n_ng_faces / total_cad_faces * 100 if total_cad_faces > 0 else 0
    print(f"NG Profile coverage: {coverage:.1f}% of faces")
    
    if coverage < 50:
        print(f"\n⚠️  Low coverage warning!")
        print(f"   Only {n_ng_faces}/{total_cad_faces} faces have NG deviation data.")
        print(f"   Uncovered faces will use minimal deviation ({ng_default_deviation:.3f} mm).")
        print(f"   For better realism, re-analyze with more faces or use --analyze-all.")
    
    # Save nominal
    nominal_path = os.path.join(output_dir, "nominal_reference.stl")
    nominal_mesh.export(nominal_path)
    print(f"\n✓ Saved nominal reference: {nominal_path}")
    
    # Generate OK samples
    print(f"\nGenerating {n_ok} OK samples...")
    print(f"  OK scale factor: {ok_scale} (smaller = closer to nominal)")
    ok_dir = os.path.join(output_dir, "OK")
    os.makedirs(ok_dir, exist_ok=True)
    
    ok_metadata = []
    for i in range(n_ok):
        # Random variation around the base scale
        scale = ok_scale * (1.0 + np.random.uniform(-scale_variation, scale_variation))
        
        if ok_profile:
            # Use actual OK profile with reduced scale for tighter tolerances
            synthetic_mesh = apply_deviation_to_mesh(
                nominal_mesh, face_mesh_map, ok_profile,
                scale_factor=scale, 
                random_seed=1000 + i,
                default_deviation=ok_default_deviation,
                smoothness='face'
            )
        else:
            # Create minimal-deviation "perfect" profile for OK parts
            perfect_profile = DeviationProfile(name="perfect")
            for tag in face_mesh_map.keys():
                perfect_profile.add_face_stats(tag, 
                                               ok_default_deviation, 
                                               ok_default_deviation * 0.5, 
                                               ok_default_deviation * 1.5, 
                                               ok_default_deviation * 2.0)
            synthetic_mesh = apply_deviation_to_mesh(
                nominal_mesh, face_mesh_map, perfect_profile,
                scale_factor=scale,
                random_seed=1000 + i,
                default_deviation=ok_default_deviation,
                smoothness='face'
            )
        
        filename = f"ok_sample_{i:04d}.stl"
        filepath = os.path.join(ok_dir, filename)
        synthetic_mesh.export(filepath)
        
        ok_metadata.append({
            'filename': filename,
            'label': 'OK',
            'sample_id': i,
            'scale_factor': float(scale),
            'seed': 1000 + i
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_ok} OK samples...")
    
    print(f"✓ Generated {n_ok} OK samples")
    
    # Generate NG samples
    print(f"\nGenerating {n_ng} NG samples...")
    print(f"  NG scale factor: {ng_scale} (larger = more deviation from nominal)")
    ng_dir = os.path.join(output_dir, "NG")
    os.makedirs(ng_dir, exist_ok=True)
    
    ng_metadata = []
    for i in range(n_ng):
        # Random variation around the base scale
        scale = ng_scale * (1.0 + np.random.uniform(-scale_variation, scale_variation))
        
        synthetic_mesh = apply_deviation_to_mesh(
            nominal_mesh, face_mesh_map, ng_profile,
            ok_profile=ok_profile,  # Use OK stats for faces not in NG profile
            scale_factor=scale,
            random_seed=2000 + i,
            default_deviation=ng_default_deviation,
            smoothness='face'
        )
        
        filename = f"ng_sample_{i:04d}.stl"
        filepath = os.path.join(ng_dir, filename)
        synthetic_mesh.export(filepath)
        
        ng_metadata.append({
            'filename': filename,
            'label': 'NG',
            'sample_id': i,
            'scale_factor': float(scale),
            'seed': 2000 + i
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_ng} NG samples...")
    
    print(f"✓ Generated {n_ng} NG samples")
    
    # Save metadata
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'nominal_step': step_path,
        'ng_profile': ng_profile.name,
        'ok_profile': ok_profile.name if ok_profile else None,
        'scale_variation': scale_variation,
        'profile_coverage': {
            'total_cad_faces': total_cad_faces,
            'ng_faces_with_data': n_ng_faces,
            'coverage_percent': coverage
        },
        'dataset': {
            'ok_samples': ok_metadata,
            'ng_samples': ng_metadata
        },
        'statistics': {
            'total_samples': n_ok + n_ng,
            'ok_count': n_ok,
            'ng_count': n_ng,
            'ng_profile_stats': ng_profile.global_stats
        }
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total samples: {n_ok + n_ng}")
    print(f"  OK: {n_ok} in {ok_dir}")
    print(f"  NG: {n_ng} in {ng_dir}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic STL data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: Analyze all faces from scan (most realistic)
  python synthetic_data_generator.py analyze-all \\
    --step 479-Part-model.stp \\
    --scan 479_ng.stl \\
    --nominal nominal_479_cad.stl \\
    --output-dir profiles \\
    --profile-name NG

  # Alternative: Learn from existing analysis (if you have one with many faces)
  python synthetic_data_generator.py learn \\
    --ng-analysis analysis_outputs/ng_part/analysis_ng.json \\
    --output-dir profiles

  # Generate synthetic dataset
  python synthetic_data_generator.py generate \\
    --step 479-Part-model.stp \\
    --ng-profile profiles/ng_profile.json \\
    --n-ok 100 --n-ng 100 \\
    --output-dir synthetic_dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze-all command (RECOMMENDED)
    analyze_parser = subparsers.add_parser('analyze-all', 
        help='Analyze ALL faces of a part (recommended for best results)')
    analyze_parser.add_argument('--step', required=True, help='Path to STEP file')
    analyze_parser.add_argument('--scan', required=True, help='Path to scanned STL')
    analyze_parser.add_argument('--nominal', required=True, help='Path to nominal STL')
    analyze_parser.add_argument('--samples-per-face', type=int, default=5000,
                               help='Points to sample per face (default: 5000)')
    analyze_parser.add_argument('--output-dir', default='profiles',
                               help='Output directory for profile')
    analyze_parser.add_argument('--profile-name', default='analyzed',
                               help='Name for this profile (e.g., OK, NG)')
    
    # Learn command (from existing analysis JSON)
    learn_parser = subparsers.add_parser('learn', 
        help='Learn from existing analysis JSON (may have incomplete coverage)')
    learn_parser.add_argument('--ok-analysis', help='Path to OK part analysis JSON')
    learn_parser.add_argument('--ng-analysis', required=True, 
                             help='Path to NG part analysis JSON')
    learn_parser.add_argument('--output-dir', default='profiles',
                             help='Output directory for profiles')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    gen_parser.add_argument('--step', required=True, help='Path to nominal STEP file')
    gen_parser.add_argument('--ok-profile', help='Path to OK deviation profile JSON')
    gen_parser.add_argument('--ng-profile', required=True,
                           help='Path to NG deviation profile JSON')
    gen_parser.add_argument('--n-ok', type=int, default=50,
                           help='Number of OK samples (default: 50)')
    gen_parser.add_argument('--n-ng', type=int, default=50,
                           help='Number of NG samples (default: 50)')
    gen_parser.add_argument('--scale-variation', type=float, default=0.2,
                           help='Random scale variation (default: 0.2 = ±20%%)')
    gen_parser.add_argument('--ok-scale', type=float, default=0.15,
                           help='Scale factor for OK parts (default: 0.15, smaller = less deviation)')
    gen_parser.add_argument('--ng-scale', type=float, default=0.5,
                           help='Scale factor for NG parts (default: 0.5, larger = more deviation)')
    gen_parser.add_argument('--output-dir', default='synthetic_dataset',
                           help='Output directory for dataset')
    
    args = parser.parse_args()
    
    if args.command == 'analyze-all':
        os.makedirs(args.output_dir, exist_ok=True)
        
        profile = analyze_all_faces_from_step(
            args.step,
            args.scan,
            args.nominal,
            samples_per_face=args.samples_per_face
        )
        profile.name = args.profile_name
        
        output_path = os.path.join(args.output_dir, f"{args.profile_name.lower()}_profile.json")
        profile.save(output_path)
        
        print(f"\n✓ Profile saved: {output_path}")
        print(profile.summary())
        
    elif args.command == 'learn':
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.ok_analysis:
            print("\nLearning OK profile...")
            ok_profile = learn_profile_from_analysis_json(args.ok_analysis, "OK")
            ok_path = os.path.join(args.output_dir, "ok_profile.json")
            ok_profile.save(ok_path)
        
        print("\nLearning NG profile...")
        ng_profile = learn_profile_from_analysis_json(args.ng_analysis, "NG")
        ng_path = os.path.join(args.output_dir, "ng_profile.json")
        ng_profile.save(ng_path)
        
    elif args.command == 'generate':
        # Load NG profile
        ng_profile = DeviationProfile.load(args.ng_profile)
        print(f"✓ Loaded NG profile: {ng_profile.name}")
        print(f"  Faces: {len(ng_profile.face_stats)}")
        
        # Load OK profile if provided
        ok_profile = None
        if args.ok_profile:
            ok_profile = DeviationProfile.load(args.ok_profile)
            print(f"✓ Loaded OK profile: {ok_profile.name}")
            print(f"  Faces: {len(ok_profile.face_stats)}")
        
        generate_synthetic_dataset(
            step_path=args.step,
            ng_profile=ng_profile,
            ok_profile=ok_profile,
            output_dir=args.output_dir,
            n_ok=args.n_ok,
            n_ng=args.n_ng,
            scale_variation=args.scale_variation,
            ok_scale=args.ok_scale,
            ng_scale=args.ng_scale
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
