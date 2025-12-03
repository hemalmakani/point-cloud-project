#!/usr/bin/env python3
"""
recreate.py - Learn patterns from a scan JSON and generate new synthetic datasets.
"""

import json
import argparse
import numpy as np
import abrasion_ops as ops
from pathlib import Path
from collections import defaultdict

def fit_distribution(values):
    """Fit a simple Normal distribution to values."""
    vals = np.array(values, dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0: return {"mean": 1.0, "std": 0.1}
    return {"mean": float(vals.mean()), "std": float(vals.std())}

def learn_model(json_path):
    """Learn statistical model from a scan JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
        
    model = {"D": data.get("D", 1.0), "types": {}}
    grouped = defaultdict(lambda: defaultdict(list))
    
    for d in data.get("defects", []):
        t = d["type_guess"]
        L, W, H = d["length_width_height"]
        grouped[t]["length"].append(L)
        grouped[t]["width"].append(W)
        grouped[t]["depth"].append(d["depth_abs_stats"]["max"])
        
    for t, stats in grouped.items():
        model["types"][t] = {
            "length": fit_distribution(stats["length"]),
            "width": fit_distribution(stats["width"]),
            "depth": fit_distribution(stats["depth"]),
            "count_weight": len(stats["length"]) / len(data["defects"])
        }
    return model

def sample_val(dist, rng):
    val = rng.normal(dist["mean"], dist["std"])
    return max(val, 1e-6)

def describe_location(centroid, bbox_min, bbox_max):
    """Translate 3D coordinates into natural language location."""
    x, y, z = centroid
    min_x, min_y, min_z = bbox_min
    max_x, max_y, max_z = bbox_max
    
    # Normalize to 0-1 range
    nx = (x - min_x) / (max_x - min_x)
    ny = (y - min_y) / (max_y - min_y)
    nz = (z - min_z) / (max_z - min_z)
    
    parts = []
    
    # Vertical (Z)
    if nz > 0.8: parts.append("top")
    elif nz < 0.2: parts.append("bottom")
    
    # Horizontal (X) - Assuming X is Left/Right
    if nx > 0.7: parts.append("right")
    elif nx < 0.3: parts.append("left")
    
    # Depth (Y) - Assuming Y is Front/Back
    if ny > 0.7: parts.append("back")
    elif ny < 0.3: parts.append("front")
    
    if not parts:
        return "center"
        
    return " ".join(parts)

def generate_description(defect_type, centroid, dims, bbox_min, bbox_max):
    """Generate a full natural language sentence."""
    loc_str = describe_location(centroid, bbox_min, bbox_max)
    L, W, D = dims
    
    # Size adjectives
    size_adj = "small"
    if L > 10.0: size_adj = "large"
    elif L > 5.0: size_adj = "medium"
    
    # Depth adjectives
    depth_adj = "shallow"
    if D > 0.5: depth_adj = "deep"
    
    # Clean up type name
    type_name = defect_type.replace("_", " ")
    
    return f"A {size_adj}, {depth_adj} {type_name} located on the {loc_str} of the component."

def generate(good_path, pattern_json, count, out_dir):
    print(f"Learning from {pattern_json}...")
    model = learn_model(pattern_json)
    rng = np.random.default_rng()
    
    base_mesh = ops.load_mesh(good_path)
    
    # Compute bbox for semantic translation
    bbox = base_mesh.get_axis_aligned_bounding_box()
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        mesh = ops.o3d.geometry.TriangleMesh(base_mesh)
        
        # Determine number of defects (randomized around original count)
        # For simplicity, let's say 3-8 defects per part
        n_defects = rng.integers(3, 8)
        
        generated_defects = []
        
        for _ in range(n_defects):
            # Pick type based on weights
            types = list(model["types"].keys())
            weights = [model["types"][t]["count_weight"] for t in types]
            t = rng.choice(types, p=weights)
            params = model["types"][t]
            
            # Sample dimensions
            L = sample_val(params["length"], rng)
            W = sample_val(params["width"], rng)
            D = sample_val(params["depth"], rng)
            
            # Random location on mesh
            v_idx = rng.integers(0, len(mesh.vertices))
            center = np.asarray(mesh.vertices)[v_idx]
            
            # Map type to operation
            is_linear = t in ["scratch", "long_scratch", "crack"]
            
            if is_linear:
                # Create a random path
                n, t1, t2 = ops.local_tangent_basis(mesh, v_idx)
                # Random direction in tangent plane
                angle = rng.uniform(0, 2*np.pi)
                dir_vec = np.cos(angle)*t1 + np.sin(angle)*t2
                p0 = center - (dir_vec * L * 0.5)
                p1 = center + (dir_vec * L * 0.5)
                mesh = ops.apply_scratch(mesh, [p0, p1], width=W, depth=D)
            else:
                # Pit / Gouge / Crack (non-linear)
                mesh = ops.apply_pit(mesh, center, radius=max(L,W)/2, depth=D)
            
            # Generate natural language description
            desc = generate_description(t, center, [L, W, D], bbox_min, bbox_max)
                
            generated_defects.append({
                "type": t,
                "centroid": center.tolist(),
                "LWD": [L, W, D],
                "description": desc
            })
            
        out_name = f"synth_{i:03d}.stl"
        out_path = Path(out_dir) / out_name
        ops.save_mesh(str(out_path), mesh)
        
        # Save descriptor
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump({
                "source_pattern": str(pattern_json),
                "defects": generated_defects
            }, f, indent=2)
            
        print(f"Generated {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    
    generate(args.good, args.pattern, args.count, args.out_dir)
