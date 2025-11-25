#!/usr/bin/env python3
"""
Direct Mesh Intersection Measurement Script for 479.stl with ICP Alignment
==========================================================================

This script uses exact geometric mesh-plane intersection instead of point cloud
sampling for maximum accuracy and deterministic results. Each measurement uses
trimesh's precise plane intersection algorithms.

Key Advantages:
- Perfect determinism: identical results every time
- Higher accuracy: no sampling approximation
- Complete coverage: uses all mesh triangles
- Eliminates measurement variation
- ICP alignment: automatically aligns scanned parts to reference CAD model

ICP (Iterative Closest Point) Alignment:
- Handles rotated, translated, or upside-down scanned parts
- Automatically aligns scan to reference coordinate system
- Ensures measurements are consistent regardless of scan orientation
- Provides quality metrics (fitness score and RMSE)

Usage:
    # Measure CAD model directly
    python measure_479_dm.py 479.stl
    
    # Measure scanned part with ICP alignment to reference
    python measure_479_dm.py scanned_479.stl --reference 479.stl
    
    # Measure upside-down scan (ICP will correct orientation)
    python measure_479_dm.py upside_down_479.stl --reference 479.stl
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import sys
import open3d as o3d


class DirectMeshMeasurer:
    """Direct mesh intersection measurement system for maximum accuracy"""
    
    def __init__(self, project_root="/Users/hemal/Desktop/point-cloud-project"):
        self.project_root = project_root
        self.trimesh_object = None
        self.reference_mesh = None
        self.plane_position = np.array([0.0, 0.0, 0.0])
        self.plane_normal = np.array([1.0, 0.0, 0.0])
        self.plane_rotation = np.array([0.0, 0.0, 0.0])
        self.intersection_points = None
        self.measurements = {}
        self.icp_fitness = None
        self.icp_rmse = None
        
    def load_stl_mesh_direct(self, stl_path):
        """Load STL file as trimesh for direct geometric intersection"""
        print(f"🔄 Loading STL mesh for direct intersection: {stl_path}")
        
        # Load as trimesh object (keeps full triangular mesh)
        # Use trimesh.load for proper mesh loading
        self.trimesh_object = trimesh.load(stl_path, force='mesh')
        
        # Check if mesh is watertight (optional validation)
        if hasattr(self.trimesh_object, 'is_watertight'):
            if not self.trimesh_object.is_watertight:
                print("⚠️ Warning: Mesh is not watertight, but proceeding...")
        
        print(f"✅ STL loaded: {len(self.trimesh_object.vertices):,} vertices, {len(self.trimesh_object.faces):,} triangles")
        
        # Center the plane at mesh center
        center = self.trimesh_object.bounds.mean(axis=0)
        self.plane_position = center
        
        print(f"📍 Plane centered at: [{self.plane_position[0]:.1f}, {self.plane_position[1]:.1f}, {self.plane_position[2]:.1f}]")
        
        return self.trimesh_object
    
    def load_reference_model(self, reference_stl_path):
        """Load reference CAD model for ICP alignment"""
        print(f"🔄 Loading reference CAD model: {reference_stl_path}")
        self.reference_mesh = trimesh.load(reference_stl_path, force='mesh')
        print(f"✅ Reference loaded: {len(self.reference_mesh.vertices):,} vertices, {len(self.reference_mesh.faces):,} triangles")
        return self.reference_mesh
    
    def align_to_reference_with_icp(self, num_samples=50000, voxel_size=1.0):
        """
        Align the scanned mesh to the reference model using ICP (Iterative Closest Point)
        This handles any rotation, translation, or flipping of the scanned part
        
        Args:
            num_samples: Number of points to sample for ICP (more = slower but more accurate)
            voxel_size: Voxel size for downsampling (larger = faster but less accurate)
        """
        if self.trimesh_object is None:
            print("❌ No scanned mesh loaded. Use load_stl_mesh_direct() first.")
            return False
        
        if self.reference_mesh is None:
            print("❌ No reference model loaded. Use load_reference_model() first.")
            return False
        
        print("\n🔄 Starting ICP alignment to reference model...")
        print(f"   Samples: {num_samples:,} points")
        print(f"   Voxel size: {voxel_size} mm")
        
        try:
            # Convert trimesh to Open3D for ICP
            scan_vertices = np.asarray(self.trimesh_object.vertices)
            scan_triangles = np.asarray(self.trimesh_object.faces)
            
            scan_mesh_o3d = o3d.geometry.TriangleMesh()
            scan_mesh_o3d.vertices = o3d.utility.Vector3dVector(scan_vertices)
            scan_mesh_o3d.triangles = o3d.utility.Vector3iVector(scan_triangles)
            
            ref_vertices = np.asarray(self.reference_mesh.vertices)
            ref_triangles = np.asarray(self.reference_mesh.faces)
            
            ref_mesh_o3d = o3d.geometry.TriangleMesh()
            ref_mesh_o3d.vertices = o3d.utility.Vector3dVector(ref_vertices)
            ref_mesh_o3d.triangles = o3d.utility.Vector3iVector(ref_triangles)
            
            # Sample point clouds from meshes
            print("   📊 Sampling point clouds...")
            scan_pcd = scan_mesh_o3d.sample_points_uniformly(number_of_points=num_samples)
            ref_pcd = ref_mesh_o3d.sample_points_uniformly(number_of_points=num_samples)
            
            # Estimate normals for better alignment
            scan_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
            ref_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
            
            # Initial alignment: Center both at origin and try multiple rotations
            print("   🔍 Finding initial alignment...")
            scan_pcd_centered = scan_pcd.translate(-scan_pcd.get_center())
            ref_pcd_centered = ref_pcd.translate(-ref_pcd.get_center())
            
            # Use RANSAC-based global registration for better initial alignment
            # This handles arbitrary rotations much better than testing specific angles
            threshold = voxel_size * 2
            
            # Compute FPFH features for better matching
            print("   🔍 Computing features for global registration...")
            
            # Downsample for feature computation
            scan_down = scan_pcd_centered.voxel_down_sample(voxel_size)
            ref_down = ref_pcd_centered.voxel_down_sample(voxel_size)
            
            # Estimate normals if not already done
            scan_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
            ref_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
            
            # Compute FPFH features
            scan_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                scan_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )
            ref_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                ref_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
            )
            
            # RANSAC-based global registration
            print("   🎲 Running RANSAC global registration...")
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                scan_down, ref_down, scan_fpfh, ref_fpfh,
                mutual_filter=True,
                max_correspondence_distance=threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            
            best_transformation = result_ransac.transformation
            best_fitness = result_ransac.fitness
            
            print(f"   ✅ RANSAC alignment found (fitness: {best_fitness:.4f})")
            
            # Refine with Point-to-Plane ICP for higher accuracy
            print("   🎯 Refining alignment with Point-to-Plane ICP...")
            
            result_icp = o3d.pipelines.registration.registration_icp(
                scan_pcd_centered, 
                ref_pcd_centered,
                threshold,
                best_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            
            # Store ICP quality metrics
            self.icp_fitness = result_icp.fitness
            self.icp_rmse = result_icp.inlier_rmse
            
            # Apply the transformation to the original trimesh object
            # Need to account for the centering we did
            scan_center = self.trimesh_object.centroid
            ref_center = self.reference_mesh.centroid
            
            # Build complete transformation:
            # 1. Translate scan to origin
            # 2. Apply ICP transformation (rotation + refinement)
            # 3. Translate to reference center
            
            # Create transformation matrices
            to_origin = np.eye(4)
            to_origin[:3, 3] = -scan_center
            
            to_ref_center = np.eye(4)
            to_ref_center[:3, 3] = ref_center
            
            # Combine: translate to ref center, apply ICP, translate to origin
            final_transform = to_ref_center @ result_icp.transformation @ to_origin
            
            # Apply complete transformation to trimesh
            self.trimesh_object.apply_transform(final_transform)
            
            # Update plane position
            center = self.trimesh_object.bounds.mean(axis=0)
            self.plane_position = center
            
            print(f"\n✅ ICP Alignment Complete!")
            print(f"   📊 Fitness Score: {self.icp_fitness:.4f} (1.0 = perfect)")
            print(f"   📊 RMSE: {self.icp_rmse:.6f} mm")
            print(f"   📍 New center: [{self.plane_position[0]:.1f}, {self.plane_position[1]:.1f}, {self.plane_position[2]:.1f}]")
            
            # Compare bounds before/after to verify alignment
            ref_bounds = self.reference_mesh.bounds
            scan_bounds = self.trimesh_object.bounds
            bounds_diff = np.abs(scan_bounds - ref_bounds).max()
            print(f"   📐 Max bounds difference: {bounds_diff:.3f} mm")
            
            if self.icp_fitness < 0.5:
                print(f"   ⚠️  Warning: Low fitness score - scan may not match reference well")
            elif self.icp_fitness > 0.9:
                print(f"   🎉 Excellent alignment quality!")
            
            if bounds_diff > 5.0:
                print(f"   ⚠️  Warning: Large bounds difference - alignment may need adjustment")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during ICP alignment: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_plane_transformation(self):
        """Apply rotation to get plane normal from Euler angles"""
        # Convert Euler angles to rotation matrix
        rx, ry, rz = self.plane_rotation
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Apply rotation to default normal (Z-axis)
        self.plane_normal = R @ np.array([0, 0, 1])
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
    
    def calculate_mesh_plane_intersection(self):
        """Calculate exact intersection of mesh with cutting plane using geometric methods"""
        if self.trimesh_object is None:
            print("❌ No trimesh object loaded. Use load_stl_mesh_direct() first.")
            return None
        
        # Apply plane transformation to get correct normal
        self._apply_plane_transformation()
        
        try:
            # Use trimesh's exact plane-mesh intersection
            lines = trimesh.intersections.mesh_plane(
                mesh=self.trimesh_object,
                plane_normal=self.plane_normal,
                plane_origin=self.plane_position
            )
            
            if len(lines) == 0:
                print("⚠️ No intersection lines found")
                return None
            
            # Extract all intersection points from line segments
            intersection_points = []
            for line in lines:
                intersection_points.extend([line[0], line[1]])
            
            intersection_points = np.array(intersection_points)
            
            # Remove duplicate points (within tolerance)
            if len(intersection_points) > 0:
                clustering = DBSCAN(eps=0.001, min_samples=1).fit(intersection_points)
                unique_labels = np.unique(clustering.labels_)
                
                # Get centroid of each cluster to remove duplicates
                unique_points = []
                for label in unique_labels:
                    cluster_points = intersection_points[clustering.labels_ == label]
                    centroid = np.mean(cluster_points, axis=0)
                    unique_points.append(centroid)
                
                intersection_points = np.array(unique_points)
            
            print(f"✅ Found {len(lines):,} intersection line segments")
            print(f"✅ Extracted {len(intersection_points):,} unique intersection points")
            
            # Calculate measurements using the exact intersection points
            measurements = self._calculate_intersection_measurements(intersection_points)
            
            # Store intersection points for visualization
            self.intersection_points = intersection_points
            
            print(f"📏 Exact geometric cross-section measurements:")
            print(f"   Width (full): {measurements['width']:.6f} mm")
            print(f"   Width (90% percentile): {measurements['width_percentile_90']:.6f} mm")
            print(f"   Height (full): {measurements['height']:.6f} mm")
            print(f"   Height (90% percentile): {measurements['height_percentile_90']:.6f} mm")
            print(f"   Area: {measurements['area']:.3f} mm²")
            print(f"   Perimeter: {measurements['perimeter']:.3f} mm")
            print(f"   Point density: {measurements['point_density']:.1f} pts/mm²")
            print(f"   Quality: {measurements['measurement_quality']}")
            
            return measurements
            
        except Exception as e:
            print(f"❌ Error in mesh-plane intersection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_intersection_measurements(self, intersection_points):
        """Calculate detailed measurements of the intersection with enhanced precision"""
        if len(intersection_points) == 0:
            return None
        
        # Project intersection points onto the plane
        # Create two orthogonal vectors in the plane
        if abs(self.plane_normal[2]) < 0.9:  # Normal not along Z
            u_axis = np.cross(self.plane_normal, [0, 0, 1])
        else:  # Normal along Z, use Y
            u_axis = np.cross(self.plane_normal, [0, 1, 0])
        u_axis = u_axis / np.linalg.norm(u_axis)
        
        v_axis = np.cross(self.plane_normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
        
        # Project points onto the plane coordinate system
        relative_points = intersection_points - self.plane_position
        u_coords = np.dot(relative_points, u_axis)
        v_coords = np.dot(relative_points, v_axis)
        
        # Calculate bounding box in plane coordinates
        u_min, u_max = np.min(u_coords), np.max(u_coords)
        v_min, v_max = np.min(v_coords), np.max(v_coords)
        
        width = u_max - u_min
        height = v_max - v_min
        
        # Enhanced precision measurements
        # Use percentiles to exclude outliers
        u_percentile_5 = np.percentile(u_coords, 5)
        u_percentile_95 = np.percentile(u_coords, 95)
        v_percentile_5 = np.percentile(v_coords, 5)
        v_percentile_95 = np.percentile(v_coords, 95)
        
        width_percentile_90 = u_percentile_95 - u_percentile_5
        height_percentile_90 = v_percentile_95 - v_percentile_5
        
        # Calculate centroid for more precise center
        centroid_u = np.mean(u_coords)
        centroid_v = np.mean(v_coords)
        
        # Calculate standard deviation for spread analysis
        std_u = np.std(u_coords)
        std_v = np.std(v_coords)
        
        # Calculate area using convex hull for more accurate area
        try:
            if len(u_coords) > 3:
                points_2d = np.column_stack([u_coords, v_coords])
                hull = ConvexHull(points_2d)
                area = hull.volume  # In 2D, volume is area
                perimeter = 0
                for i in range(len(hull.vertices)):
                    j = (i + 1) % len(hull.vertices)
                    p1 = points_2d[hull.vertices[i]]
                    p2 = points_2d[hull.vertices[j]]
                    perimeter += np.linalg.norm(p2 - p1)
            else:
                area = 0
                perimeter = 0
        except:
            area = width * height  # Fallback to bounding box area
            perimeter = 2 * (width + height)
        
        # Point density
        point_density = len(intersection_points) / max(area, 1e-6)
        
        # Quality assessment based on point count and distribution
        if len(intersection_points) > 200:
            quality = "high"
        elif len(intersection_points) > 100:
            quality = "medium"
        else:
            quality = "low"
        
        return {
            'width': width,
            'height': height,
            'width_percentile_90': width_percentile_90,
            'height_percentile_90': height_percentile_90,
            'area': area,
            'perimeter': perimeter,
            'centroid_u': centroid_u,
            'centroid_v': centroid_v,
            'std_u': std_u,
            'std_v': std_v,
            'point_count': len(intersection_points),
            'point_density': point_density,
            'measurement_quality': quality,
            'u_coords': u_coords,
            'v_coords': v_coords
        }


class AutomatedDirectMeasurer:
    """Automated measurement system using direct mesh intersection"""
    
    def __init__(self, 
                 stl_path="/Users/hemal/Desktop/point-cloud-project/479.stl",
                 reference_stl_path=None,
                 use_icp_alignment=False):
        """
        Initialize automated measurer
        
        Args:
            stl_path: Path to STL file to measure (can be a scan)
            reference_stl_path: Path to reference CAD model for ICP alignment (optional)
            use_icp_alignment: If True and reference provided, align scan to reference
        """
        self.stl_path = stl_path
        self.reference_stl_path = reference_stl_path
        self.use_icp_alignment = use_icp_alignment
        self.measurer = DirectMeshMeasurer()
        self.results = []
        
        # Load STL once for all measurements
        print("🔄 Loading STL mesh for direct intersection measurements...")
        print("   Using DIRECT MESH INTERSECTION (slower but perfectly accurate)")
        self.measurer.load_stl_mesh_direct(stl_path)
        print("✅ STL loaded successfully")
        
        # Apply ICP alignment if requested
        if use_icp_alignment and reference_stl_path:
            print("\n🎯 ICP ALIGNMENT ENABLED")
            print(f"   Reference: {reference_stl_path}")
            print(f"   Scan: {stl_path}")
            
            self.measurer.load_reference_model(reference_stl_path)
            success = self.measurer.align_to_reference_with_icp()
            
            if not success:
                print("⚠️  ICP alignment failed, proceeding with original orientation")
            print()
        elif use_icp_alignment and not reference_stl_path:
            print("⚠️  ICP alignment requested but no reference model provided")
            print("   Proceeding without alignment\n")
        else:
            print("   No ICP alignment requested\n")
    
    def _setup_plane(self, position, rotation):
        """Configure plane at specific position and rotation"""
        self.measurer.plane_position = np.array(position)
        self.measurer.plane_rotation = np.array(rotation)
    
    def _measure_at_plane(self):
        """Perform measurement at current plane configuration"""
        measurements = self.measurer.calculate_mesh_plane_intersection()
        return measurements
    
    def _check_tolerance(self, measurement_name, measured_value, min_val, max_val, dimension_name):
        """Check if measured value is within tolerance"""
        passed = min_val <= measured_value <= max_val
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"   {dimension_name}: {measured_value:.6f} mm (expected {min_val:.2f} to {max_val:.2f}) {status}")
        
        return passed
    
    def measure_108_1(self):
        """
        Measurement 108: Width 38.5-39.8 mm, Height 10-10.5 mm
        Location: Front section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 108_1: Width 38.5-39.8 mm, Height 10-10.5 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [-45.5012, 0.0174, 7.6730]
        rotation = [1.570796, 0.000000, 1.570796]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "108_1", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width and height measurements
        width = measurements['width']
        height = measurements['height']
        
        # Check tolerances
        width_pass = self._check_tolerance("108_1", width, 38.5, 39.8, "Width (full)")
        height_pass = self._check_tolerance("108_1", height, 10.0, 10.5, "Height (full)")
        
        overall_pass = width_pass and height_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "108_1",
            "passed": overall_pass,
            "width": width,
            "height": height,
            "width_pass": width_pass,
            "height_pass": height_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_108_2(self):
        """
        Measurement 108: Width 38.5-39.8 mm, Height 10-10.5 mm
        Location: Front section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 108_2: Width 38.5-39.8 mm, Height 10-10.5 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [45.5012, 0.0174, 7.6730]
        rotation = [1.570796, 0.000000, 1.570796]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "108_2", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width and height measurements
        width = measurements['width']
        height = measurements['height']
        
        # Check tolerances
        width_pass = self._check_tolerance("108_2", width, 38.5, 39.8, "Width (full)")
        height_pass = self._check_tolerance("108_2", height, 10.0, 10.5, "Height (full)")
        
        overall_pass = width_pass and height_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "108_2",
            "passed": overall_pass,
            "width": width,
            "height": height,
            "width_pass": width_pass,
            "height_pass": height_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_108_3(self):
        """
        Measurement 108: Width 38.5-39.8 mm, Height 10-10.5 mm
        Location: Front section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 108_3: Width 38.5-39.8 mm, Height 10-10.5 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [37.9988, 45.0177, 8.1734]
        rotation = [1.614430, 0.000000, 3.141593]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "108_3", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width and height measurements
        width = measurements['width']
        height = measurements['height']
        
        # Check tolerances
        width_pass = self._check_tolerance("108_3", width, 38.5, 39.8, "Width (full)")
        height_pass = self._check_tolerance("108_3", height, 10.0, 10.5, "Height (full)")
        
        overall_pass = width_pass and height_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "108_3",
            "passed": overall_pass,
            "width": width,
            "height": height,
            "width_pass": width_pass,
            "height_pass": height_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_108_4(self):
        """
        Measurement 108: Width 38.5-39.8 mm, Height 10-10.5 mm
        Location: Front section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 108_4: Width 38.5-39.8 mm, Height 10-10.5 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [37.9988, -45.0177, 8.1734]
        rotation = [1.614430, 0.000000, 3.141593]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "108_4", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width and height measurements
        width = measurements['width']
        height = measurements['height']
        
        # Check tolerances
        width_pass = self._check_tolerance("108_4", width, 38.5, 39.8, "Width (full)")
        height_pass = self._check_tolerance("108_4", height, 10.0, 10.5, "Height (full)")
        
        overall_pass = width_pass and height_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "108_4",
            "passed": overall_pass,
            "width": width,
            "height": height,
            "width_pass": width_pass,
            "height_pass": height_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_109(self):
        """
        Measurement 109: Width 71-72 mm
        Location: Side section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 109: Width 71-72 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [-0.0012, 0.0174, 7.6730]
        rotation = [1.570796, 0.000000, 2.356194]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "109", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width measurement
        width = measurements['width']
        
        # Check tolerance
        width_pass = self._check_tolerance("109", width, 71.0, 72.0, "Width (full)")
        
        overall_pass = width_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "109",
            "passed": overall_pass,
            "width": width,
            "width_pass": width_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_19(self):
        """
        Measurement 19: Width 36.5-36.64 mm
        Location: Top section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 19: Width 36.5-36.64 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [0.4988, 0.0174, 11.6730]
        rotation = [3.141593, 0.000000, 2.356194]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "19", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width measurement
        width = measurements['width']
        
        # Check tolerance
        width_pass = self._check_tolerance("19", width, 36.5, 36.64, "Width (full)")
        
        overall_pass = width_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "19",
            "passed": overall_pass,
            "width": width,
            "width_pass": width_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_18(self):
        """
        Measurement 18: Width 25.36-25.5 mm
        Location: Bottom section
        """
        print("=" * 70)
        print("📏 MEASUREMENT 18: Width 25.36-25.5 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [0.9988, 0.0174, -21.8270]
        rotation = [3.141593, 0.000000, 2.356194]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "18", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width measurement
        width = measurements['width']
        
        # Check tolerance
        width_pass = self._check_tolerance("18", width, 25.36, 25.5, "Width (full)")
        
        overall_pass = width_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "18",
            "passed": overall_pass,
            "width": width,
            "width_pass": width_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    
    def measure_width_height(self):
        """
        Width/Height Measurement: Width 129.9-130.1 mm, Height 94.79-95.09 mm
        Location: Main body cross-section
        """
        print("=" * 70)
        print("📏 WIDTH/HEIGHT MEASUREMENT: Width 129.9-130.1 mm, Height 94.79-95.09 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [-0.0017, 0.0177, 7.6714]
        rotation = [1.570796, -0.000000, 6.283185]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "width_height", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width and height measurements
        width = measurements['width']
        height = measurements['height']
        
        # Check tolerances
        width_pass = self._check_tolerance("width_height", width, 129.9, 130.1, "Width (full)")
        height_pass = self._check_tolerance("width_height", height, 94.79, 95.09, "Height (full)")
        
        overall_pass = width_pass and height_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "width_height",
            "passed": overall_pass,
            "width": width,
            "height": height,
            "width_pass": width_pass,
            "height_pass": height_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    def measure_width_height_2(self):
        """
        Width/Height Measurement: Width 129.9-130.1 mm, Height 94.79-95.09 mm
        Location: Main body cross-section
        """
        print("=" * 70)
        print("📏 WIDTH/HEIGHT MEASUREMENT: Width 129.9-130.1 mm, Height 94.79-95.09 mm")
        print("=" * 70)
        
        # Setup plane configuration
        position = [-0.0017, 0.0177, 7.6714]
        rotation = [1.570796, -0.000000, 7.853982]
        
        self._setup_plane(position, rotation)
        
        # Perform measurement
        measurements = self._measure_at_plane()
        
        if measurements is None:
            print("❌ MEASUREMENT FAILED - No intersection points found\n")
            self.results.append({"name": "width_height", "passed": False, "reason": "No intersection"})
            return False
        
        # Use full width and height measurements
        width = measurements['width']
        height = measurements['height']
        
        # Check tolerances
        width_pass = self._check_tolerance("width_height", width, 129.9, 130.1, "Width (full)")
        height_pass = self._check_tolerance("width_height", height, 94.79, 95.09, "Height (full)")
        
        overall_pass = width_pass and height_pass
        status = "✅ PASSED" if overall_pass else "❌ FAILED"
        
        print(f"\n🎯 Overall Status: {status}\n")
        
        self.results.append({
            "name": "width_height",
            "passed": overall_pass,
            "width": width,
            "height": height,
            "width_pass": width_pass,
            "height_pass": height_pass,
            "point_count": measurements['point_count']
        })
        
        return overall_pass
    def run_all_measurements(self):
        """Run all preset measurements and generate summary report"""
        print("\n")
        print("🚀 " + "=" * 66 + " 🚀")
        print("   DIRECT MESH INTERSECTION MEASUREMENT SYSTEM - 479.STL")
        print("🚀 " + "=" * 66 + " 🚀")
        print("\n")
        
        # Run all measurements
        measurements = [
            ("108_1", self.measure_108_1),
            ("108_2", self.measure_108_2),
            ("108_3", self.measure_108_3),
            ("108_4", self.measure_108_4),  
            ("109", self.measure_109),
            ("19", self.measure_19),
            ("18", self.measure_18),
            ("width_height", self.measure_width_height),
            ("width_height_2", self.measure_width_height_2)
        ]
        
        for name, func in measurements:
            try:
                func()
            except Exception as e:
                print(f"❌ ERROR in measurement {name}: {e}\n")
                import traceback
                traceback.print_exc()
                self.results.append({"name": name, "passed": False, "reason": str(e)})
        
        # Generate summary report
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of all measurements"""
        print("\n")
        print("📊 " + "=" * 66 + " 📊")
        print("   DIRECT MESH INTERSECTION SUMMARY REPORT")
        print("📊 " + "=" * 66 + " 📊")
        print("\n")
        
        # Show ICP alignment quality if used
        if self.use_icp_alignment and self.measurer.icp_fitness is not None:
            print("🎯 ICP Alignment Quality:")
            print(f"   Fitness Score: {self.measurer.icp_fitness:.4f} (1.0 = perfect)")
            print(f"   RMSE: {self.measurer.icp_rmse:.6f} mm")
            
            if self.measurer.icp_fitness > 0.9:
                print("   Status: ✅ Excellent alignment")
            elif self.measurer.icp_fitness > 0.7:
                print("   Status: ✅ Good alignment")
            elif self.measurer.icp_fitness > 0.5:
                print("   Status: ⚠️  Fair alignment")
            else:
                print("   Status: ❌ Poor alignment - results may be unreliable")
            print()
        
        passed_count = sum(1 for r in self.results if r.get("passed", False))
        total_count = len(self.results)
        
        for result in self.results:
            name = result["name"]
            passed = result.get("passed", False)
            status = "✅ PASS" if passed else "❌ FAIL"
            point_count = result.get("point_count", "N/A")
            
            print(f"  Measurement {name.ljust(15)}: {status} ({point_count} intersection points)")
            
            if not passed and "reason" in result:
                print(f"    Reason: {result['reason']}")
        
        print("\n" + "-" * 70)
        print(f"  Total: {passed_count}/{total_count} measurements passed")
        
        if passed_count == total_count:
            print("\n  🎉 ALL MEASUREMENTS PASSED! 🎉")
            print("  ✨ Perfect deterministic results with direct mesh intersection ✨")
        else:
            print(f"\n  ⚠️  {total_count - passed_count} measurement(s) failed")
        
        print("-" * 70 + "\n")
        
        return passed_count == total_count


def main():
    """
    Main function to run direct mesh intersection measurements
    
    Usage:
        python measure_479_dm.py [scan_stl] [--reference ref_stl] [--icp]
    
    Examples:
        # Measure without ICP alignment
        python measure_479_dm.py 479.stl
        
        # Measure with ICP alignment to reference
        python measure_479_dm.py upside_down_479.stl --reference 479.stl --icp
        
        # Short form
        python measure_479_dm.py upside_down_479.stl --reference 479.stl
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Direct Mesh Intersection Measurement with Optional ICP Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure CAD model directly
  python measure_479_dm.py 479.stl
  
  # Measure scanned part with ICP alignment
  python measure_479_dm.py scanned_479.stl --reference 479.stl --icp
  
  # ICP is auto-enabled when reference is provided
  python measure_479_dm.py upside_down_479.stl --reference 479.stl
        """
    )
    
    parser.add_argument(
        'stl_file',
        nargs='?',
        default='/Users/hemal/Desktop/point-cloud-project/479.stl',
        help='STL file to measure (default: 479.stl)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        help='Reference CAD model for ICP alignment (enables ICP automatically)'
    )
    
    parser.add_argument(
        '--icp',
        action='store_true',
        help='Enable ICP alignment (requires --reference)'
    )
    
    parser.add_argument(
        '--no-icp',
        action='store_true',
        help='Disable ICP alignment even if reference is provided'
    )
    
    args = parser.parse_args()
    
    # Determine if ICP should be used
    use_icp = False
    if args.reference:
        # If reference provided, enable ICP by default unless --no-icp
        use_icp = not args.no_icp
    elif args.icp:
        print("⚠️  Warning: --icp flag provided but no --reference specified")
        print("   ICP alignment will be disabled\n")
    
    try:
        # Create direct mesh measurer
        print(f"📄 Input file: {args.stl_file}")
        if args.reference:
            print(f"📄 Reference file: {args.reference}")
            print(f"🎯 ICP Alignment: {'Enabled' if use_icp else 'Disabled'}")
        print()
        
        measurer = AutomatedDirectMeasurer(
            stl_path=args.stl_file,
            reference_stl_path=args.reference,
            use_icp_alignment=use_icp
        )
        
        # Run all measurements
        all_passed = measurer.run_all_measurements()
        
        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
