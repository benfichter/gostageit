"""
Furniture Rendering Module - Places 3D models into room images
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
# Force EGL backend for headless rendering environments (e.g., Runpod)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import pyrender


class FurnitureModelLibrary:
    """Manages 3D furniture models and their properties"""
    
    # Simple primitive models for demonstration
    # In production, load actual 3D models from files
    
    @staticmethod
    def create_box_mesh(dimensions: Dict[str, float], color=(0.6, 0.4, 0.2)) -> trimesh.Trimesh:
        """Create a simple box mesh for furniture"""
        box = trimesh.creation.box(extents=[
            dimensions['width'],
            dimensions['height'], 
            dimensions['depth']
        ])
        box.visual.vertex_colors = np.array([color + (1.0,)] * len(box.vertices))
        return box
    
    @staticmethod
    def get_furniture_model(furniture_type: str, dimensions: Dict[str, float]) -> trimesh.Trimesh:
        """Get or create a 3D model for furniture type"""
        
        # Color schemes for different furniture types
        colors = {
            'sofa': (0.3, 0.3, 0.5),  # Blue-gray
            'loveseat': (0.3, 0.3, 0.5),
            'armchair': (0.4, 0.3, 0.3),  # Brown
            'coffee_table': (0.5, 0.3, 0.2),  # Wood brown
            'side_table': (0.5, 0.3, 0.2),
            'tv_console': (0.2, 0.2, 0.2),  # Dark gray
            'dining_table': (0.6, 0.4, 0.3),  # Light wood
            'rug': (0.7, 0.6, 0.5),  # Beige
            'bed': (0.8, 0.8, 0.9),  # Light blue
            'bookshelf': (0.4, 0.25, 0.15),  # Dark wood
        }
        
        color = colors.get(furniture_type, (0.5, 0.5, 0.5))
        
        # For now, create simple box meshes
        # In production, load actual models from .obj/.gltf files
        if furniture_type == 'sofa':
            # Create sofa with back
            base = trimesh.creation.box(extents=[
                dimensions['width'],
                dimensions['height'] * 0.5,
                dimensions['depth']
            ])
            back = trimesh.creation.box(extents=[
                dimensions['width'],
                dimensions['height'] * 0.5,
                dimensions['depth'] * 0.3
            ])
            back.apply_translation([0, dimensions['height'] * 0.25, -dimensions['depth'] * 0.35])
            
            sofa = trimesh.util.concatenate([base, back])
            sofa.visual.vertex_colors = np.array([color + (1.0,)] * len(sofa.vertices))
            return sofa
            
        elif furniture_type == 'coffee_table':
            # Create table with legs
            top = trimesh.creation.box(extents=[
                dimensions['width'],
                0.05,
                dimensions['depth']
            ])
            top.apply_translation([0, dimensions['height'] - 0.025, 0])
            
            leg_height = dimensions['height'] - 0.05
            leg_size = 0.05
            legs = []
            for x in [-dimensions['width']/2 + leg_size, dimensions['width']/2 - leg_size]:
                for z in [-dimensions['depth']/2 + leg_size, dimensions['depth']/2 - leg_size]:
                    leg = trimesh.creation.box(extents=[leg_size, leg_height, leg_size])
                    leg.apply_translation([x, leg_height/2, z])
                    legs.append(leg)
            
            table = trimesh.util.concatenate([top] + legs)
            table.visual.vertex_colors = np.array([color + (1.0,)] * len(table.vertices))
            return table
            
        elif furniture_type == 'rug':
            # Flat rectangular rug
            rug = trimesh.creation.box(extents=[
                dimensions['width'],
                0.01,  # Very thin
                dimensions['depth']
            ])
            rug.visual.vertex_colors = np.array([color + (1.0,)] * len(rug.vertices))
            return rug
            
        else:
            # Default box for other furniture
            return FurnitureModelLibrary.create_box_mesh(dimensions, color)


class FurnitureRenderer:
    """Renders 3D furniture models into room images"""
    
    def __init__(self):
        self.model_library = FurnitureModelLibrary()
        
    def render_furniture_in_room(
        self, 
        room_image: np.ndarray,
        placements: List[Dict],
        room_analysis: Dict,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Render furniture models into the room image
        
        Args:
            room_image: Original room image (RGB)
            placements: List of furniture placements from placement engine
            room_analysis: Room analysis from MoGe
            output_path: Optional path to save rendered image
            
        Returns:
            Rendered image with furniture
        """
        
        # Set up the scene
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        
        # Create camera from MoGe intrinsics
        camera = self._create_camera_from_intrinsics(
            room_analysis.get('intrinsics'),
            room_image.shape
        )
        # PyRender expects OpenGL camera coords (Z points backward, Y up)
        # MoGe/placement pipeline uses computer-vision coords (Z forward, Y down),
        # so flip Y/Z axes on the camera pose to make the furniture visible.
        camera_pose = np.eye(4)
        camera_pose[1, 1] = -1
        camera_pose[2, 2] = -1
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = self._look_at(
            eye=[0, 2, -2],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
        scene.add(light, pose=light_pose)
        
        # Add furniture to scene
        for placement in placements:
            mesh = self.model_library.get_furniture_model(
                placement['type'],
                placement['dimensions']
            )
            
            # Apply transformation
            transform = self._create_transform_matrix(
                placement['position'],
                placement['rotation']
            )
            
            # Convert trimesh to pyrender mesh
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh_pyrender, pose=transform)
        
        # Render the scene
        renderer = pyrender.OffscreenRenderer(
            room_image.shape[1], 
            room_image.shape[0]
        )
        
        color, depth = renderer.render(scene)
        renderer.delete()
        
        # Composite rendered furniture with original image
        composite = self._composite_images(
            room_image,
            color,
            depth,
            room_analysis.get('depth')
        )
        
        # Save if path provided
        if output_path:
            Image.fromarray(composite).save(output_path)
        
        return composite
    
    def render_placement_visualization(
        self,
        room_image: np.ndarray,
        placements: List[Dict],
        room_analysis: Dict,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create a top-down visualization of furniture placement
        """
        bounds = room_analysis['floor_bounds']
        
        # Calculate image dimensions for top-down view
        room_width = bounds['x_max'] - bounds['x_min']
        room_depth = bounds['z_max'] - bounds['z_min']
        
        pixels_per_meter = 100
        img_width = int(room_width * pixels_per_meter) + 100
        img_height = int(room_depth * pixels_per_meter) + 100
        
        # Create blank canvas
        canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240
        
        # Draw room boundaries
        room_rect = [
            50, 50,
            img_width - 100,
            img_height - 100
        ]
        cv2.rectangle(canvas, 
                     (room_rect[0], room_rect[1]),
                     (room_rect[0] + room_rect[2], room_rect[1] + room_rect[3]),
                     (100, 100, 100), 2)
        
        # Draw grid lines
        for i in range(0, img_width, pixels_per_meter):
            cv2.line(canvas, (i, 0), (i, img_height), (220, 220, 220), 1)
        for i in range(0, img_height, pixels_per_meter):
            cv2.line(canvas, (0, i), (img_width, i), (220, 220, 220), 1)
        
        # Color scheme for furniture
        colors = {
            'sofa': (150, 150, 200),
            'loveseat': (150, 150, 200),
            'armchair': (180, 150, 150),
            'coffee_table': (160, 120, 80),
            'side_table': (160, 120, 80),
            'tv_console': (80, 80, 80),
            'dining_table': (180, 140, 100),
            'rug': (200, 180, 160),
            'bed': (180, 180, 220),
            'bookshelf': (120, 80, 60),
        }
        
        # Draw furniture
        for placement in placements:
            # Convert 3D position to 2D image coordinates
            x = placement['position'][0]
            z = placement['position'][2]
            
            # Convert to pixel coordinates
            px = int((x - bounds['x_min']) * pixels_per_meter) + 50
            py = int((z - bounds['z_min']) * pixels_per_meter) + 50
            
            # Get dimensions
            width = placement['dimensions']['width'] * pixels_per_meter
            depth = placement['dimensions']['depth'] * pixels_per_meter
            
            # Apply rotation
            rotation_deg = np.degrees(placement['rotation'])
            
            # Create rotated rectangle
            rect_center = (px, py)
            rect_size = (int(width), int(depth))
            rect_angle = rotation_deg
            
            # Get rectangle points
            box = cv2.boxPoints((rect_center, rect_size, rect_angle))
            # numpy removed np.int0 alias; cast explicitly to platform int type
            box = box.astype(np.intp)
            
            # Draw filled rectangle
            color = colors.get(placement['type'], (150, 150, 150))
            cv2.fillPoly(canvas, [box], color)
            cv2.polylines(canvas, [box], True, (50, 50, 50), 2)
            
            # Add label
            cv2.putText(canvas, placement['type'],
                       (px - 30, py + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add title and dimensions
        cv2.putText(canvas, "Furniture Placement (Top View)",
                   (img_width // 2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.putText(canvas, f"Room: {room_width:.1f}m x {room_depth:.1f}m",
                   (img_width // 2 - 100, img_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(str(output_path), canvas)
        
        return canvas
    
    def _create_camera_from_intrinsics(self, intrinsics: Optional[np.ndarray], 
                                       image_shape: Tuple[int, int]) -> pyrender.Camera:
        """Create pyrender camera from intrinsics matrix"""
        if intrinsics is None:
            # Estimate from image dimensions
            h, w = image_shape[:2]
            focal = max(h, w)
            intrinsics = np.array([
                [focal, 0, w/2],
                [0, focal, h/2],
                [0, 0, 1]
            ])
        
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # Create camera
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy
        )
        
        return camera
    
    def _create_transform_matrix(self, position: np.ndarray, rotation: float) -> np.ndarray:
        """Create 4x4 transformation matrix from position and rotation"""
        transform = np.eye(4)
        
        # Apply rotation (around Y axis)
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        transform[0, 0] = cos_r
        transform[0, 2] = sin_r
        transform[2, 0] = -sin_r
        transform[2, 2] = cos_r
        
        # Apply translation
        transform[0, 3] = position[0]
        transform[1, 3] = position[1]
        transform[2, 3] = position[2]
        
        return transform
    
    def _look_at(self, eye, target, up):
        """Create look-at matrix for camera/light positioning"""
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)
        
        z = eye - target
        z = z / np.linalg.norm(z)
        
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        
        y = np.cross(z, x)
        
        mat = np.eye(4)
        mat[:3, 0] = x
        mat[:3, 1] = y
        mat[:3, 2] = z
        mat[:3, 3] = eye
        
        return mat
    
    def _composite_images(self, background: np.ndarray, foreground: np.ndarray,
                         depth_fg: np.ndarray, depth_bg: Optional[np.ndarray]) -> np.ndarray:
        """Composite rendered furniture with original room image"""
        
        # Create mask where furniture is visible
        mask = depth_fg > 0
        
        # Simple alpha blending
        result = background.copy()
        result[mask] = foreground[mask]
        
        # Optional: Add shadows or more sophisticated blending
        # This would require more advanced rendering techniques
        
        return result


class SimpleRenderer:
    """Simplified 2D renderer for systems without pyrender"""
    
    @staticmethod
    def render_2d_visualization(
        room_image: np.ndarray,
        placements: List[Dict],
        room_analysis: Dict,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create a simple 2D visualization overlaying furniture boxes on the image
        """
        overlay = room_image.copy()
        
        # Project 3D positions to 2D image coordinates
        for placement in placements:
            # Get 3D bounding box corners
            corners_3d = SimpleRenderer._get_3d_box_corners(
                placement['position'],
                placement['dimensions'],
                placement['rotation']
            )
            
            # Project to 2D
            corners_2d = SimpleRenderer._project_to_2d(
                corners_3d,
                room_analysis.get('intrinsics'),
                room_image.shape
            )
            
            # Draw the box
            SimpleRenderer._draw_3d_box(overlay, corners_2d, placement['type'])
        
        # Add legend
        SimpleRenderer._add_legend(overlay, placements)
        
        if output_path:
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        return overlay
    
    @staticmethod
    def _get_3d_box_corners(position: np.ndarray, dimensions: Dict[str, float], 
                           rotation: float) -> np.ndarray:
        """Get 8 corners of 3D bounding box"""
        w, h, d = dimensions['width'], dimensions['height'], dimensions['depth']
        
        # Create corners in local coordinates
        corners = np.array([
            [-w/2, -h/2, -d/2],
            [w/2, -h/2, -d/2],
            [w/2, -h/2, d/2],
            [-w/2, -h/2, d/2],
            [-w/2, h/2, -d/2],
            [w/2, h/2, -d/2],
            [w/2, h/2, d/2],
            [-w/2, h/2, d/2],
        ])
        
        # Apply rotation around Y axis
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, 0, sin_r],
            [0, 1, 0],
            [-sin_r, 0, cos_r]
        ])
        
        corners = corners @ rotation_matrix.T
        
        # Apply translation
        corners = corners + position
        
        return corners
    
    @staticmethod
    def _project_to_2d(points_3d: np.ndarray, intrinsics: Optional[np.ndarray],
                      image_shape: Tuple[int, int]) -> np.ndarray:
        """Project 3D points to 2D image coordinates"""
        if intrinsics is None:
            # Simple perspective projection
            h, w = image_shape[:2]
            focal = max(h, w)
            intrinsics = np.array([
                [focal, 0, w/2],
                [0, focal, h/2],
                [0, 0, 1]
            ])
        
        # Project points
        points_2d = []
        for point in points_3d:
            if point[2] > 0:  # In front of camera
                x = point[0] / point[2] * intrinsics[0, 0] + intrinsics[0, 2]
                y = point[1] / point[2] * intrinsics[1, 1] + intrinsics[1, 2]
                points_2d.append([int(x), int(y)])
            else:
                points_2d.append([0, 0])  # Behind camera
        
        return np.array(points_2d)
    
    @staticmethod
    def _draw_3d_box(image: np.ndarray, corners_2d: np.ndarray, furniture_type: str):
        """Draw 3D bounding box on image"""
        # Color scheme
        colors = {
            'sofa': (100, 100, 200),
            'loveseat': (100, 100, 200),
            'armchair': (150, 100, 100),
            'coffee_table': (139, 90, 43),
            'side_table': (139, 90, 43),
            'tv_console': (50, 50, 50),
            'dining_table': (160, 120, 80),
            'rug': (200, 180, 160),
        }
        color = colors.get(furniture_type, (128, 128, 128))
        
        # Define edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
        ]
        
        # Draw edges
        for edge in edges:
            pt1 = tuple(corners_2d[edge[0]])
            pt2 = tuple(corners_2d[edge[1]])
            cv2.line(image, pt1, pt2, color, 2)
        
        # Add label
        center = np.mean(corners_2d[:4], axis=0).astype(int)
        cv2.putText(image, furniture_type,
                   tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    @staticmethod
    def _add_legend(image: np.ndarray, placements: List[Dict]):
        """Add legend showing placed furniture"""
        y_offset = 30
        for i, placement in enumerate(placements):
            text = f"{placement['type']}: {placement['dimensions']['width']:.1f}m x {placement['dimensions']['depth']:.1f}m"
            cv2.putText(image, text,
                       (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
