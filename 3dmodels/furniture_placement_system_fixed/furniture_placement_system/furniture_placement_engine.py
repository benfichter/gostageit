#!/usr/bin/env python3
"""
Furniture Placement Engine using Rule-Based System with Physics Simulation
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import trimesh
from scipy.spatial.distance import cdist


@dataclass
class FurnitureItem:
    """Represents a furniture item to be placed"""
    type: str
    dimensions: Dict[str, float]  # width, depth, height in meters
    model_path: Optional[str] = None
    priority: int = 1
    placement_rules: Optional[Dict] = None


class InteriorDesignRules:
    """Encapsulates interior design principles and ergonomic standards"""
    
    ERGONOMIC_DIMENSIONS = {
        'sofa': {
            'width': {'min': 1.5, 'max': 3.0, 'standard': 2.1},
            'depth': {'min': 0.8, 'max': 1.1, 'standard': 0.9},
            'height': {'min': 0.7, 'max': 0.9, 'standard': 0.8},
            'seat_height': 0.45
        },
        'loveseat': {
            'width': {'min': 1.2, 'max': 1.8, 'standard': 1.5},
            'depth': {'min': 0.8, 'max': 1.0, 'standard': 0.9},
            'height': {'min': 0.7, 'max': 0.9, 'standard': 0.8}
        },
        'armchair': {
            'width': {'min': 0.6, 'max': 1.0, 'standard': 0.8},
            'depth': {'min': 0.7, 'max': 0.9, 'standard': 0.8},
            'height': {'min': 0.7, 'max': 0.9, 'standard': 0.8}
        },
        'coffee_table': {
            'width': {'min': 0.8, 'max': 1.5, 'standard': 1.2},
            'depth': {'min': 0.4, 'max': 0.8, 'standard': 0.6},
            'height': {'min': 0.35, 'max': 0.5, 'standard': 0.42}
        },
        'side_table': {
            'width': {'min': 0.4, 'max': 0.7, 'standard': 0.5},
            'depth': {'min': 0.4, 'max': 0.6, 'standard': 0.5},
            'height': {'min': 0.5, 'max': 0.7, 'standard': 0.6}
        },
        'tv_console': {
            'width': {'min': 1.2, 'max': 2.4, 'standard': 1.8},
            'depth': {'min': 0.35, 'max': 0.6, 'standard': 0.45},
            'height': {'min': 0.4, 'max': 0.7, 'standard': 0.55}
        },
        'dining_table': {
            'width': {'min': 1.0, 'max': 2.4, 'standard': 1.6},
            'depth': {'min': 0.8, 'max': 1.2, 'standard': 0.9},
            'height': {'min': 0.72, 'max': 0.78, 'standard': 0.75}
        },
        'rug': {
            'width': {'min': 1.5, 'max': 4.0, 'standard': 2.4},
            'depth': {'min': 1.0, 'max': 3.0, 'standard': 1.7},
            'height': {'min': 0.01, 'max': 0.03, 'standard': 0.02}
        },
        'bookshelf': {
            'width': {'min': 0.6, 'max': 1.5, 'standard': 0.9},
            'depth': {'min': 0.25, 'max': 0.45, 'standard': 0.35},
            'height': {'min': 1.5, 'max': 2.2, 'standard': 1.8}
        }
    }
    
    CLEARANCE_RULES = {
        'walkway_minimum': 0.6,
        'walkway_comfortable': 0.9,
        'sofa_to_coffee_table': {'min': 0.35, 'max': 0.5},
        'dining_chair_pullout': 0.8,
        'doorway_clearance': 0.9,
        'tv_viewing_distance': lambda diagonal: diagonal * 2.5,
        'furniture_to_wall': 0.05,  # Small gap to prevent scraping
    }
    
    PLACEMENT_PRIORITIES = {
        'sofa': ['against_longest_wall', 'facing_focal_point', 'away_from_door'],
        'coffee_table': ['centered_with_seating', 'accessible_from_all_sides'],
        'tv_console': ['opposite_primary_seating', 'against_wall'],
        'rug': ['under_seating_group', 'defines_conversation_area'],
        'armchair': ['angled_to_sofa', 'creates_conversation_circle'],
    }
    
    @classmethod
    def get_standard_dimensions(cls, furniture_type: str) -> Dict[str, float]:
        """Get standard dimensions for a furniture type"""
        if furniture_type not in cls.ERGONOMIC_DIMENSIONS:
            return {'width': 1.0, 'depth': 0.5, 'height': 0.7}
        
        dims = cls.ERGONOMIC_DIMENSIONS[furniture_type]
        return {
            'width': dims['width']['standard'],
            'depth': dims['depth']['standard'],
            'height': dims['height']['standard']
        }
    
    @classmethod
    def scale_dimensions_for_room(cls, furniture_type: str, room_area: float) -> Dict[str, float]:
        """Scale furniture dimensions based on room size"""
        base_dims = cls.get_standard_dimensions(furniture_type)
        
        # Calculate scale factor based on room area
        # Small room: <12m², Medium: 12-25m², Large: >25m²
        if room_area < 12:
            scale_factor = 0.85
        elif room_area < 25:
            scale_factor = 1.0
        else:
            scale_factor = 1.15
        
        # Apply scaling with min/max constraints
        scaled = {}
        constraints = cls.ERGONOMIC_DIMENSIONS.get(furniture_type, {})
        
        for dim in ['width', 'depth', 'height']:
            scaled_value = base_dims[dim] * scale_factor
            if dim in constraints:
                scaled_value = np.clip(
                    scaled_value,
                    constraints[dim]['min'],
                    constraints[dim]['max']
                )
            scaled[dim] = scaled_value
        
        return scaled


class PhysicsPlacementEngine:
    """Handles physics-based furniture placement validation and optimization"""
    
    def __init__(self):
        self.physics_client = None
        self.room_id = None
        self.furniture_ids = []
        
    def initialize_physics(self):
        """Initialize PyBullet physics engine"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        return self.physics_client
    
    def create_room_boundaries(self, points_calibrated: np.ndarray, floor_mask: np.ndarray):
        """Create collision geometry for room boundaries"""
        # Extract floor points
        floor_points = points_calibrated[floor_mask]
        
        if len(floor_points) < 3:
            return None
        
        # Find room boundaries from floor points
        x_min, x_max = np.min(floor_points[:, 0]), np.max(floor_points[:, 0])
        z_min, z_max = np.min(floor_points[:, 2]), np.max(floor_points[:, 2])
        y_floor = np.mean(floor_points[:, 1])
        
        # Create floor plane
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[(x_max - x_min)/2, 0.01, (z_max - z_min)/2]
        )
        
        floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_shape,
            basePosition=[(x_min + x_max)/2, y_floor, (z_min + z_max)/2]
        )
        
        # Create walls (simplified as boxes)
        wall_height = 2.5
        wall_thickness = 0.1
        
        walls = []
        # Back wall (max z)
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[(x_max - x_min)/2, wall_height/2, wall_thickness/2]
        )
        walls.append(p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[(x_min + x_max)/2, y_floor + wall_height/2, z_max]
        ))
        
        # Side walls
        side_wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness/2, wall_height/2, (z_max - z_min)/2]
        )
        walls.append(p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=side_wall_shape,
            basePosition=[x_min, y_floor + wall_height/2, (z_min + z_max)/2]
        ))
        walls.append(p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=side_wall_shape,
            basePosition=[x_max, y_floor + wall_height/2, (z_min + z_max)/2]
        ))
        
        self.room_id = {'floor': floor_id, 'walls': walls, 'bounds': {
            'x_min': x_min, 'x_max': x_max,
            'z_min': z_min, 'z_max': z_max,
            'y_floor': y_floor
        }}
        
        return self.room_id
    
    def create_furniture_collision_box(self, dimensions: Dict[str, float], position: np.ndarray, rotation: float = 0):
        """Create a collision box for furniture"""
        half_extents = [
            dimensions['width'] / 2,
            dimensions['height'] / 2,
            dimensions['depth'] / 2
        ]
        
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        
        # Create quaternion for rotation around Y axis
        orientation = p.getQuaternionFromEuler([0, rotation, 0])
        
        furniture_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        return furniture_id
    
    def check_collision(self, furniture_id: int, position: np.ndarray) -> bool:
        """Check if furniture at position collides with existing items"""
        p.resetBasePositionAndOrientation(furniture_id, position, [0, 0, 0, 1])
        p.stepSimulation()
        
        # Check contacts
        contact_points = p.getContactPoints(bodyA=furniture_id)
        
        # Ignore floor contacts
        has_collision = False
        for contact in contact_points:
            if contact[2] != self.room_id['floor']:  # Not floor
                has_collision = True
                break
        
        return has_collision
    
    def cleanup(self):
        """Clean up physics simulation"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class RuleBasedFurniturePlacement:
    """Main placement system using rules and physics"""
    
    def __init__(self, log_fn=None):
        self.rules = InteriorDesignRules()
        self.physics = PhysicsPlacementEngine()
        self.log = log_fn or print
        
    def determine_furniture_set(self, room_analysis: Dict) -> List[FurnitureItem]:
        """Determine what furniture to place based on room type and dimensions"""
        furniture_list = []
        room_area = room_analysis['dimensions']['area']
        room_width = room_analysis['dimensions']['width']
        room_depth = room_analysis['dimensions']['depth']
        
        # Classify room type
        room_type = self._classify_room(room_analysis)
        self.log(f"Room classified as: {room_type}")
        self.log(f"Room dimensions: {room_width:.2f}m x {room_depth:.2f}m = {room_area:.2f}m²")
        
        if room_type == 'living_room':
            # Primary seating
            if room_area > 15:
                furniture_list.append(FurnitureItem(
                    type='sofa',
                    dimensions=self.rules.scale_dimensions_for_room('sofa', room_area),
                    priority=1
                ))
                # Add accent seating for larger rooms
                if room_area > 20:
                    for i in range(2):
                        furniture_list.append(FurnitureItem(
                            type='armchair',
                            dimensions=self.rules.scale_dimensions_for_room('armchair', room_area),
                            priority=2
                        ))
            else:
                furniture_list.append(FurnitureItem(
                    type='loveseat',
                    dimensions=self.rules.scale_dimensions_for_room('loveseat', room_area),
                    priority=1
                ))
            
            # Coffee table
            furniture_list.append(FurnitureItem(
                type='coffee_table',
                dimensions=self.rules.scale_dimensions_for_room('coffee_table', room_area),
                priority=2
            ))
            
            # TV console if wall is long enough
            if max(room_width, room_depth) > 3.5:
                furniture_list.append(FurnitureItem(
                    type='tv_console',
                    dimensions=self.rules.scale_dimensions_for_room('tv_console', room_area),
                    priority=3
                ))
            
            # Area rug
            rug_dims = self.rules.scale_dimensions_for_room('rug', room_area)
            # Ensure rug is appropriately sized for the room
            rug_dims['width'] = min(rug_dims['width'], room_width * 0.7)
            rug_dims['depth'] = min(rug_dims['depth'], room_depth * 0.7)
            furniture_list.append(FurnitureItem(
                type='rug',
                dimensions=rug_dims,
                priority=4
            ))
            
            # Side tables for larger rooms
            if room_area > 18:
                furniture_list.append(FurnitureItem(
                    type='side_table',
                    dimensions=self.rules.scale_dimensions_for_room('side_table', room_area),
                    priority=5
                ))
        
        elif room_type == 'bedroom':
            # Bed (simplified as a large box for now)
            bed_dims = {
                'width': 1.6 if room_area < 12 else 1.8,  # Queen or King
                'depth': 2.0,
                'height': 0.6
            }
            furniture_list.append(FurnitureItem(
                type='bed',
                dimensions=bed_dims,
                priority=1
            ))
            
            # Nightstands
            for i in range(2 if room_area > 10 else 1):
                furniture_list.append(FurnitureItem(
                    type='side_table',
                    dimensions=self.rules.scale_dimensions_for_room('side_table', room_area),
                    priority=2
                ))
        
        elif room_type == 'dining_room':
            furniture_list.append(FurnitureItem(
                type='dining_table',
                dimensions=self.rules.scale_dimensions_for_room('dining_table', room_area),
                priority=1
            ))
        
        return furniture_list
    
    def optimize_placement(self, furniture_items: List[FurnitureItem], room_analysis: Dict) -> List[Dict]:
        """Find optimal placement for each furniture item"""
        self.log("Initializing physics simulation...")
        self.physics.initialize_physics()
        self.physics.create_room_boundaries(
            room_analysis['points_calibrated'],
            room_analysis['floor_mask']
        )
        
        placements = []
        placed_items = []
        
        # Sort by priority
        furniture_items.sort(key=lambda x: x.priority)
        
        for item in furniture_items:
            self.log(f"Placing {item.type}...")
            
            # Generate candidate positions
            candidates = self._generate_placement_candidates(item, room_analysis, placed_items)
            
            # Score and select best placement
            best_placement = None
            best_score = -float('inf')
            
            # Create temporary collision box for testing
            test_position = [0, 0, 0]
            furniture_id = self.physics.create_furniture_collision_box(
                item.dimensions,
                test_position
            )
            
            for candidate in candidates:
                score = self._score_placement(
                    item, candidate, placed_items, room_analysis, furniture_id
                )
                
                if score > best_score:
                    best_score = score
                    best_placement = candidate
            
            if best_placement and best_score > 0:
                placements.append({
                    'type': item.type,
                    'dimensions': item.dimensions,
                    'position': best_placement['position'],
                    'rotation': best_placement['rotation'],
                    'score': best_score
                })
                placed_items.append({
                    'type': item.type,
                    'dimensions': item.dimensions,
                    'position': best_placement['position'],
                    'rotation': best_placement['rotation'],
                    'furniture_id': furniture_id
                })
                self.log(f"  Placed at {best_placement['position']} with score {best_score:.2f}")
            else:
                self.log(f"  Could not find valid placement for {item.type}")
                p.removeBody(furniture_id)
        
        self.physics.cleanup()
        return placements
    
    def _classify_room(self, room_analysis: Dict) -> str:
        """Classify room type based on dimensions and features"""
        area = room_analysis['dimensions']['area']
        width = room_analysis['dimensions']['width']
        depth = room_analysis['dimensions']['depth']
        aspect_ratio = max(width, depth) / min(width, depth)
        
        # Simple heuristic classification
        if area < 10:
            return 'bedroom'
        elif aspect_ratio > 2.0:
            return 'hallway'
        elif area < 15:
            return 'bedroom' if area < 12 else 'dining_room'
        else:
            return 'living_room'
    
    def _generate_placement_candidates(self, item: FurnitureItem, room_analysis: Dict, 
                                      placed_items: List[Dict]) -> List[Dict]:
        """Generate candidate positions for furniture placement"""
        candidates = []
        bounds = room_analysis['floor_bounds']
        
        # Create grid of possible positions
        x_range = bounds['x_max'] - bounds['x_min']
        z_range = bounds['z_max'] - bounds['z_min']
        
        # Adjust grid density based on furniture size
        grid_step = min(item.dimensions['width'], item.dimensions['depth']) / 2
        
        # Special case for rugs - place in center
        if item.type == 'rug':
            center_x = (bounds['x_min'] + bounds['x_max']) / 2
            center_z = (bounds['z_min'] + bounds['z_max']) / 2
            candidates.append({
                'position': np.array([center_x, bounds['y_floor'] + 0.01, center_z]),
                'rotation': 0
            })
            return candidates
        
        # Generate positions along walls first (preferred for most furniture)
        wall_offset = item.dimensions['depth'] / 2 + self.rules.CLEARANCE_RULES['furniture_to_wall']
        
        # Along back wall (max z)
        for x in np.arange(bounds['x_min'] + item.dimensions['width']/2, 
                          bounds['x_max'] - item.dimensions['width']/2, grid_step):
            candidates.append({
                'position': np.array([x, bounds['y_floor'] + item.dimensions['height']/2, 
                                    bounds['z_max'] - wall_offset]),
                'rotation': 0
            })
        
        # Along side walls
        for z in np.arange(bounds['z_min'] + item.dimensions['depth']/2,
                          bounds['z_max'] - item.dimensions['depth']/2, grid_step):
            # Left wall
            candidates.append({
                'position': np.array([bounds['x_min'] + wall_offset,
                                    bounds['y_floor'] + item.dimensions['height']/2, z]),
                'rotation': np.pi/2
            })
            # Right wall
            candidates.append({
                'position': np.array([bounds['x_max'] - wall_offset,
                                    bounds['y_floor'] + item.dimensions['height']/2, z]),
                'rotation': -np.pi/2
            })
        
        # For coffee tables and center furniture, also try center positions
        if item.type in ['coffee_table', 'dining_table']:
            for x in np.arange(bounds['x_min'] + x_range * 0.3, 
                             bounds['x_max'] - x_range * 0.3, grid_step):
                for z in np.arange(bounds['z_min'] + z_range * 0.3,
                                 bounds['z_max'] - z_range * 0.3, grid_step):
                    candidates.append({
                        'position': np.array([x, bounds['y_floor'] + item.dimensions['height']/2, z]),
                        'rotation': 0
                    })
        
        return candidates
    
    def _score_placement(self, item: FurnitureItem, candidate: Dict, placed_items: List[Dict],
                        room_analysis: Dict, furniture_id: int) -> float:
        """Score a placement candidate based on multiple criteria"""
        position = candidate['position']
        rotation = candidate['rotation']
        score = 100.0  # Base score
        
        # 1. Check collision (hard constraint)
        if self.physics.check_collision(furniture_id, position):
            return -1000.0
        
        # 2. Check clearances
        min_clearance = self._calculate_min_clearance(item, position, placed_items)
        if min_clearance < self.rules.CLEARANCE_RULES['walkway_minimum']:
            score -= 50.0
        else:
            score += min_clearance * 10.0
        
        # 3. Furniture-specific scoring
        if item.type == 'sofa' or item.type == 'loveseat':
            # Prefer against longest wall
            wall_distance = self._distance_to_nearest_wall(position, room_analysis)
            if wall_distance < 0.3:
                score += 30.0
            
            # Prefer facing center of room
            room_center = np.array([
                (room_analysis['floor_bounds']['x_min'] + room_analysis['floor_bounds']['x_max']) / 2,
                position[1],
                (room_analysis['floor_bounds']['z_min'] + room_analysis['floor_bounds']['z_max']) / 2
            ])
            facing_center = self._is_facing_point(position, rotation, room_center)
            if facing_center:
                score += 20.0
        
        elif item.type == 'coffee_table':
            # Must be near seating
            sofa_distance = self._distance_to_furniture_type('sofa', position, placed_items)
            if sofa_distance is not None:
                optimal_distance = 0.45
                distance_penalty = abs(sofa_distance - optimal_distance)
                if distance_penalty < 0.2:
                    score += 40.0
                else:
                    score -= distance_penalty * 20.0
            else:
                # No sofa placed yet, use heuristic
                score += 10.0
        
        elif item.type == 'tv_console':
            # Should be opposite primary seating
            sofa_pos = self._get_furniture_position('sofa', placed_items)
            if sofa_pos is not None:
                # Check if opposite to sofa
                relative_pos = position - sofa_pos
                if np.dot(relative_pos, np.array([0, 0, 1])) < 0:  # Opposite in Z
                    score += 30.0
            
            # Must be against wall
            wall_distance = self._distance_to_nearest_wall(position, room_analysis)
            if wall_distance < 0.2:
                score += 25.0
        
        elif item.type == 'armchair':
            # Should create conversation circle with sofa
            sofa_pos = self._get_furniture_position('sofa', placed_items)
            if sofa_pos is not None:
                distance_to_sofa = np.linalg.norm(position - sofa_pos)
                if 1.5 < distance_to_sofa < 2.5:  # Good conversation distance
                    score += 25.0
        
        # 4. Visual balance and symmetry
        symmetry_score = self._calculate_symmetry_score(item, position, placed_items, room_analysis)
        score += symmetry_score * 15.0
        
        return score
    
    def _distance_to_nearest_wall(self, position: np.ndarray, room_analysis: Dict) -> float:
        """Calculate distance from position to nearest wall"""
        bounds = room_analysis['floor_bounds']
        distances = [
            abs(position[0] - bounds['x_min']),
            abs(position[0] - bounds['x_max']),
            abs(position[2] - bounds['z_min']),
            abs(position[2] - bounds['z_max'])
        ]
        return min(distances)
    
    def _calculate_min_clearance(self, item: FurnitureItem, position: np.ndarray, 
                                placed_items: List[Dict]) -> float:
        """Calculate minimum clearance to other furniture"""
        if not placed_items:
            return 2.0  # Large clearance if nothing placed yet
        
        min_clearance = float('inf')
        for placed in placed_items:
            distance = np.linalg.norm(position - placed['position'])
            # Subtract half-widths to get actual clearance
            clearance = distance - (
                max(item.dimensions['width'], item.dimensions['depth'])/2 +
                max(placed['dimensions']['width'], placed['dimensions']['depth'])/2
            )
            min_clearance = min(min_clearance, clearance)
        
        return min_clearance
    
    def _distance_to_furniture_type(self, furniture_type: str, position: np.ndarray, 
                                   placed_items: List[Dict]) -> Optional[float]:
        """Calculate distance to nearest furniture of given type"""
        min_distance = None
        for placed in placed_items:
            if placed['type'] == furniture_type or placed['type'] == 'loveseat' and furniture_type == 'sofa':
                distance = np.linalg.norm(position - placed['position'])
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance
    
    def _get_furniture_position(self, furniture_type: str, placed_items: List[Dict]) -> Optional[np.ndarray]:
        """Get position of first furniture of given type"""
        for placed in placed_items:
            if placed['type'] == furniture_type:
                return placed['position']
        return None
    
    def _is_facing_point(self, position: np.ndarray, rotation: float, point: np.ndarray) -> bool:
        """Check if furniture at position/rotation is facing a point"""
        # Calculate facing direction (assuming rotation around Y axis)
        facing_dir = np.array([np.sin(rotation), 0, np.cos(rotation)])
        to_point = point - position
        to_point = to_point / np.linalg.norm(to_point)
        
        # Check if angle is less than 45 degrees
        dot_product = np.dot(facing_dir, to_point)
        return dot_product > 0.7  # cos(45°) ≈ 0.7
    
    def _calculate_symmetry_score(self, item: FurnitureItem, position: np.ndarray,
                                 placed_items: List[Dict], room_analysis: Dict) -> float:
        """Calculate visual balance and symmetry score"""
        if not placed_items:
            # First item, prefer center
            room_center = np.array([
                (room_analysis['floor_bounds']['x_min'] + room_analysis['floor_bounds']['x_max']) / 2,
                position[1],
                (room_analysis['floor_bounds']['z_min'] + room_analysis['floor_bounds']['z_max']) / 2
            ])
            distance_to_center = np.linalg.norm(position[:2] - room_center[:2])
            return 1.0 / (1.0 + distance_to_center)
        
        # Calculate center of mass of existing furniture
        com = np.mean([p['position'] for p in placed_items], axis=0)
        
        # New center of mass with this item
        new_com = (com * len(placed_items) + position) / (len(placed_items) + 1)
        
        # Score based on how close new COM is to room center
        room_center = np.array([
            (room_analysis['floor_bounds']['x_min'] + room_analysis['floor_bounds']['x_max']) / 2,
            position[1],
            (room_analysis['floor_bounds']['z_min'] + room_analysis['floor_bounds']['z_max']) / 2
        ])
        
        old_distance = np.linalg.norm(com - room_center)
        new_distance = np.linalg.norm(new_com - room_center)
        
        # Return improvement in balance
        improvement = old_distance - new_distance
        return improvement
