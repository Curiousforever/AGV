from argparse import ArgumentParser
import os
from random import choice
import vizdoom as vzd
import numpy as np
import random
import math
import cv2
import time

#DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
DEFAULT_CONFIG = "../../scenarios/level1.cfg"

import cv2
CELL_SIZE = 32  # Size of each cell in our grid map
MAP_SIZE = 400  # Number of cells in each dimension (creates a 100x100 grid)
FORWARD_STEP = 60  # Distance moved in one forward step (increased from 10)
TURN_ANGLE = 15  # Angle turned in one turn step
WALL_THRESHOLD = 8  # Distance threshold for detecting walls
PATH_THRESHOLD = 40

class GridMazeExplorer:
    def __init__(self):
        # Initialize grid map (0=unknown, 1=free space, 2=wall)
        self.grid_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        
        # Starting position (center of grid)
        self.pos_x = 0
        self.pos_y = 0
        self.orientation = 270  # 0=east, 90=north, 180=west, 270=south

        self.doom_x = 0
        self.doom_y = 0
        self.doom_angle = 0

        # Target location variables
        self.target_x = 476.56150818
        self.target_y = -1781.43080139
        self.target_detection_range = 300  # Doom units
        self.target_navigation_active = False
        self.target_reached = False

        self.unstucking = False
        self.unstuck_steps = 0
        self.unstuck_direction = None
        self.unstuck_last_pos = None
        
        # DFS tracking
        self.visited = set([(self.pos_x, self.pos_y)])
        self.decision_points = {}  # {(x,y): [tried_directions]}
        self.movement_history = []  # List of (from_pos, to_pos, direction) for backtracking
        self.last_dp_pos = None    # Last decision point position
        
        # Add tracking for confirmed non-decision points
        self.confirmed_non_dp = set()  # Set of positions confirmed not to be decision points
        
        # State tracking
        self.state = "EXPLORING"   # EXPLORING, SCANNING, BACKTRACKING
        self.scan_stage = 0        # 0=forward, 1=right, 2=left
        self.open_directions = []  # List of open directions found during scanning
        self.stuck_counter = 0     # Counter for detecting when stuck
        self.last_pos = (self.pos_x, self.pos_y)
        
        # Add first move flag
        self.is_first_move = True
        
        # Visualization
        self.map_display = np.zeros((MAP_SIZE*CELL_SIZE, MAP_SIZE*CELL_SIZE, 3), dtype=np.uint8)
        self.backtrack_path = []   # For visualizing backtracking

    def check_target_proximity(self):
        distance_to_target = math.sqrt((self.doom_x - self.target_x)**2 + 
                                   (self.doom_y - self.target_y)**2)
    
        if distance_to_target < self.target_detection_range and not self.target_reached:
            if not self.target_navigation_active:
                print(f"\n=== TARGET DETECTED {distance_to_target:.1f} UNITS AWAY! ===")
                print(f"Current position: ({self.doom_x:.1f}, {self.doom_y:.1f})")
                print(f"Target position: ({self.target_x:.1f}, {self.target_y:.1f})")
                print("=== ABANDONING EXPLORATION TO NAVIGATE TO TARGET ===\n")
                self.target_navigation_active = True
        
            return True
        return False
    
    def update_position(self, doom_x, doom_y, angle):
        """Update position based on game variables."""

        prev_doom_x, prev_doom_y = self.doom_x, self.doom_y
    
    # Update Doom coordinates
        self.doom_x = doom_x
        self.doom_y = doom_y
        self.doom_angle = angle

        # Convert Doom coordinates to grid coordinates
        grid_x = int(doom_x / CELL_SIZE + MAP_SIZE // 2)
        grid_y = int(doom_y / CELL_SIZE + MAP_SIZE // 2)
        
        # Update position and orientation
        prev_x, prev_y = self.pos_x, self.pos_y
        self.pos_x = grid_x
        self.pos_y = grid_y
        self.orientation = angle
        
        # Ensure position is within bounds
        self.pos_x = max(0, min(self.pos_x, MAP_SIZE-1))
        self.pos_y = max(0, min(self.pos_y, MAP_SIZE-1))
        
        # Mark current cell as free space
        self.grid_map[self.pos_y, self.pos_x] = 1
        
        # Check if we've moved
        current_pos = (self.pos_x, self.pos_y)
        if current_pos != self.last_pos:
            # Record movement in history
            if self.state == "EXPLORING":
                # Calculate cardinal direction of movement
                direction = None
                dx = self.pos_x - prev_x
                dy = self.pos_y - prev_y
                
                if abs(dx) > abs(dy):  # Primarily horizontal movement
                    direction = 0 if dx > 0 else 180  # East or West
                else:  # Primarily vertical movement
                    direction = 90 if dy > 0 else 270  # North or South
                    
                if direction is not None:
                    self.movement_history.append((
                        (prev_x, prev_y),
                        current_pos,
                        direction
                    ))
            
            # Reset stuck counter
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1
        
        # Mark as visited
        self.visited.add(current_pos)
        self.last_pos = current_pos
        
    def analyze_depth_buffer(self, depth_buffer):
        """Analyze depth buffer to detect walls and openings (center region only)."""
        if (depth_buffer is None):
            return None
            
        # Get depth buffer dimensions
        h, w = depth_buffer.shape
        normalized_depth = cv2.normalize(depth_buffer, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        
        # Only analyze the center region
        center = depth_buffer[:, :]

        viz = depth_colored.copy()
        
        # Calculate maximum distance in center section (higher = open passage)
        max_center = np.max(center) if center.size > 0 else 0
        
        # Create visualization
        section_viz = depth_colored.copy()
        
        # Draw vertical lines showing the center section
        cv2.line(section_viz, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        # Add distance text for the center section
        left_half = depth_buffer[:, :w//2]
        right_half = depth_buffer[:, w//2:]
        left_max = np.max(left_half) if left_half.size > 0 else 0
        right_max = np.max(right_half) if right_half.size > 0 else 0
        # Add threshold info and status
        center_open = max_center > WALL_THRESHOLD
        path_open = max_center > PATH_THRESHOLD

        status = "PATH OPEN" if center_open else "PATH BLOCKED"
        color = (0, 255, 0) if center_open else (0, 0, 255)
        cv2.putText(viz, status, (w//2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Depth Analysis', viz)

        
        return {
            'center': max_center,
            'center_open': center_open,
            'path_open': path_open,
            'left_max': left_max,
            'right_max': right_max

        }
    def update_grid_from_depth(self, depth_map):
        """Update grid map with walls and open spaces."""
        h, w = depth_map.shape
        center_x, center_y = w // 2, h // 2
        
        # Mark cells in front of agent
        angle_rad = math.radians(self.orientation)
        for dist in range(1, 20):
            dx = int(dist * math.cos(angle_rad))
            dy = int(dist * math.sin(angle_rad))
            
            grid_x = self.pos_x + dx
            grid_y = self.pos_y + dy  # Negative because y-axis is flipped
            
            if 0 <= grid_x < MAP_SIZE and 0 <= grid_y < MAP_SIZE:
                depth_val = depth_map[center_y, center_x] if center_y < h and center_x < w else 0
                
                if depth_val < WALL_THRESHOLD:
                    # Mark as wall
                    self.grid_map[grid_y, grid_x] = 2
                    break
                else:
                    # Mark as free space
                    self.grid_map[grid_y, grid_x] = 1
    
    def decide_action(self, depth_info, game):
        """Main decision dispatcher with coherent state transitions."""

        if depth_info is None:
            print("No depth information available, using default action")
            return [False, False, False, False, 0]  # Return default action when no depth info


        if self.check_target_proximity():
            # If yes, navigate to it
            return self.navigate_to_target()

        print("\n=== DECISION POINTS ===")
        for dp_pos, dp_info in self.decision_points.items():
            doom_dp_x = (dp_pos[0] - MAP_SIZE // 2) * CELL_SIZE
            doom_dp_y = (dp_pos[1] - MAP_SIZE // 2) * CELL_SIZE
            open_dirs = dp_info["open"] if isinstance(dp_info, dict) else []
            tried_dirs = dp_info["tried"] if isinstance(dp_info, dict) else dp_info
            untried = [d for d in open_dirs if d not in tried_dirs] if isinstance(dp_info, dict) else []
        
            print(f"  DP at grid ({dp_pos[0]}, {dp_pos[1]}) = Doom ({doom_dp_x:.1f}, {doom_dp_y:.1f}): {len(untried)} untried of {len(open_dirs)} open paths")
        print("=====================\n")

        # Check if we're currently in the process of unstucking
        if self.unstucking:
            # Get current position
            current_pos = (self.pos_x, self.pos_y)
            
            # Calculate progress since last position
            if self.unstuck_last_pos:
                distance_moved = math.sqrt((current_pos[0] - self.unstuck_last_pos[0])**2 + 
                                          (current_pos[1] - self.unstuck_last_pos[1])**2)
                
                # If we're making significant progress, exit unstucking
                if distance_moved > 5.0:
                    print(f"Made progress with side movement: {distance_moved:.2f} units")
                    self.unstucking = False
                    self.stuck_counter = 0
                    self.unstuck_steps = 0
                    # Resume normal exploration
                    return [False, False, True, False, 0]  # Try moving forward
            
            # Update last position for next comparison
            self.unstuck_last_pos = current_pos
            
            # Increment step counter
            self.unstuck_steps += 1
            
            # If we've tried enough steps without success, check depth info
            if self.unstuck_steps >= 5:
                # Get new depth readings
                left_max = depth_info['left_max']
                right_max = depth_info['right_max']
                
                # SCENARIO 3: Both sides blocked after unstucking attempts
                if left_max < 0.01 and right_max < 0.01:
                    print("Both sides completely blocked after unstucking attempts. Initiating backtracking.")
                    self.unstucking = False
                    self.state = "BACKTRACKING"
                    return self.backtrack()
                
                # Change direction if we haven't made progress
                print(f"Unstucking attempts exceeded limit. Switching direction.")
                
                # Flip the direction
                if self.unstuck_direction[0]:  # If was moving left
                    self.unstuck_direction = [False, True, False, False, 0]  # Now move right
                else:  # If was moving right
                    self.unstuck_direction = [True, False, False, False, 0]  # Now move left
                
                # Reset step counter but stay in unstucking mode
                self.unstuck_steps = 0
                return self.unstuck_direction
            
            # Continue moving in the chosen direction
            print(f"Continuing unstucking in direction {self.unstuck_direction}, step {self.unstuck_steps}/10")
            return self.unstuck_direction
        
        # First move or state-based dispatch
        if self.is_first_move:
            print("\n=== MAKING INITIAL 90° LEFT TURN ===")
            self.is_first_move = False
            return [False, False, False, False, -90]  # Turn 90° left
        
        current_pos = (self.pos_x, self.pos_y)
        current_cardinal = self.get_cardinal_direction()
        
        # State-based dispatch
        if self.state == "EXPLORING":
            return self.handle_exploring_state(depth_info, current_pos, current_cardinal)
        elif self.state == "SCANNING":
            return self.handle_scanning_state(depth_info, current_pos, current_cardinal)
        elif self.state == "BACKTRACKING":
            return self.handle_backtracking_state(depth_info, current_pos)
        
        return [False, False, False, False, 0]
    
    def get_cardinal_direction(self):
        """Get current cardinal direction (0, 90, 180, 270)."""
        return round(self.orientation / 90) * 90 % 360
    
    def handle_exploring_state(self, depth_info, current_pos, current_cardinal):
        """Handle EXPLORING state actions with coherent logic."""
        # SCENARIO 1: Front path open but agent is stuck
        if depth_info['center_open'] and self.stuck_counter > 10:
            print("\n=== PATH OPEN BUT STUCK! Starting unstucking sequence ===")
            # Initialize unstucking process
            self.unstucking = True
            self.unstuck_steps = 0
            self.unstuck_last_pos = (self.pos_x, self.pos_y)
            
            # Choose direction based on depth data
            left_max = depth_info['left_max']
            right_max = depth_info['right_max']
            
            if left_max > right_max:
                print(f"Left side more open (left: {left_max:.3f} vs right: {right_max:.3f}), moving left")
                self.unstuck_direction = [True, False, False, False, 0]  # Move left
                return self.unstuck_direction
            else:
                print(f"Right side more open (right: {right_max:.3f} vs left: {left_max:.3f}), moving right")
                self.unstuck_direction = [False, True, False, False, 0]  # Move right
                return self.unstuck_direction
        
        # SCENARIO 2: Front path blocked - scan for alternatives
        elif not depth_info['center_open'] and self.is_decision_point(depth_info):
            print(f"\n=== PATH BLOCKED at {current_cardinal}°, scanning for alternatives ===")
            return self.start_scanning(current_cardinal, skip_forward=True)
        
        # Normal case: path is open and we're not stuck
        elif depth_info['center_open']:
            print(f"\n=== PATH OPEN at {current_cardinal}°, exploring ===")
            
            # Check if we're at a potential decision point
            if depth_info['path_open'] and self.is_decision_point(depth_info):
                print(f"Potential decision point detected at {current_pos}")
                return self.start_scanning(current_cardinal)
            else:
                # Not a decision point, move forward
                return [False, 0, True, False, 0]
            
        else:
            print("\n=== ALL PATHS BLOCKED! Initiating backtracking ===")
            self.state = "BACKTRACKING"
            return self.backtrack()


    
    def start_scanning(self, current_cardinal, skip_forward=False):
        """Start the scanning process to look in different directions."""
        self.state = "SCANNING"
        self.open_directions = []

        time.sleep(0.1)
        
        # If forward direction should be checked
        if not skip_forward:
            # Forward is already known to be open from the calling context
            self.open_directions = [current_cardinal]
            print(f"Forward path is open at {current_cardinal}°")
        else:
            print(f"Forward path is blocked at {current_cardinal}°")
        
        # Turn right 90° to check the right path
        self.scan_stage = 1  # Stage 1: checking right
        print(f"Scanning: Turning right 90° from {current_cardinal}°")
        return [False, False, False, False, 90]  # Turn right 90° exactly
    
    def handle_scanning_state(self, depth_info, current_pos, current_cardinal):
        """Handle SCANNING state actions."""
        if self.scan_stage == 1:
            return self.scan_right_path(depth_info, current_cardinal)
        elif self.scan_stage == 2:
            return self.scan_left_path(depth_info, current_cardinal, current_pos)
        return [False, False, False, False, 0]
    
    def scan_right_path(self, depth_info, current_cardinal):
        """Check if the right path is open during scanning."""
        # We've just turned right, check if path is open
        if depth_info['path_open']:
            print(f"Found open path to the right at {current_cardinal}°")
            # Add to open directions if not already present
            if current_cardinal not in self.open_directions:
                self.open_directions.append(current_cardinal)

        time.sleep(0.1)
        
        # Turn left 180° to check the left path (-180° from current orientation)
        self.scan_stage = 2
        print(f"Scanning: Turning left 180° to check left path")
        return [False, False, False, False, -180]  # Turn left 180° exactly
    
    def scan_left_path(self, depth_info, current_cardinal, current_pos):
        """Check if the left path is open during scanning."""
        # We've just turned left, check if path is open
        if depth_info['path_open']:
            print(f"Found open path to the left at {current_cardinal}°")
            # Add to open directions if not already present
            if current_cardinal not in self.open_directions:
                self.open_directions.append(current_cardinal)

        time.sleep(0.1)
        
        # Scanning complete, process results
        print(f"\n=== SCANNING COMPLETE! Found {len(set(self.open_directions))} unique open paths ===")
        
        # Confirm if this is a decision point (2+ open paths)
        is_decision_point = self.confirm_decision_point()
        
        if is_decision_point:
            print(f"Confirmed decision point with {len(set(self.open_directions))} open paths")
        else:
            print(f"Not a decision point - fewer than 2 open paths found")
            self.confirmed_non_dp.add(current_pos)
        
        # Remove any duplicate directions
        self.open_directions = list(set(self.open_directions))
        return self.make_decision_after_scanning(current_pos)
    
    def make_decision_after_scanning(self, current_pos):
        """Choose a path after completing scanning with better handling for blocked scenarios."""
        # SCENARIO 3: All paths blocked (front, left, and right)
        if len(self.open_directions) == 0:
            print("\n=== ALL PATHS BLOCKED! Initiating backtracking ===")
            self.state = "BACKTRACKING"
            return self.backtrack()
        
        # Potential decision point with multiple options
        elif len(set(self.open_directions)) >= 2:
            print(f"Confirmed decision point with {len(set(self.open_directions))} open paths")
            return self.handle_decision_point(current_pos)
        
        # Just one path available - not a decision point
        else:
            print(f"Single path available - not a decision point")
            return self.follow_single_path(current_pos)
    
    def handle_decision_point(self, current_pos):
        """Handle logic at a decision point with multiple options."""
        # Record as decision point if not already
        if current_pos not in self.decision_points:
        # Initialize with empty tried list and current open directions
            self.decision_points[current_pos] = {
                "tried": [],
                "open": list(set(self.open_directions))  # Store the unique open directions
            }
        
        # Filter out already tried directions
        untried = self.get_untried_directions(current_pos)
        
        if untried:
            return self.choose_random_untried_path(current_pos, untried)
        else:
            # All directions tried at this point, backtracking
            print(f"All paths from {current_pos} have been tried and failed")
            print(f"Backtracking to previous decision point with untried paths")
            self.state = "BACKTRACKING"
            return self.backtrack()
    
    def get_untried_directions(self, current_pos):
        """Get list of untried directions at a decision point."""
        open_directions = self.decision_points[current_pos]["open"]
        tried_directions = self.decision_points[current_pos]["tried"]
        return [d for d in open_directions if d not in tried_directions]
    
    def choose_random_untried_path(self, current_pos, untried):
        """Choose a random untried path at a decision point."""
        # Choose random untried direction
        chosen_dir = random.choice(untried)
        self.decision_points[current_pos]["tried"].append(chosen_dir)
        self.last_dp_pos = current_pos
        
        print(f"Decision point! Chose direction {chosen_dir}° from options: {self.open_directions}")
        
        # Return to exploring with the chosen direction
        self.state = "EXPLORING"
        
        # Turn to face the chosen direction
        return self.turn_to_direction(chosen_dir)
    
    def follow_single_path(self, current_pos):
        """Follow the only available path."""
        chosen_dir = self.open_directions[0]
        
        print(f"Only one path available: {chosen_dir}°")
        self.state = "EXPLORING"
        
        # Turn to face that direction
        return self.turn_to_direction(chosen_dir)
    
    def turn_to_direction(self, target_direction):
        """Create an action to turn to the specified direction."""
        angle_diff = target_direction - self.orientation
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
        return [False, False, False, False, -angle_diff]
    
    def handle_backtracking_state(self, depth_info, current_pos):
        """Handle BACKTRACKING state actions."""
        nearest_dp = self.find_nearest_decision_point_with_untried_paths()

        if nearest_dp is None:
            print("No decision points with untried paths found. Exploration complete.")
            self.state = "EXPLORING"  # Reset to exploring
            return [False, False, False, False, 0]
        
        grid_distance = abs(current_pos[0] - nearest_dp[0]) + abs(current_pos[1] - nearest_dp[1])
    
        if grid_distance <= 3: 
            return self.handle_decision_point(nearest_dp)    
        return self.backtrack()
    
    def backtrack(self):
        """Backtrack to previous decision point with untried paths."""
        print("\n=== BACKTRACKING ===")
        
        # Calculate current cardinal direction
        current_cardinal = self.get_cardinal_direction()
        
        # Add current position to backtracking visualization
        current_pos = (self.pos_x, self.pos_y)
        self.backtrack_path.append(current_pos)
        
        # If we have movement history to backtrack
        if self.movement_history:
            return self.backtrack_along_history(current_pos)
        
        # No movement history left
        print("No movement history to backtrack, resetting to exploration")
        self.state = "EXPLORING"
        return [False, False, False, False, 0]
    
    def backtrack_along_history(self, current_pos):
        """Backtrack along the recorded movement history."""
        # Get the last move
        last_from, last_to, last_dir = self.movement_history[-1]

        distance = math.sqrt((current_pos[0] - last_to[0])**2 + (current_pos[1] - last_to[1])**2)
    

        if distance > 3:
            return self.perform_backtrack_step(last_to, current_pos)
        
        # If we're at the position where this move ended
        return self.perform_backtrack_step(last_from, current_pos)
    
    def perform_backtrack_step(self, last_from, current_pos):
        """Execute one step of backtracking."""
        dx = last_from[0] - current_pos[0]
        dy = last_from[1] - current_pos[1]
    
    # Calculate the actual angle to the previous position
        backtrack_angle = None
    
        if abs(dx) > abs(dy):  # Primarily horizontal movement
            backtrack_angle = 0 if dx > 0 else 180  # East or West
        else:  # Primarily vertical movement
            backtrack_angle = 90 if dy > 0 else 270  # North or South
    
        print(f"Backtracking from {current_pos} to {last_from}, calculated angle: {backtrack_angle}°")

    # Turn to face the correct direction
        angle_diff = backtrack_angle - self.orientation
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
    
        if abs(angle_diff) < 5:  # Small threshold for angle difference
        # We're facing the right way, move forward 
            if math.sqrt((current_pos[0] - last_from[0])**2 + (current_pos[1] - last_from[1])**2) < 5:
                self.movement_history.pop()
            return [False, False, True, False, 0]
        else:
        # Need to turn first
            print(f"Turning {angle_diff}° to face direction {backtrack_angle}°")
            return [False, False, False, False, -angle_diff]       
    
    def find_nearest_decision_point_with_untried_paths(self):
        """Find the nearest decision point that has untried paths."""
        nearest_dp = None
        nearest_dist = float('inf')

        current_pos = (self.pos_x, self.pos_y)
        
        for dp_pos in self.decision_points:
            # Check if this decision point has untried paths
            open_dirs = self.decision_points[dp_pos]["open"]
            tried_dirs = self.decision_points[dp_pos]["tried"]
            untried = [d for d in open_dirs if d not in tried_dirs]
            
            if untried:
                dist = abs(dp_pos[0] - current_pos[0]) + abs(dp_pos[1] - current_pos[1])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_dp = dp_pos
        
        return nearest_dp
    
    def is_decision_point(self, depth_info):
        """
        Check if current position could be a decision point.
        For a point to be considered a decision point, it must:
        1. Be spaced sufficiently from the last decision point
        2. Have at least 2 open paths (will be confirmed after scanning)
        """
        current_pos = (self.pos_x, self.pos_y)

        if current_pos in self.confirmed_non_dp:
            print(f"Position {current_pos} already confirmed as non-decision point, skipping check")
            return False
        
        # Check spacing from last decision point
        for dp_pos in self.decision_points:
            manhattan_dist = abs(current_pos[0] - dp_pos[0]) + abs(current_pos[1] - dp_pos[1])
            if manhattan_dist < 9:  # Too close to an existing decision point
                print(f"Too close to existing decision point at {dp_pos} (dist={manhattan_dist}), skipping")
                return False
        
        # Initial check passed - we'll confirm if it's a true decision point after scanning
        return True
    
    def confirm_decision_point(self):
        """
        Confirm if current location is a decision point after scanning.
        Requires at least 2 open paths from the scanned directions (forward, right, left).
        """
        # Count unique open paths found during scanning
        open_path_count = len(set(self.open_directions))
        
        # A decision point must have at least 2 open paths
        return open_path_count >= 2

    def visualize(self):
        """Create a more compact and effective visualization of the exploration map."""
        # Use a smaller size for visualization
        display_size = 900  # Fixed display size
        cell_size = 2  # Smaller cell size for visualization
        
        # Create display map
        self.map_display = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        self.map_display.fill(30)  # Dark gray background for unexplored
        
        # Calculate center and scale
        center_x, center_y = display_size // 2, display_size // 2
        
        # Add grid lines every 5 cells
        for i in range(0, display_size, cell_size * 5):
            cv2.line(self.map_display, (i, 0), (i, display_size), (50, 50, 50), 1)
            cv2.line(self.map_display, (0, i), (display_size, i), (50, 50, 50), 1)
        
        # Draw the visited cells - FLIPPED Y-AXIS
        for pos in self.visited:
            x, y = pos
            screen_x = center_x + (x - MAP_SIZE // 2) * cell_size
            # Flip Y-axis here
            screen_y = center_y - (y - MAP_SIZE // 2) * cell_size
            
            if 0 <= screen_x < display_size - cell_size and 0 <= screen_y < display_size - cell_size:
                cv2.rectangle(self.map_display, 
                             (int(screen_x), int(screen_y)), 
                             (int(screen_x + cell_size), int(screen_y + cell_size)), 
                             (0, 80, 0), -1)  # Dark green for visited
        
        # Draw exploration path with direction arrows - FLIPPED Y-AXIS
        if self.movement_history:
            for from_pos, to_pos, direction in self.movement_history:
                from_x = center_x + (from_pos[0] - MAP_SIZE // 2) * cell_size + cell_size // 2
                # Flip Y-axis here
                from_y = center_y - (from_pos[1] - MAP_SIZE // 2) * cell_size + cell_size // 2
                to_x = center_x + (to_pos[0] - MAP_SIZE // 2) * cell_size + cell_size // 2
                # Flip Y-axis here
                to_y = center_y - (to_pos[1] - MAP_SIZE // 2) * cell_size + cell_size // 2
                
                if (0 <= from_x < display_size and 0 <= from_y < display_size and
                    0 <= to_x < display_size and 0 <= to_y < display_size):
                    # Draw path line
                    cv2.line(self.map_display, 
                            (int(from_x), int(from_y)), 
                            (int(to_x), int(to_y)), 
                            (0, 120, 255), 2)  # Orange line
        
        # Draw decision points - FLIPPED Y-AXIS
        for pos in self.decision_points:
            x, y = pos
            screen_x = center_x + (x - MAP_SIZE // 2) * cell_size
            # Flip Y-axis here
            screen_y = center_y - (y - MAP_SIZE // 2) * cell_size
            
            if 0 <= screen_x < display_size - cell_size and 0 <= screen_y < display_size - cell_size:
                # Draw decision point with number of tried directions
                cv2.rectangle(self.map_display,
                             (int(screen_x), int(screen_y)),
                             (int(screen_x + cell_size), int(screen_y + cell_size)),
                             (200, 0, 200), -1)  # Filled purple for decision points
                
                # Add number if there's enough space
                if cell_size >= 10:
                    # Fix the tried/open count display
                    if isinstance(self.decision_points[pos], dict):
                        tried_count = len(self.decision_points[pos]["tried"])
                        open_count = len(self.decision_points[pos]["open"])
                        cv2.putText(self.map_display, f"{tried_count}/{open_count}", 
                                   (int(screen_x + 1), int(screen_y + cell_size - 2)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        tried_count = len(self.decision_points[pos])
                        cv2.putText(self.map_display, f"{tried_count}", 
                                   (int(screen_x + 2), int(screen_y + cell_size - 2)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw current position and orientation - FLIPPED Y-AXIS
        agent_x = center_x + (self.pos_x - MAP_SIZE // 2) * cell_size + cell_size // 2
        # Flip Y-axis here
        agent_y = center_y - (self.pos_y - MAP_SIZE // 2) * cell_size + cell_size // 2
        
        if 0 <= agent_x < display_size and 0 <= agent_y < display_size:
            # Draw agent position
            cv2.circle(self.map_display, (int(agent_x), int(agent_y)), 
                      cell_size//2 + 2, (0, 255, 255), -1)
            
            # Draw orientation indicator - need to adjust angle for flipped Y
            angle_rad = math.radians(self.orientation)
            # For flipped Y, cos remains the same but sin needs to be negated
            end_x = int(agent_x + cell_size * 1.5 * math.cos(angle_rad))
            end_y = int(agent_y - cell_size * 1.5 * math.sin(angle_rad))  # Note the negative sign here
            
            if 0 <= end_x < display_size and 0 <= end_y < display_size:
                cv2.line(self.map_display, (int(agent_x), int(agent_y)), 
                        (end_x, end_y), (255, 0, 0), 2)
        
        # Rest of visualization code remains the same
        # Add text info at top
        cv2.putText(self.map_display, f"State: {self.state}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.map_display, f"Pos: ({self.pos_x},{self.pos_y}) Dir: {self.orientation}°", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.map_display, f"DPs: {len(self.decision_points)} Visited: {len(self.visited)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add a simple legend
        legend_y = display_size - 80
        cv2.putText(self.map_display, "🟩 Visited", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 0), 1)
        cv2.putText(self.map_display, "🟧 Path", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
        cv2.putText(self.map_display, "🟪 Decision Pt", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
        cv2.putText(self.map_display, "🟡 Agent", (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return self.map_display

    def navigate_to_target(self):
        """Navigate toward the target position."""
        # Calculate direction to target
        dx = self.target_x - self.doom_x
        dy = self.target_y - self.doom_y
        
        # Calculate angle to target in radians
        target_angle = math.atan2(dy, dx)
        # Convert to degrees (0-360 range)
        target_angle_deg = (math.degrees(target_angle) + 360) % 360
        
        # Calculate the angle difference between current orientation and target direction
        angle_diff = target_angle_deg - self.orientation
        # Normalize to -180 to 180 range
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
        
        print(f"Target direction: {target_angle_deg:.1f}°, Current: {self.orientation:.1f}°, Diff: {angle_diff:.1f}°")
        
        # If we're facing approximately the right direction
        if abs(angle_diff) < 10:
            # Move forward toward target
            print("Facing toward target, moving forward")
            return [False, False, True, False, 0]
        else:
            # Turn to face target
            print(f"Turning {angle_diff:.1f}° to face target")
            return [False, False, False, False, angle_diff]

if __name__ == "__main__":
    # Create all windows with proper flags to make them insensitive to mouse
    cv2.namedWindow('ViZDoom Depth Buffer', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('ViZDoom Labels Buffer', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('ViZDoom Screen', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Depth Analysis', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Decision Process', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Exploration Map', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    
    # Optional: Position windows in a logical arrangement
    cv2.moveWindow('ViZDoom Screen', 0, 0)
    cv2.moveWindow('ViZDoom Depth Buffer', 0, 500)
    cv2.moveWindow('Depth Analysis', 650, 0)
    cv2.moveWindow('Exploration Map', 650, 500)
    cv2.moveWindow('Decision Process', 1100, 300)
    
    # Rest of your initialization code...
    """
    ############################################################################################################################################################
    These are pre-set configurations for level2 of the task, please dont change them

    ############################################################################################################################################################
    """
    
    parser = ArgumentParser("ViZDoom example showing different buffers (screen, depth, labels).")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)

    # OpenCV uses a BGR colorspace by default.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Sets resolution for all buffers.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Disables buffer with top down map of he current episode/level .
    game.set_automap_buffer_enabled(False)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    #game.set_render_hud(True)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)

    """
    ##############################################################################################################################################################
    Feel free to change anything after this
    ##############################################################################################################################################################
    """
    #uncomment this if you want to play the game with keyboard controls
    #game.set_mode(vzd.Mode.SPECTATOR)

    
    #The buttons you can use to make actionns currently are:
    #MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT_RIGHT_DELTA
    #setting available buttons here:
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.TURN_LEFT_RIGHT_DELTA])

    #check this link for all available buttons, use any you find useful: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Types.md#button


    #The state variables which you get from the game currently are:
    #POSITION_X, POSITION_Y, ANGLE
    #setting available game variables here:
    game.set_available_game_variables([vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])

    #check this link for all available game variables, use any you find useful: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Types.md#gamevariable

    game.init()

    #action to the game is given through a list of values for each of the buttons available, given below are a list of possible actions you could give to the
    #game currently
    episodes = 10
    sleep_time = 0.028

    # Add these variables right before the main game loop
    paused = False
    pause_message = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(pause_message, "PAUSED - Press 'P' to resume", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(pause_message, "Press 'Q' to quit", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# Create exploration agent
    agent = GridMazeExplorer()

# Main game loop
    for i in range(episodes):
        print("Episode #" + str(i + 1))
        game.new_episode()

        # At the start of main game loop, add:
        move_counter = 0  # Counter for implementation of longer first moves

        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                continue

        # Get depth buffer and screen buffer
            depth_buffer = state.depth_buffer
            screen_buffer = state.screen_buffer
            labels_buffer = state.labels_buffer
        
        # Display buffers
            if depth_buffer is not None:
                cv2.imshow('ViZDoom Depth Buffer', depth_buffer)
            if labels_buffer is not None:
                cv2.imshow('ViZDoom Labels Buffer', labels_buffer)
            if screen_buffer is not None:
                cv2.imshow('ViZDoom Screen', screen_buffer)
            
        # Get position information
            if state.game_variables is not None and len(state.game_variables) >= 3:
                doom_x = state.game_variables[0]
                doom_y = state.game_variables[1]
                angle = state.game_variables[2]
            
            # Update agent's position
                agent.update_position(doom_x, doom_y, angle)
            
        # Analyze depth buffer
            depth_info = agent.analyze_depth_buffer(depth_buffer)
        
        # Create and show exploration map visualization
            map_vis = agent.visualize()
            cv2.imshow('Exploration Map', map_vis)
        
        # Decide on action based on exploration strategy
            action = agent.decide_action(depth_info, game)
        
        # Replace the key handling section
            key = cv2.waitKey(1 if not paused else 0)  # Non-blocking or blocking based on pause state
        

            if key == ord('p') or key == ord('P'):  # Toggle pause with 'P' key
                paused = not paused
                print("GAME " + ("PAUSED" if paused else "RESUMED"))
    # Show/hide pause message
                if paused:
                    cv2.imshow('Pause Status', pause_message)
                else:
                    cv2.destroyWindow('Pause Status')

        # Replace the movement execution code in the main game loop with this

        # For forward/backward movements, move exactly 60 units in coordinate space
            if action[2] or action[3]:  # If moving forward or backward
                # Get starting position
                start_x, start_y = doom_x, doom_y
                
                # Calculate target position (60 units away in current direction)
                angle_rad = math.radians(angle)
                if action[2]:  # Forward movement
                    target_x = start_x + 60 * math.cos(angle_rad)
                    target_y = start_y + 60 * math.sin(angle_rad)
                    
                    # NEW PATH CENTERING CODE - analyze depth buffer to center on open path
                    if depth_buffer is not None:
                        h, w = depth_buffer.shape
                        # Find column with maximum depth value (most open space)
                        middle_slice = depth_buffer[h//3:2*h//3, :]  # Use middle third for stability
                        col_sums = np.sum(middle_slice, axis=0)  # Sum depth values in each column
                        max_col = np.argmax(col_sums)  # Find column with maximum depth
                        center_col = w // 2
                        
                        # Calculate adjustment based on distance from center
                        offset = max_col - center_col
                        if abs(offset) > w//5:  # Only adjust if significantly off-center
                            # Check if strafe direction has enough open space
                            left_quarter = depth_buffer[:, :w//4]
                            right_quarter = depth_buffer[:, 3*w//4:]
                            left_max_depth = np.max(left_quarter)
                            right_max_depth = np.max(right_quarter)
                            
                            # Minimum depth for safe strafing
                            min_safe_depth = 0.05
                            
                            if offset > 0 and right_max_depth > min_safe_depth:  # Max depth on right AND path open
                                move_action = [False, True, True, False, 0]  # Right + Forward
                                print(f"Centering: Max depth at column {max_col}, strafing RIGHT (depth: {right_max_depth:.3f})")
                            elif offset < 0 and left_max_depth > min_safe_depth:  # Max depth on left AND path open
                                move_action = [True, False, True, False, 0]  # Left + Forward
                                print(f"Centering: Max depth at column {max_col}, strafing LEFT (depth: {left_max_depth:.3f})")
                            else:
                                # Path not open enough in strafe direction, forward only
                                move_action = [False, False, False, True, 0]
                                print(f"Center offset detected but strafe unsafe (L:{left_max_depth:.3f}, R:{right_max_depth:.3f})")
                        else:
                            move_action = [False, False, True, False, 0]  # Standard forward when centered
                    else:
                        move_action = [False, False, True, False, 0]  # Default forward if no depth data
                else:  # Backward movement
                    target_x = start_x - 60 * math.cos(angle_rad)
                    target_y = start_y - 60 * math.sin(angle_rad)
                    move_action = [False, False, False, True, 0]
                
                print(f"Moving from ({start_x:.1f}, {start_y:.1f}) to ({target_x:.1f}, {target_y:.1f})")
                
                # Keep moving until we're close enough to the target
                movement_tries = 0
                max_tries = 15  # Safety limit to prevent infinite loop
                
                while movement_tries < max_tries:
                    # Make the movement action
                    game.make_action(move_action)
                    
                    # MISSING LINE: Increment movement_tries counter
                    movement_tries += 1  # Add this line!
                    
                    # Get new state
                    new_state = game.get_state()
                    if new_state and new_state.game_variables is not None:
                        current_x = new_state.game_variables[0]
                        current_y = new_state.game_variables[1]
                        
                        # Calculate distance to target
                        distance_to_target = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
                        
                        # If we're close enough to target, stop
                        if distance_to_target < 5.0:  # 5 unit threshold
                            print(f"Reached target position: ({current_x:.1f}, {current_y:.1f})")
                            break
                            
                        # If we haven't moved much, we might be stuck
                        distance_moved = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
                        if movement_tries > 5 and distance_moved < 10:
                            print("Movement appears to be blocked, analyzing depth buffer to find way around...")
                            
                            # Get current depth buffer
                            current_depth = new_state.depth_buffer
                            if current_depth is not None:
                                # Split buffer into left and right halves
                                h, w = current_depth.shape
                                left_half = current_depth[:, :w//3]
                                right_half = current_depth[:, 2*w//3:]
                                
                                # Calculate maximum depth for each half
                                left_max = np.max(left_half)
                                right_max = np.max(right_half)
                                
                                print(f"Left max depth: {left_max:.2f}, Right max depth: {right_max:.2f}")
                                
                                # Check if both sides are blocked
                                if left_max < 0.01 and right_max < 0.01:
                                    print("Both sides completely blocked. Giving up and backtracking.")
                                    agent.state = "BACKTRACKING"  # Force agent to backtrack
                                    agent.stuck_counter = 0  # Reset the stuck counter
                                    break  # Exit the movement loop
                                
                                # Choose direction with brighter (higher) depth readings
                                if left_max >= right_max:
                                    print("Left side appears more open, moving left")
                                    # Move left until progress is made
                                    side_move_action = [True, False, True, False, 0]
                                    side_rev = [False, True, True, False, 0]
                                    counter = 0
                                else:
                                    print("Right side appears more open, moving right")
                                    # Move right until progress is made
                                    side_move_action = [False, True, True, False, 0]
                                    side_rev = [True, False, True, False, 0]
                                    counter = 0
                                
                                # Execute side movement for a few steps
                                side_steps = 0
                                last_side_pos = (current_x, current_y)
                                
                                while side_steps < 10:  # Try up to 10 steps sideways
                                    game.make_action(side_move_action)
                                    side_state = game.get_state()
                                    
                                    if side_state and side_state.game_variables is not None:
                                        side_x = side_state.game_variables[0]
                                        side_y = side_state.game_variables[1]
                                        
                                        # Check if we're making progress
                                        side_distance = math.sqrt((side_x - last_side_pos[0])**2 + 
                                                               (side_y - last_side_pos[1])**2)
                                        
                                        if side_distance > 5.0:
                                            print(f"Made progress with side movement: {side_distance:.2f} units")
                                            counter = 1
                                            break
                                        
                                        last_side_pos = (side_x, side_y)
                                    
                                    side_steps += 1
                                    
                                    # Update visualizations
                                    if side_state:
                                        if side_state.depth_buffer is not None:
                                            cv2.imshow('ViZDoom Depth Buffer', side_state.depth_buffer)
                                        if side_state.screen_buffer is not None:
                                            cv2.imshow('ViZDoom Screen', side_state.screen_buffer)
                                        cv2.waitKey(1)


                                while side_steps < 15 and counter == 0:  # Try up to 10 steps sideways
                                    game.make_action(side_rev)
                                    side_state = game.get_state()
                                    
                                    if side_state and side_state.game_variables is not None:
                                        side_x = side_state.game_variables[0]
                                        side_y = side_state.game_variables[1]
                                        
                                        # Check if we're making progress
                                        side_distance = math.sqrt((side_x - last_side_pos[0])**2 + 
                                                               (side_y - last_side_pos[1])**2)
                                        
                                        if side_distance > 5.0:
                                            print(f"Made progress with side movement: {side_distance:.2f} units")
                                            counter = 1
                                            break
                                        
                                        last_side_pos = (side_x, side_y)
                                    
                                    side_steps += 1
                                    
                                    # Update visualizations
                                    if side_state:
                                        if side_state.depth_buffer is not None:
                                            cv2.imshow('ViZDoom Depth Buffer', side_state.depth_buffer)
                                        if side_state.screen_buffer is not None:
                                            cv2.imshow('ViZDoom Screen', side_state.screen_buffer)
                                        cv2.waitKey(1)
                                
                                # Resume forward movement after side maneuver
                                print("Resuming forward movement after side maneuver")
                            else:
                                print("Movement appears to be blocked, stopping movement attempt")
                                break
                    # Render frames during movement for visualization
                    if new_state:
                        # Update visualization while moving
                        if new_state.depth_buffer is not None:
                            cv2.imshow('ViZDoom Depth Buffer', new_state.depth_buffer)
                        if new_state.screen_buffer is not None:
                            cv2.imshow('ViZDoom Screen', new_state.screen_buffer)
                        cv2.waitKey(1)
                
                # Update agent position after the movement
                if game.get_state() and game.get_state().game_variables is not None:
                    agent.update_position(
                        game.get_state().game_variables[0], 
                        game.get_state().game_variables[1], 
                        game.get_state().game_variables[2]
                    )
            else:
                # For turning actions, just use a single action
                game.make_action(action)
                
                # Update agent position after turning
                if game.get_state() and game.get_state().game_variables is not None:
                    agent.update_position(
                        game.get_state().game_variables[0], 
                        game.get_state().game_variables[1], 
                        game.get_state().game_variables[2]
                    )
        
        # Debug info
            print(f"Position: ({doom_x:.1f}, {doom_y:.1f}), Angle: {angle:.1f}")
            print(f"Action: {action}")

    print("Episode finished!")
    print("************************")

cv2.destroyAllWindows()