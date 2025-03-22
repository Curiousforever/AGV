from argparse import ArgumentParser
import os
from random import choice
import vizdoom as vzd
import numpy as np
import cv2
import math
import random
import matplotlib.pyplot as plt

# Default configuration
DEFAULT_CONFIG = "../../scenarios/level1.cfg"

# Coordinate conversion functions
def doom_to_pixel(doom_x, doom_y, map_center_x, map_center_y, scale_factor):
    """Convert Doom game coordinates to pixel coordinates on the map image."""
    pixel_x = map_center_x + int(doom_x / scale_factor)
    pixel_y = map_center_y - int(doom_y / scale_factor)
    return pixel_x, pixel_y

def pixel_to_doom(pixel_x, pixel_y, map_center_x, map_center_y, scale_factor):
    """Convert pixel coordinates on the map image to Doom game coordinates."""
    doom_x = (pixel_x - map_center_x) * scale_factor
    doom_y = (map_center_y - pixel_y) * scale_factor
    return doom_x, doom_y


# Simple Pure Pursuit controller for path following
class SimplePurePursuitController:
    def __init__(self, path, lookahead_distance=25):
        # Keep existing initialization
        self.path = path
        self.lookahead_distance = lookahead_distance
        self.control_state = "FOLLOW"
        self.last_target = None
        self.closest_point_on_path = None
        
        # Add damping parameters
        self.previous_turn_value = 0
        self.damping_factor = 0.7  # Higher values = smoother movement

    def update(self, x, y, angle_degrees):
        # Find target point at lookahead distance
        target = self.find_lookahead_point(x, y)
        if target is None:
            return [False, False, False, False, 0]
            
        # Store target for visualization
        self.last_target = target
            
        # Calculate angle to target point - KEEPING THE NEGATIVE SIGN
        dx = target[0] - x
        dy = target[1] - y
        target_angle = math.degrees(math.atan2(dy, dx))
        target_angle = -target_angle  # Keep the negation as requested
        
        # Calculate steering angle
        angle_diff = target_angle - angle_degrees
        
        # NEW: Normalize to -180 to 180 degrees instead of 0 to 360
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
        
        # Calculate raw turn value
        raw_turn_value = int(angle_diff * 0.1)  # Reduced from 0.2 to 0.1 for stability
        
        # Apply damping to smooth steering and prevent oscillation
        damped_turn_value = raw_turn_value * (1 - self.damping_factor) + self.previous_turn_value * self.damping_factor
        turn_value = int(damped_turn_value)
        self.previous_turn_value = damped_turn_value
        
        # Limit maximum turning rate
        turn_value = max(-8, min(8, turn_value))
        turn_value = -turn_value
        
        # Move forward if roughly facing the right direction
        move_forward = abs(angle_diff) < 60
        
        # Debug output
        print(f"Target: {target_angle:.1f}°, Current: {angle_degrees:.1f}°, Diff: {angle_diff:.1f}°, Turn: {turn_value}")
        
        return [False, False, move_forward, False, turn_value]
        
    def find_lookahead_point(self, x, y):
        """
        Find a point on the path that is lookahead_distance away.
        """
        if not self.path or len(self.path) < 2:
            return None
            
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(self.path):
            dist = math.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Store closest point for visualization
        self.closest_point_on_path = self.path[closest_idx]
        
        # Look ahead from closest point
        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            point = self.path[i]
            dist = math.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist >= self.lookahead_distance:
                target_idx = i
                break
                
        # If we reached the end without finding a point at lookahead distance
        if target_idx >= len(self.path) - 1:
            target_idx = len(self.path) - 1
        
        # Store index for visualization    
        self.target_idx = target_idx
            
        return self.path[target_idx]

def visualize_path(map_image, path, start_pos, goal_pos, filename="planned_path.png"):
    """Visualize and save the planned path."""
    map_color = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    
    # Draw the path
    for i in range(len(path) - 1):
        start_point = (int(path[i][0]), int(path[i][1]))
        end_point = (int(path[i + 1][0]), int(path[i + 1][1]))
        cv2.line(map_color, start_point, end_point, (0, 255, 0), 2)
    
    # Draw start and goal
    cv2.circle(map_color, (int(start_pos[0]), int(start_pos[1])), 5, (0, 0, 255), -1)
    cv2.circle(map_color, (int(goal_pos[0]), int(goal_pos[1])), 5, (255, 0, 0), -1)
    
    cv2.imwrite(filename, map_color)
    return map_color

def load_trajectory_from_csv(filename='planned_trajectory.csv'):
    """Load a pre-planned trajectory from CSV file."""
    try:
        # Load the CSV file with numpy
        path_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        if path_data.size > 0:
            print(f"Successfully loaded trajectory with {len(path_data)} waypoints")
            return path_data.tolist()  # Convert to list of points
        else:
            print("Warning: CSV file contained no valid trajectory points")
            return None
    except Exception as e:
        print(f"Error loading trajectory from CSV: {e}")
        return None

if __name__ == "__main__":
    """
    ############################################################################################################################################################
    These are pre-set configurations for level1 and level2 of the task, please dont change them
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

    # Enables buffer with top down map of he current episode/level .
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    #game.set_render_hud(True)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)

    #entire map is shown
    game.add_game_args("+viz_am_center 1")

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
    
    # Load map image for planning
    map_image = cv2.imread('map_full.png', cv2.IMREAD_GRAYSCALE)
    if map_image is None:
        raise FileNotFoundError("Map image not found. Please check the file path.")

    # Define start and goal positions in pixel coordinates
    initial_position = (450, 214)  # White pixel
    goal_position = (492, 367)     # Blue skull
    
    # Map calibration parameters for coordinate conversion
    map_center_x = 449  # Approximate map center X
    map_center_y = 214  # Approximate map center Y
    scale_factor = 11.5  # Scale factor between game and pixel coordinates
    
        # Try to load the trajectory first (if file exists from previous planning)
    loaded_path = load_trajectory_from_csv('planned_trajectory2.csv')
        
    path = loaded_path
        # Create visualization for loaded path
    path_image = visualize_path(map_image, path, initial_position, goal_position, 
                                    filename="loaded_trajectory.png")
    cv2.imshow("Loaded Path", path_image)
    cv2.waitKey(1000)  # Display for 1 second

    controller = SimplePurePursuitController(path, lookahead_distance=25)
    
    # Quick turning phase to ensure agent is facing the right direction
    print("Starting agent orientation calibration...")
    game.new_episode()

    # Get initial position
    state = game.get_state()
    if state:
        doom_x = state.game_variables[0]
        doom_y = state.game_variables[1]
        angle = state.game_variables[2]
        map_x, map_y = doom_to_pixel(doom_x, doom_y, map_center_x, map_center_y, scale_factor)
        
        # Calculate angle to first path segment
        if len(path) > 1:
            path_dx = path[1][0] - map_x
            path_dy = path[1][1] - map_y
            path_angle = math.degrees(math.atan2(path_dy, path_dx))
            path_angle = -path_angle
            
            # Calculate how much we need to turn - NO INVERSION NEEDED
            angle_diff = path_angle - angle
            while angle_diff < -180: angle_diff += 360
            while angle_diff > 180: angle_diff -= 360
            
            print(f"Initial calibration: Agent angle={angle:.1f}, Path angle={path_angle:.1f}, Diff={angle_diff:.1f}")
            
            # If we need to make a large turn, do it now
            if abs(angle_diff) > 90:
                print("Initial calibration: Performing pre-emptive turn...")
                # Calculate turn direction and duration
                turn_dir = 10 if angle_diff > 0 else -10
                turn_frames = min(int(abs(angle_diff) / 10) + 5, 20)  # Cap at 20 frames
                
                # Execute turn
                for _ in range(turn_frames):
                    game.make_action([False, False, False, False, turn_dir])
                    cv2.waitKey(50)  # Small delay between turns

    game.new_episode()  # Start fresh after calibration
    
    episodes = 1
    sleep_time = 0.028
    
    
    for i in range(episodes):
        print(f"Episode #{i+1}")
        game.new_episode()
        
        # Record actual trajectory for visualization
        actual_trajectory = []
        
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                continue
            
            # Get current position and orientation
            doom_x = state.game_variables[0]
            doom_y = state.game_variables[1]
            angle = state.game_variables[2]
            
            # Convert game coordinates to map coordinates
            map_x, map_y = doom_to_pixel(doom_x, doom_y, map_center_x, map_center_y, scale_factor)
            
            # Record current position
            actual_trajectory.append((map_x, map_y))
            
            # Get action from controller
            action = controller.update(map_x, map_y, angle)
            
            # Replace the visualization section with this simplified version
            # Display live visualization with feedback indicators
            path_vis = path_image.copy()
            cv2.circle(path_vis, (int(map_x), int(map_y)), 5, (0, 255, 255), -1)  # Current position

            # Show the closest point on path (cross-track error visualization)
            if hasattr(controller, 'closest_point_on_path') and controller.closest_point_on_path:
                proj_x, proj_y = controller.closest_point_on_path
                cv2.line(path_vis, (int(map_x), int(map_y)), (int(proj_x), int(proj_y)), 
                        (0, 0, 255), 2)  # Red line shows deviation from path
                cv2.circle(path_vis, (int(proj_x), int(proj_y)), 3, (0, 0, 255), -1)

            # Show lookahead target point
            if hasattr(controller, 'last_target') and controller.last_target:
                target_x, target_y = controller.last_target
                cv2.circle(path_vis, (int(target_x), int(target_y)), 7, (255, 0, 255), 2)
                cv2.line(path_vis, (int(map_x), int(map_y)), (int(target_x), int(target_y)), 
                        (255, 0, 255), 1)

            # Show controller mode
            cv2.putText(path_vis, "Mode: Simple Pure Pursuit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Show lookahead distance
            cv2.putText(path_vis, f"Lookahead: {controller.lookahead_distance}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Path Following", path_vis)
            
            # Show automap
            automap = state.automap_buffer
            if automap is not None:
                cv2.imshow('ViZDoom Map', automap)
            
            # Apply action to the game
            try:
                # Validate action format before execution
                action_to_execute = [
                    bool(action[0]),  # MOVE_LEFT
                    bool(action[1]),  # MOVE_RIGHT
                    bool(action[2]),  # MOVE_FORWARD
                    bool(action[3]),  # MOVE_BACKWARD
                    max(-15, min(15, int(action[4])))  # TURN_LEFT_RIGHT_DELTA (clamped)
                ]
                game.make_action(action_to_execute)
            except Exception as e:
                print(f"Action execution error: {e}")
                print(f"Attempted action: {action}")
                # Try recovery with a safe action
                try:
                    game.make_action([False, False, False, False, 0])
                except:
                    print("Recovery failed, continuing to next frame")
            
            # Debug info
            print(f"Pos: ({doom_x:.1f}, {doom_y:.1f}), Map: ({map_x:.1f}, {map_y:.1f}), Angle: {angle:.1f}")
            print(f"Action: {action}")
            
            cv2.waitKey(int(sleep_time * 1000))
        
        print("Episode finished!")
        
        # Visualize the actual trajectory vs planned path
        if actual_trajectory:
            final_vis = path_image.copy()
            # Draw the actual trajectory
            for i in range(len(actual_trajectory) - 1):
                start_point = (int(actual_trajectory[i][0]), int(actual_trajectory[i][1]))
                end_point = (int(actual_trajectory[i+1][0]), int(actual_trajectory[i+1][1]))
                cv2.line(final_vis, start_point, end_point, (255, 0, 0), 2)
            
            # Save the final visualization
            cv2.imwrite("actual_trajectory.png", final_vis)
            cv2.imshow("Final Trajectory", final_vis)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()