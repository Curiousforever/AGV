import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

# Load the map image
map_image = cv2.imread('map_full.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if map_image is None:
    raise FileNotFoundError("Map image not found. Please check the file path.")

# Identify the initial and goal positions
initial_position = (450, 215) # White pixel coordinates (approximate)
goal_position = (492, 367)

# Node class representing a state in the space
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# RRT* algorithm
class RRTStar:
    def __init__(self, start, goal, step_size=15.0, max_iter = 100000, goal_bias=0.1):  # Increased from 10.0 to 30.0
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_size = (map_image.shape[1], map_image.shape[0])
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.goal_region_radius = 5  # Adjusted for the map scale
        self.path = None
        self.goal_reached = False
        self.goal_bias = goal_bias  # Add goal bias parameter
        self.search_radius = 50  # Define a search radius for finding neighbors

        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.setup_visualization()

    def setup_visualization(self):
        """Set up the visualization environment (grid, start, goal, obstacles)."""
        self.ax.imshow(map_image, cmap='gray')
        self.ax.plot(self.start.x, self.start.y, 'bo', label='Start')
        self.ax.plot(self.goal.x, self.goal.y, 'ro', label='Goal')
        self.ax.set_xlim(0, self.map_size[0])
        self.ax.set_ylim(self.map_size[1], 0)  # Inverted y-axis for image coordinates
        self.ax.grid(True)

    def plan(self):
        """Main RRT* planning loop."""
        for i in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rand_node)
            new_node = self.steer(nearest_node, rand_node)

            # Check if both the new node and the path to it are collision-free
            if self.is_collision_free(new_node) and self.check_path(nearest_node, new_node):
                # Find neighbors for rewiring
                neighbors = self.find_neighbors(new_node)
                
                # Choose the best parent
                new_node = self.choose_parent(neighbors, nearest_node, new_node)
                
                # Add the node to the tree
                self.node_list.append(new_node)
                
                # Rewire the tree
                self.rewire(new_node, neighbors)

            # Check if we've reached the goal
            if self.reached_goal(new_node):
                self.path = self.generate_final_path(new_node)
                self.goal_reached = True
                return

    def get_random_node(self):
        """Generate a random node in the map with goal bias."""
        if random.random() < self.goal_bias:
            return Node(self.goal.x, self.goal.y)
        else:
            rand_node = Node(
                random.uniform(0, self.map_size[0]),
                random.uniform(0, self.map_size[1])
            )
            return rand_node

    def steer(self, from_node, to_node):
        """Steer from one node to another, step-by-step."""
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(from_node.x + self.step_size * math.cos(theta),
                        from_node.y + self.step_size * math.sin(theta))
        new_node.cost = from_node.cost + self.step_size
        new_node.parent = from_node
        return new_node

    def is_collision_free(self, node1, node2=None):
        """
        Check if a node or path between nodes collides with obstacles.
        If node2 is None, checks single node. Otherwise checks path between nodes.
        Obstacles are represented by pixels with grayscale value close to #2f1b0b.
        """
        # Single node check
        if node2 is None:
            if (node1.x < 1 or node1.x >= (self.map_size[0]-1) or 
                node1.y < 1 or node1.y >= (self.map_size[1]-1)):
                return False
            
            x, y = int(node1.x), int(node1.y)
            obstacle_gray = int(0.299 * 0x2f + 0.587 * 0x1b + 0.114 * 0x0b)
            return not ((abs(int(map_image[y, x]) - obstacle_gray) <= 5) or (abs(int(map_image[y+1, x]) - obstacle_gray) <= 5) or (abs(int(map_image[y-1, x]) - obstacle_gray) <= 5) or (abs(int(map_image[y, x+1]) - obstacle_gray) <= 5) or (abs(int(map_image[y, x-1]) - obstacle_gray) <= 5))

        # Path check
        x1, y1 = int(node1.x), int(node1.y)
        x2, y2 = int(node2.x), int(node2.y)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            # Check current point
            if not self.is_collision_free(Node(x1, y1)):
                return False
            if x1 == x2 and y1 == y2:
                break
                
            # Bresenham's line algorithm
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return True

    def reached_goal(self, node):
        """Check if the goal has been reached."""
        return np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y]) < self.goal_region_radius

    def generate_final_path(self, goal_node):
        """Generate the final path from the start to the goal."""
        path = []
        node = goal_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]  # Reverse the path

    def get_nearest_node(self, node_list, rand_node):
        """Find the nearest node in the tree to the random node."""
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y]) for node in node_list]
        nearest_node = node_list[np.argmin(distances)]
        return nearest_node

    def draw_tree(self, node):
        """Draw a tree edge from the current node to its parent."""
        if node.parent:
            self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")

    def draw_path(self):
        """Draw the final path from start to goal."""
        if self.path:
            self.ax.plot([x[0] for x in self.path], [x[1] for x in self.path], '-g', label='Path')

    def check_path(self, node1, node2):

        x1, y1 = int(node1.x), int(node1.y)
        x2, y2 = int(node2.x), int(node2.y)
    
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
    
    # Define neighbor offsets (8-connectivity)
        neighbors = [(-1, -1), (0, -1), (1, -1),  # Top row
                    (-1, 0),           (1, 0),    # Middle row (excluding center)
                    (-1, 1),  (0, 1),  (1, 1)     # Bottom row
                    ]
    
        while True:
        # Check center pixel
            if not self.is_collision_free(Node(x1, y1)):
                return False
            
        # Check all 8 surrounding pixels
            for nx, ny in neighbors:
                if not self.is_collision_free(Node(x1 + nx, y1 + ny)):
                    return False
                
            if x1 == x2 and y1 == y2:
                break
            
        # Bresenham's line algorithm
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy     
        return True

    def find_neighbors(self, new_node, radius=None):
        """Find nearby nodes within the search radius."""
        if radius is None:
            radius = self.search_radius
        
        return [node for node in self.node_list
                if np.linalg.norm([node.x - new_node.x, node.y - new_node.y]) < radius]

    def choose_parent(self, neighbors, nearest_node, new_node):
        """Choose the best parent for the new node based on cost."""
        min_cost = nearest_node.cost + np.linalg.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y])
        best_node = nearest_node

        for neighbor in neighbors:
            # Calculate potential cost through this neighbor
            potential_cost = neighbor.cost + np.linalg.norm([new_node.x - neighbor.x, new_node.y - neighbor.y])
            
            # Check if this neighbor provides a lower-cost path AND the path is collision-free
            if potential_cost < min_cost and self.check_path(neighbor, new_node):
                best_node = neighbor
                min_cost = potential_cost

        # Update the new node with the best parent and cost
        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, neighbors):
        """Rewire the tree by checking if any neighbor should adopt the new node as a parent."""
        for neighbor in neighbors:
            # Calculate potential cost through the new node
            potential_cost = new_node.cost + np.linalg.norm([neighbor.x - new_node.x, neighbor.y - new_node.y])
            
            # If the new path is better AND collision-free, rewire
            if potential_cost < neighbor.cost and self.check_path(new_node, neighbor):
                neighbor.parent = new_node
                neighbor.cost = potential_cost

    def smooth_path(self, path, smoothing_factor=0.1):
        """Smooth the path using a moving average filter to make it more feasible for vehicle control."""
        if not path or len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]  # Keep the start point
        
        for i in range(1, len(path)-1):
            # Create a weighted average of current point and neighbors
            smoothed_x = (1-smoothing_factor) * path[i][0] + smoothing_factor * (path[i-1][0] + path[i+1][0])/2
            smoothed_y = (1-smoothing_factor) * path[i][1] + smoothing_factor * (path[i-1][1] + path[i+1][1])/2
            smoothed_path.append([smoothed_x, smoothed_y])
        
        smoothed_path.append(path[-1])  # Keep the goal point
        return smoothed_path

    def check_curvature(self, path, max_angle_change=30):
        """Check if the path has turns that are too sharp (measured in degrees)."""
        for i in range(1, len(path)-1):
            # Calculate vectors between consecutive points
            v1 = [path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]]
            v2 = [path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]]
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # Avoid division by zero
            if mag1 * mag2 == 0:
                continue
                
            angle = math.acos(dot_product/(mag1*mag2)) * 180/math.pi
            if angle > max_angle_change:
                return False
        return True

    def densify_path(self, path, desired_step=3.0):
        """
        Add intermediate waypoints to make the path have a smaller step size.
        
        Args:
            path: Original path as list of [x,y] waypoints
            desired_step: Desired distance between consecutive waypoints
            
        Returns:
            Dense path with additional waypoints
        """
        if not path or len(path) < 2:
            return path
            
        dense_path = [path[0]]  # Start with the first point
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # Calculate distance between current waypoints
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Calculate number of segments needed
            num_segments = max(1, int(dist / desired_step))
            
            # Create intermediate waypoints
            for j in range(1, num_segments):
                alpha = j / num_segments
                x = p1[0] + alpha * (p2[0] - p1[0])
                y = p1[1] + alpha * (p2[1] - p1[1])
                dense_path.append([x, y])
                
            # Add the endpoint (except for the last iteration)
            if i < len(path) - 2:
                dense_path.append(p2)
        
        # Always add the final destination point
        dense_path.append(path[-1])
        
        print(f"Original path length: {len(path)}, Dense path length: {len(dense_path)}")
        return dense_path

# Modify in the main execution
if __name__ == "__main__":
    start = [initial_position[0], initial_position[1]]
    goal = [goal_position[0], goal_position[1]]

    # Use smaller step size for more gradual turns
    rrt_star = RRTStar(start, goal, step_size=10.0, max_iter=100000, goal_bias=0.05)
    
    # Run the planning algorithm
    rrt_star.plan()
    
    # Apply path smoothing if a path is found
    if rrt_star.path:
        # First apply smoothing
        smoothed_path = rrt_star.smooth_path(rrt_star.path, smoothing_factor=0.2)
        
        # Then densify the path with smaller step size
        dense_path = rrt_star.densify_path(smoothed_path, desired_step=3.0)
        
        # Save the dense path instead
        rrt_star.path = dense_path

    # Create a color version of the map for visualization
    map_color = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

    # Draw the tree
    for node in rrt_star.node_list:
        if node.parent:
            start_point = (int(node.parent.x), int(node.parent.y))
            end_point = (int(node.x), int(node.y))
            cv2.line(map_color, start_point, end_point, (255, 0, 0), 1)  # Blue for tree

    # Draw the final path if found
    if rrt_star.path:
        for i in range(len(rrt_star.path) - 1):
            start_point = (int(rrt_star.path[i][0]), int(rrt_star.path[i][1]))
            end_point = (int(rrt_star.path[i + 1][0]), int(rrt_star.path[i + 1][1]))
            cv2.line(map_color, start_point, end_point, (0, 255, 0), 2)  # Green for path

    # Draw start and goal points
    cv2.circle(map_color, (start[0], start[1]), 5, (0, 0, 255), -1)  # Red for start
    cv2.circle(map_color, (goal[0], goal[1]), 5, (255, 0, 0), -1)  # Blue for goal

    # Display the result
    cv2.imshow('RRT* Path Planning', map_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Add this after visualization
    if rrt_star.path:
        # Save path to file for controller
        path_data = np.array(rrt_star.path)
        np.savetxt('planned_trajectory.csv', path_data, delimiter=',', header='x,y', comments='')
        print(f"Path saved to 'planned_trajectory.csv' with {len(rrt_star.path)} waypoints")