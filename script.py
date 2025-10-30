import time
import math
import threading
import argparse
import turtle
import numpy as np
import airsim
from astar import astar 

CELL_SIZE_PIXELS = 15
METERS_PER_CELL = 0.5  # High resolution: each cell is 0.5 meters
TAKEOFF_ALT = 3.0  # Higher altitude for better LiDAR coverage and safety
REACH_THRESH_M = 0.25  # Tighter threshold for 0.5m cells
UNKNOWN_CELL = -1
FREE_CELL = 0
DISEASE_CELL = 1
OBSTACLE_CELL = 2

# LiDAR detection parameters
LIDAR_RANGE = 8.0  # meters - scan further ahead
OBSTACLE_THRESHOLD = 2.0  # meters - more conservative obstacle detection
DISEASE_DETECTION_RANGE = 1.0  # meters - range to detect disease
LIDAR_MIN_HEIGHT = -1.0  # relative to drone, filter ground points
LIDAR_MAX_HEIGHT = 4.0  # relative to drone, filter high points
OBSTACLE_BUFFER_CELLS = 2  # Add buffer around detected obstacles

DISEASE_CELL_CORD = [
    # Bottom-left patch (scaled for 0.5m cells)
    (10, 16), (11, 16), (12, 16), (13, 16),
    (10, 17), (11, 17), (12, 17), (13, 17),
    (10, 18), (11, 18), (12, 18), (13, 18),

    # Center-left patch
    (36, 30), (36, 31), (36, 32), (36, 33),
    (37, 30), (37, 31), (37, 32), (37, 33),
    (38, 30), (38, 31), (38, 32), (38, 33),

    # Top-right patch (north-east)
    (76, 80), (77, 80), (78, 80), (79, 80),
    (76, 81), (77, 81), (78, 81), (79, 81),
    (76, 82), (77, 82), (78, 82), (79, 82),

    # Scattered infection line (like crop row disease)
    (60, 50), (61, 50), (62, 51), (63, 51),
    (64, 52), (65, 52), (66, 53), (67, 53)
]

class Visualizer:
    def __init__(self, grid_size, drone_start):
        self.grid_size = grid_size
        self.obstacles = set()
        self.diseases = set()
        self.drone_pos = drone_start

        self.screen = turtle.Screen()
        self.screen.title(f"AirSim Drone - Grid: {grid_size}x{grid_size} @ 0.5m/cell (25m x 25m area)")
        self.screen.setup(width=1000, height=1000)
        self.screen.tracer(0)

        self.drawer = turtle.Turtle()
        self.drawer.hideturtle()
        self.drawer.speed(0)
        self.drawer.penup()

        self._draw_grid_background()

        self.draw_cell(drone_start[0], drone_start[1], "blue")

        self.drone_t = turtle.Turtle()
        self.drone_t.shape("square")
        scale = CELL_SIZE_PIXELS / 20
        self.drone_t.shapesize(stretch_wid=scale, stretch_len=scale)
        self.drone_t.color("red")
        self.drone_t.penup()
        self.drone_t.speed(1)
        self.move_drone(drone_start[0], drone_start[1])

    def _draw_grid_background(self):
        self.drawer.color("lightgrey")
        start_x = -(self.grid_size * CELL_SIZE_PIXELS) / 2
        start_y = -(self.grid_size * CELL_SIZE_PIXELS) / 2
        for i in range(self.grid_size + 1):
            gx = start_x + i * CELL_SIZE_PIXELS
            self.drawer.goto(gx, start_y)
            self.drawer.setheading(90)
            self.drawer.pendown()
            self.drawer.forward(self.grid_size * CELL_SIZE_PIXELS)
            self.drawer.penup()
        for j in range(self.grid_size + 1):
            gy = start_y + j * CELL_SIZE_PIXELS
            self.drawer.goto(start_x, gy)
            self.drawer.setheading(0)
            self.drawer.pendown()
            self.drawer.forward(self.grid_size * CELL_SIZE_PIXELS)
            self.drawer.penup()
        self.drawer.color("black")
        self.screen.update()

    def grid_to_screen(self, x, y):
        start_x = -(self.grid_size * CELL_SIZE_PIXELS) / 2
        start_y = -(self.grid_size * CELL_SIZE_PIXELS) / 2
        sx = start_x + x * CELL_SIZE_PIXELS + CELL_SIZE_PIXELS / 2
        sy = start_y + y * CELL_SIZE_PIXELS + CELL_SIZE_PIXELS / 2
        return sx, sy

    def draw_cell(self, x, y, color):
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
        gx, gy = self.grid_to_screen(x, y)
        self.drawer.goto(gx - CELL_SIZE_PIXELS/2, gy - CELL_SIZE_PIXELS/2)
        self.drawer.fillcolor(color)
        self.drawer.begin_fill()
        for _ in range(4):
            self.drawer.forward(CELL_SIZE_PIXELS)
            self.drawer.left(90)
        self.drawer.end_fill()
        self.screen.update()

    def move_drone(self, x, y):
        gx, gy = self.grid_to_screen(x, y)
        self.drone_t.goto(gx, gy)
        self.screen.update()

    def mark_visited(self, x, y):
        if (x, y) not in self.diseases:
            self.draw_cell(x, y, "lightblue")

    def mark_obstacle(self, x, y):
        self.obstacles.add((x, y))
        self.draw_cell(x, y, "black")

    def mark_disease(self, x, y):
        self.diseases.add((x, y))
        self.draw_cell(x, y, "green")

def start_turtle_loop():
    turtle.mainloop()

class AirSimDroneController:
    def __init__(self, grid_size, drone_start, visualize=True):
        self.client = airsim.MultirotorClient(ip="10.138.185.234")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Enable LiDAR
        self.lidar_name = "LidarSensor1"
        
        self.grid_size = grid_size
        self.drone_start = drone_start
        self.visualize = visualize
        
        # Initialize grid map (drone's knowledge)
        self.drone_map = np.full((grid_size, grid_size), fill_value=UNKNOWN_CELL)
        self.drone_map[drone_start] = FREE_CELL
        
        # Ground truth for disease (simulated)
        self.disease_locations = set(DISEASE_CELL_CORD)
        
        self.home_position = None
        
    def grid_to_ned(self, cell):
        """Convert grid coordinates to NED (North-East-Down) coordinates"""
        x, y = cell
        north = y * METERS_PER_CELL
        east = x * METERS_PER_CELL
        down = -TAKEOFF_ALT  # negative becauseDown is negative up
        return north, east, down
    
    def ned_to_grid(self, north, east):
        """Convert NED coordinates to grid coordinates"""
        x = int(round(east / METERS_PER_CELL))
        y = int(round(north / METERS_PER_CELL))
        return x, y
    
    def takeoff(self):
        print("Taking off...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-TAKEOFF_ALT, 2).join()
        time.sleep(1)
        
        # Store home position
        pos = self.client.getMultirotorState().kinematics_estimated.position
        self.home_position = (pos.x_val, pos.y_val, pos.z_val)
        print(f"Home position (NED): {self.home_position}")
    
    def get_lidar_data(self):
        """Get LiDAR point cloud data"""
        try:
            lidar_data = self.client.getLidarData(self.lidar_name)
            if len(lidar_data.point_cloud) < 3:
                return []
            
            # Convert to numpy array and reshape
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 3)
            return points
        except Exception as e:
            print(f"LiDAR error: {e}")
            return []
    
    def detect_obstacles_from_lidar(self, current_cell):
        """Process LiDAR data to detect obstacles in nearby cells with safety buffer"""
        points = self.get_lidar_data()
        if len(points) == 0:
            return []
        
        obstacles = set()
        pos = self.client.getMultirotorState().kinematics_estimated.position
        
        for point in points:
            # Calculate distance from drone
            dist = np.linalg.norm(point[:2])  # Horizontal distance
            
            # Filter points by height relative to drone
            point_height_relative = point[2] - pos.z_val
            
            # Only consider points within detection range and at obstacle height
            if (dist < LIDAR_RANGE and 
                LIDAR_MIN_HEIGHT < point_height_relative < LIDAR_MAX_HEIGHT and
                dist > 0.1):  # Ignore very close points (noise)
                
                # Convert LiDAR point to grid coordinates
                # LiDAR gives relative position in body frame
                # For NED frame: x=forward(North), y=right(East), z=down
                world_north = pos.x_val + point[0]
                world_east = pos.y_val + point[1]
                
                grid_x, grid_y = self.ned_to_grid(world_north, world_east)
                
                # Check if within grid bounds
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # Mark cells with detected obstacles
                    if dist < OBSTACLE_THRESHOLD:
                        # Add the obstacle cell
                        obstacles.add((grid_x, grid_y))
                        
                        # Add buffer cells around obstacle for safety
                        for dx in range(-OBSTACLE_BUFFER_CELLS, OBSTACLE_BUFFER_CELLS + 1):
                            for dy in range(-OBSTACLE_BUFFER_CELLS, OBSTACLE_BUFFER_CELLS + 1):
                                buf_x, buf_y = grid_x + dx, grid_y + dy
                                if 0 <= buf_x < self.grid_size and 0 <= buf_y < self.grid_size:
                                    obstacles.add((buf_x, buf_y))
        
        return list(obstacles)
    
    def check_disease(self, cell):
        """Simulate disease detection (could use camera/multispectral sensor)"""
        x, y = cell
        # Simulate detection within range
        for disease_cell in self.disease_locations:
            dx = abs(disease_cell[0] - x)
            dy = abs(disease_cell[1] - y)
            if dx <= 1 and dy <= 1:  # Adjacent cells
                return disease_cell
        return None
    
    def scan_surroundings(self, current_cell, viz=None):
        """Scan surroundings using LiDAR and update map"""
        # Detect obstacles with LiDAR
        obstacles = self.detect_obstacles_from_lidar(current_cell)
        
        for obs in obstacles:
            if self.drone_map[obs] == UNKNOWN_CELL:
                self.drone_map[obs] = OBSTACLE_CELL
                if viz:
                    viz.mark_obstacle(obs[0], obs[1])
        
        # Check for diseases in current and adjacent cells
        disease_cell = self.check_disease(current_cell)
        if disease_cell:
            self.drone_map[disease_cell] = DISEASE_CELL
            if viz:
                viz.mark_disease(disease_cell[0], disease_cell[1])
        
        # Mark unexplored neighbors as free if no obstacle detected
        neighbors = self.get_neighbors(current_cell)
        for neighbor in neighbors:
            if self.drone_map[neighbor] == UNKNOWN_CELL:
                # Check if LiDAR detected this as obstacle
                if neighbor not in obstacles:
                    self.drone_map[neighbor] = FREE_CELL
    
    def get_neighbors(self, cell):
        """Get valid neighboring cells"""
        x, y = cell
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors
    
    def move_to_cell(self, target_cell):
        """Move drone to target grid cell"""
        north, east, down = self.grid_to_ned(target_cell)
        print(f"Moving to cell {target_cell} -> NED({north:.2f}, {east:.2f}, {down:.2f})")
        
        # Use slower velocity for higher precision with 0.5m cells
        self.client.moveToPositionAsync(north, east, down, velocity=1.5).join()
        time.sleep(0.15)
    
    def land_and_disarm(self):
        """Land the drone and disarm"""
        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

class PathFinder:
    @staticmethod
    def get_frontiers(drone_map, visited):
        """Find frontier cells (free cells adjacent to unknown cells)"""
        frontiers = set()
        grid_size = len(drone_map)
        
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in visited:
                    continue
                if drone_map[x, y] == FREE_CELL:
                    # Check if adjacent to unknown cell
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < grid_size and 0 <= ny < grid_size and
                            drone_map[nx, ny] == UNKNOWN_CELL):
                            frontiers.add((x, y))
                            break
        return frontiers
    
    @staticmethod
    def select_best_frontier(current_pos, frontiers, drone_map):
        """Select best frontier using distance and unexplored neighbors"""
        if not frontiers:
            return None
        
        scored = []
        for f in frontiers:
            dist = abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
            unexplored = sum(
                1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                if (0 <= f[0]+dx < len(drone_map) and 
                    0 <= f[1]+dy < len(drone_map) and
                    drone_map[f[0]+dx, f[1]+dy] == UNKNOWN_CELL)
            )
            scored.append((dist, unexplored, f))
        
        # Select frontier with good balance
        scored.sort(key=lambda x: (x[0], -x[1]))  # Min distance, max unexplored
        return scored[0][2]

def main(args):
    grid_size = args.grid
    drone_start = tuple(map(int, args.droneStart.split(',')))
    
    # Initialize AirSim controller
    controller = AirSimDroneController(grid_size, drone_start, visualize=not args.no_visualize)
    
    # Initialize visualizer
    viz = None
    if not args.no_visualize:
        viz = Visualizer(grid_size, drone_start)
        t = threading.Thread(target=start_turtle_loop, daemon=True)
        t.start()
        time.sleep(0.5)
    
    # Takeoff
    controller.takeoff()
    
    start_time = time.time()
    steps_taken = 0
    
    current_pos = drone_start
    visited = {current_pos}
    
    # Scan initial position
    controller.scan_surroundings(current_pos, viz)
    
    # Main exploration loop
    while True:
        # Get frontiers
        frontiers = PathFinder.get_frontiers(controller.drone_map, visited)
        
        if not frontiers:
            print("No more frontiers to explore!")
            break
        
        # Select best frontier
        target = PathFinder.select_best_frontier(current_pos, frontiers, controller.drone_map)
        
        if target is None:
            break
        
        # Plan path to target
        path = astar(current_pos, target, controller.drone_map, grid_size)
        
        if path is None or len(path) <= 1:
            print(f"No valid path to {target}")
            visited.add(target)  # Mark as visited to avoid trying again
            continue
        
        # Follow path
        for node in path[1:]:
            steps_taken += 1
            controller.move_to_cell(node)
            
            if viz:
                viz.move_drone(node[0], node[1])
                viz.mark_visited(node[0], node[1])
                time.sleep(0.1)
            
            # Scan surroundings at new position
            controller.scan_surroundings(node, viz)
            
            current_pos = node
            visited.add(node)
    
    end_time = time.time()
    print(f"\nExploration Complete!")
    print(f"Time taken: {(end_time - start_time):.2f} seconds")
    print(f"Steps taken: {steps_taken}")
    print(f"Cells visited: {len(visited)}")
    
    # Count detected diseases and obstacles
    diseases_found = np.sum(controller.drone_map == DISEASE_CELL)
    obstacles_found = np.sum(controller.drone_map == OBSTACLE_CELL)
    print(f"Diseases detected: {diseases_found}")
    print(f"Obstacles detected: {obstacles_found}")
    
    controller.land_and_disarm()
    print("Mission complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirSim Drone Exploration with LiDAR")
    parser.add_argument("--grid", type=int, default=50, help="Grid size")
    parser.add_argument("--droneStart", type=str, default="0,0", help="Starting position (x,y)")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    main(args)