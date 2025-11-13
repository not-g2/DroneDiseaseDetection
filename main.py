import time
import math
import threading
import numpy as np
import airsim
from astar import astar 
import matplotlib.pyplot as plt
import os

CELL_SIZE_PIXELS = 5
METERS_PER_CELL = 0.5  
TAKEOFF_ALT = 3.0  
REACH_THRESH_M = 0.25  
UNKNOWN_CELL = -1
FREE_CELL = 0
DISEASE_CELL = 1
OBSTACLE_CELL = 2

LIDAR_RANGE = 8.0  
OBSTACLE_THRESHOLD = 2.0  
DISEASE_DETECTION_RANGE = 1.0  
LIDAR_MIN_HEIGHT = -1.0  
LIDAR_MAX_HEIGHT = 4.0  
OBSTACLE_BUFFER_CELLS = 1  

class AirSimDroneController:
    def __init__(self, grid_size, drone_start, vehicle_name, lidar_name, visualize=True, ip="127.0.0.1"):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()
        
        self.vehicle_name = vehicle_name
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        self.lidar_name = lidar_name
        self.grid_size = grid_size
        self.drone_start = drone_start
        self.visualize = visualize
        self.start_cell = drone_start
        self.origin_ned = None
        self.snapshots = []
        self.snap_freq = 50

        
        self.drone_map = np.full((grid_size, grid_size), fill_value=UNKNOWN_CELL, dtype=np.int8)
        
        sx, sy = drone_start
        self.drone_map[sy, sx] = FREE_CELL

        self.disease_locations = []
        self.home_position = None

    def grid_to_ned(self, cell):
        """Convert grid coordinates to NED coordinates relative to start position"""
        if self.origin_ned is None:
            raise RuntimeError("origin_ned not set; call takeoff() first to set origin")
        x_cell, y_cell = cell
        dx = (x_cell - self.start_cell[0]) * METERS_PER_CELL
        dy = (y_cell - self.start_cell[1]) * METERS_PER_CELL

        
        ned_x = self.origin_ned[0] + dy  
        ned_y = self.origin_ned[1] + dx  
        ned_z = -TAKEOFF_ALT
        return ned_x, ned_y, ned_z

    def ned_to_grid(self, north, east):
        """Convert world NED coords to grid coords anchored at origin_ned and start_cell"""
        if self.origin_ned is None:
            raise RuntimeError("origin_ned not set; call takeoff() first to set origin")
        
        dx_m = east - self.origin_ned[1]   
        dy_m = north - self.origin_ned[0]  

        x = int(round(dx_m / METERS_PER_CELL)) + self.start_cell[0]
        y = int(round(dy_m / METERS_PER_CELL)) + self.start_cell[1]
        return x, y

    def takeoff(self):
        print(f"{self.vehicle_name}: Taking off...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(-TAKEOFF_ALT, 2, vehicle_name=self.vehicle_name).join()
        time.sleep(1)

        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        self.home_position = (pos.x_val, pos.y_val, pos.z_val)
        
        self.origin_ned = (pos.x_val, pos.y_val, pos.z_val)
        print(f"{self.vehicle_name}: Home position (NED): {self.home_position}")

    def get_lidar_data(self):
        """Get LiDAR point cloud data (sensor-local points)."""
        try:
            lidar_data = self.client.getLidarData(lidar_name=self.lidar_name, vehicle_name=self.vehicle_name)
            if not hasattr(lidar_data, "point_cloud") or len(lidar_data.point_cloud) < 3:
                return np.zeros((0, 3), dtype=np.float32)
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            return points
        except Exception as e:
            print(f"{self.vehicle_name} LiDAR error: {e}")
            return np.zeros((0, 3), dtype=np.float32)

    def detect_obstacles_from_lidar(self, current_cell):
        """Process LiDAR data to detect obstacles in nearby cells with safety buffer"""
        points = self.get_lidar_data()
        if points.shape[0] == 0:
            return []

        obstacles = set()
        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position

        for point in points:
            horiz_dist = math.hypot(point[0], point[1])
            if horiz_dist < 1e-3 or horiz_dist > LIDAR_RANGE:
                continue

            if not (LIDAR_MIN_HEIGHT < point[2] < LIDAR_MAX_HEIGHT):
                continue

            world_north = pos.x_val + point[0]
            world_east = pos.y_val + point[1]

            grid_x, grid_y = self.ned_to_grid(world_north, world_east)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                if horiz_dist < OBSTACLE_THRESHOLD:
                    
                    for dx in range(-OBSTACLE_BUFFER_CELLS, OBSTACLE_BUFFER_CELLS + 1):
                        for dy in range(-OBSTACLE_BUFFER_CELLS, OBSTACLE_BUFFER_CELLS + 1):
                            bx, by = grid_x + dx, grid_y + dy
                            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                                obstacles.add((bx, by))
        
        return list(obstacles)

    def check_disease(self, cell):
        x, y = cell
        for disease_cell in self.disease_locations:
            dx = abs(disease_cell[0] - x)
            dy = abs(disease_cell[1] - y)
            if dx <= 1 and dy <= 1:
                return disease_cell
        return None

    def scan_surroundings(self, current_cell, viz=None, frontiers=None):
        """Scan surroundings using LiDAR and update map"""
        if frontiers is None:
            frontiers = set()

        obstacles = self.detect_obstacles_from_lidar(current_cell)
                
        for obs in obstacles:
            ox, oy = obs
            if self.drone_map[oy, ox] in (UNKNOWN_CELL, FREE_CELL):
                self.drone_map[oy, ox] = OBSTACLE_CELL
                if viz:
                    viz.mark_obstacle(ox, oy)

        disease_cell = self.check_disease(current_cell)
        if disease_cell:
            dx, dy = disease_cell
            self.drone_map[dy, dx] = DISEASE_CELL
            if viz:
                viz.mark_disease(dx, dy)

        neighbors = self.get_neighbors(current_cell)
        for neighbor in neighbors:
            nx, ny = neighbor
            if self.drone_map[ny, nx] == UNKNOWN_CELL:
                if neighbor not in obstacles:
                    dx = abs(nx - current_cell[0])
                    dy = abs(ny - current_cell[1])
                    if dx <= 1 and dy <= 1:
                        self.drone_map[ny, nx] = FREE_CELL
                        frontiers.add((nx, ny))
        return frontiers

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def move_to_cell(self, target_cell):
        
        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        current_grid = self.ned_to_grid(pos.x_val, pos.y_val)
        obstacles = self.detect_obstacles_from_lidar(current_grid)

        if target_cell in obstacles or self.drone_map[target_cell[1], target_cell[0]] == OBSTACLE_CELL:
            print(f"{self.vehicle_name}: WARNING: Target cell {target_cell} is blocked!")
            return False

        north, east, down = self.grid_to_ned(target_cell)
        
        self.client.moveToPositionAsync(north, east, down, velocity=1.0, vehicle_name=self.vehicle_name).join()
        time.sleep(0.2)
        return True

    def land_and_disarm(self):
        print(f"{self.vehicle_name}: Landing...")
        try:
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        self.client.enableApiControl(False, vehicle_name=self.vehicle_name)

class PathFinder:
    @staticmethod
    def next_step(cur_pos, frontiers, drone_map):
        """Select best frontier using Pareto optimization"""
        if not frontiers:
            return None, None

        grid_h, grid_w = drone_map.shape  
        scored = []

        disease_positions = np.argwhere(drone_map == DISEASE_CELL)  
        for f in frontiers:
            fx, fy = f
            dist = abs(fx - cur_pos[0]) + abs(fy - cur_pos[1])
            unexplored = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    if drone_map[ny, nx] == UNKNOWN_CELL:
                        unexplored += 1
            if disease_positions.size > 0:
                dxs = np.abs(disease_positions[:, 1] - cur_pos[0])  
                dys = np.abs(disease_positions[:, 0] - cur_pos[1])  
                manh = dxs + dys
                min_dist = np.min(manh)
            else:
                min_dist = float('inf')
            if min_dist <= 1:
                disease_proximity = 50.0
            elif min_dist < 5:
                disease_proximity = 10.0 / float(min_dist)
            else:
                disease_proximity = 1.0 / float(min_dist) if min_dist != float('inf') else 0.0

            scored.append((dist, unexplored, disease_proximity, f))

        front = PathFinder.pareto_front(scored)
        chosen = PathFinder.knee_by_utopia(front)
        if chosen is None:
            return None, None
        best_frontier = chosen[3]
        path = astar(cur_pos, best_frontier, drone_map, grid_w)
        return best_frontier, path

    @staticmethod
    def knee_by_utopia(pareto):
        if not pareto:
            return None
        min_dist = min(p[0] for p in pareto)
        max_unexplored = max(p[1] for p in pareto)
        max_disease_prox = max(p[2] for p in pareto)
        utopia = (min_dist, -max_unexplored, -max_disease_prox)
        best = None
        best_d = float('inf')
        for dist, unexplored, disease_prox, f in pareto:
            d = math.hypot(dist - utopia[0], (-unexplored) - utopia[1], disease_prox - utopia[2])
            if d < best_d:
                best_d = d
                best = (dist, unexplored, disease_prox, f)
        return best

    @staticmethod
    def pareto_front(points):
        pts = [(d, -u, -dis, f) for d, u, dis, f in points]
        front = []
        for i, (d1, u1, dis1, _) in enumerate(pts):
            dominated = False
            for j, (d2, u2, dis2, _) in enumerate(pts):
                if j == i:
                    continue
                if (d2 <= d1) and (u2 <= u1) and (dis2 <= dis1) and ((d2 < d1) or (u2 < u1) or (dis2 < dis1)):
                    dominated = True
                    break
            if not dominated:
                front.append(points[i])
        return front

    @staticmethod
    def select_best_frontier(current_pos, frontiers, drone_map):
        best_frontier, path = PathFinder.next_step(current_pos, frontiers, drone_map)
        return best_frontier

def drone_task(controller, viz, quadrant_bounds):
    current_pos = controller.drone_start
    visited = {current_pos}
    frontiers = set()

    x_min, x_max, y_min, y_max = quadrant_bounds

    controller.scan_surroundings(current_pos, viz, frontiers)

    steps_taken = 0
    redundant_coverage = 0
    start_time = time.time()
    t80_recorded = False
    time_to_80 = None

    while frontiers:
        valid_frontiers = {f for f in frontiers if x_min <= f[0] <= x_max and y_min <= f[1] <= y_max}
        
        if not valid_frontiers:
            break

        target = PathFinder.select_best_frontier(current_pos, valid_frontiers, controller.drone_map)
        if target is None:
            break

        path = astar(current_pos, target, controller.drone_map, controller.grid_size)
        
        if path is None or len(path) <= 1:
            visited.add(target)
            frontiers.discard(target)
            continue

        for node in path[1:]:
            steps_taken += 1
            if node in visited:
                redundant_coverage += 1

            controller.scan_surroundings(current_pos, viz, frontiers)
            if controller.drone_map[node[1], node[0]] == OBSTACLE_CELL:
                break

            success = controller.move_to_cell(node)
            if not success:
                break

            if viz:
                
                viz.move_drone(node[0], node[1])
                viz.mark_visited(node[0], node[1])
                time.sleep(0.02)

            current_pos = node
            visited.add(node)
            frontiers.discard(node)

            if steps_taken % controller.snap_freq == 0:
                controller.snapshots.append(controller.drone_map.copy())
                
        known_cells = np.sum(controller.drone_map != UNKNOWN_CELL)
        coverage_percent = (known_cells / controller.drone_map.size) * 100

        if (not t80_recorded) and coverage_percent >= 80.0:
            time_to_80 = time.time() - start_time
            t80_recorded = True

    known_cells = np.sum(controller.drone_map != UNKNOWN_CELL)
    coverage_percent = (known_cells / controller.drone_map.size) * 100
    elapsed_time = time.time() - start_time

    print(f"\n{controller.vehicle_name} Exploration Summary:")
    print(f"  Coverage achieved: {coverage_percent:.1f}%")
    if time_to_80:
        print(f"  Time to 80% coverage: {time_to_80:.1f} seconds")
    else:
        print(f"  80% coverage not reached")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Steps taken: {steps_taken}")
    print(f"  Redundant visits: {redundant_coverage}")

    controller.land_and_disarm()

def visualize_exploration(controllers):
    """
    Combine 4 drone maps into one RGB visualization.
    Each drone's explored area occupies a distinct quadrant of the global grid.
    """

    UNKNOWN_CELL = -1
    FREE_CELL = 0
    OBSTACLE_CELL = 2
    DISEASE_CELL = 1

    drone_colors = [
        [1.0, 0.0, 0.0],   
        [0.0, 0.0, 1.0],   
        [0.0, 1.0, 0.0],   
        [1.0, 1.0, 0.0]    
    ]

    
    local_size = controllers[0].drone_map.shape[0]
    global_size = local_size * 2  

    
    rgb_map = np.ones((global_size, global_size, 3))  
    combined_map = np.full((global_size, global_size), UNKNOWN_CELL)

    
    offsets = [
        (0, 0),                 
        (local_size, 0),        
        (local_size, local_size),
        (0, local_size)         
    ]

    
    for i, ctrl in enumerate(controllers):
        drone_map = ctrl.drone_map
        color = np.array(drone_colors[i])
        y_off, x_off = offsets[i]

        for y in range(local_size):
            for x in range(local_size):
                val = drone_map[y, x]
                gy, gx = y + y_off, x + x_off

                if val == FREE_CELL:
                    rgb_map[gy, gx] = color * 0.7 + rgb_map[gy, gx] * 0.3
                    combined_map[gy, gx] = FREE_CELL
                elif val == OBSTACLE_CELL:
                    rgb_map[gy, gx] = [0, 0, 0]
                    combined_map[gy, gx] = OBSTACLE_CELL
                elif val == DISEASE_CELL:
                    rgb_map[gy, gx] = [1, 0, 1]
                    combined_map[gy, gx] = DISEASE_CELL

    
    known_cells = np.sum(combined_map != UNKNOWN_CELL)
    coverage_percent = known_cells / combined_map.size * 100

    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_map, origin="lower")
    plt.title(f"🛰️ Multi-Drone Exploration Map (4 Quadrants) — Coverage: {coverage_percent:.1f}%")
    plt.xlabel("Global Grid X")
    plt.ylabel("Global Grid Y")
    plt.grid(True, color="gray", linestyle=":", linewidth=0.5)

    
    for i, color in enumerate(drone_colors, 1):
        plt.scatter([], [], color=color, label=f"Drone {i}")
    plt.scatter([], [], color='black', label='Obstacle')
    plt.scatter([], [], color='magenta', label='Disease')
    plt.legend(loc='upper right')
    plt.savefig("images/final.png")

def visualize_exploration_i(controllers, index):
    """
    Combine 4 drone maps into one RGB visualization.
    Each drone's explored area occupies a distinct quadrant of the global grid.
    """

    UNKNOWN_CELL = -1
    FREE_CELL = 0
    OBSTACLE_CELL = 2
    DISEASE_CELL = 1

    drone_colors = [
        [1.0, 0.0, 0.0],   
        [0.0, 0.0, 1.0],   
        [0.0, 1.0, 0.0],   
        [1.0, 1.0, 0.0]    
    ]

    
    local_size = controllers[0].drone_map.shape[0]
    global_size = local_size * 2  

    
    rgb_map = np.ones((global_size, global_size, 3))  
    combined_map = np.full((global_size, global_size), UNKNOWN_CELL)

    
    offsets = [
        (0, 0),                 
        (local_size, 0),        
        (local_size, local_size),
        (0, local_size)         
    ]

    
    for i, ctrl in enumerate(controllers):
        drone_map = ctrl.snapshots[index]
        color = np.array(drone_colors[i])
        y_off, x_off = offsets[i]

        for y in range(local_size):
            for x in range(local_size):
                val = drone_map[y, x]
                gy, gx = y + y_off, x + x_off

                if val == FREE_CELL:
                    rgb_map[gy, gx] = color * 0.7 + rgb_map[gy, gx] * 0.3
                    combined_map[gy, gx] = FREE_CELL
                elif val == OBSTACLE_CELL:
                    rgb_map[gy, gx] = [0, 0, 0]
                    combined_map[gy, gx] = OBSTACLE_CELL
                elif val == DISEASE_CELL:
                    rgb_map[gy, gx] = [1, 0, 1]
                    combined_map[gy, gx] = DISEASE_CELL

    
    known_cells = np.sum(combined_map != UNKNOWN_CELL)
    coverage_percent = known_cells / combined_map.size * 100

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_map, origin="lower")
    plt.title(f"🛰️ Multi-Drone Exploration Map (4 Quadrants) — Coverage: {coverage_percent:.1f}%")
    plt.xlabel("Global Grid X")
    plt.ylabel("Global Grid Y")
    plt.grid(True, color="gray", linestyle=":", linewidth=0.5)
    
    for i, color in enumerate(drone_colors, 1):
        plt.scatter([], [], color=color, label=f"Drone {i}")
    plt.scatter([], [], color='black', label='Obstacle')
    plt.scatter([], [], color='magenta', label='Disease')
    plt.legend(loc='upper right')
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/temp{index}.png")

def main():
    GLOBAL_GRID_SIZE = 40       
    LOCAL_GRID_SIZE = 20  

    quadrants_bounds = [
        (0, 19, 0, 19),  
        (0, 19, 0, 19),  
        (0, 19, 0, 19),  
        (0, 19, 0, 19),  
    ]
    drones = []
    vizs = []
    local_center = (10,10)
    for i in range(4):
        vehicle_name = f"Drone{i+1}"
        lidar_name = f"LidarSensor{i+1}"

        controller = AirSimDroneController(
            20, 
            local_center, 
            vehicle_name, 
            lidar_name, 
            visualize=False
        )
        drones.append(controller)
        vizs.append(None)

    drones[0].diseases=[[10, 5], [11, 5], [10,4]]
    drones[1].diseases = [[5, 10], [6, 10]]
    drones[2].diseases = [[15, 15]]
    
    for d in drones:
        d.takeoff()

    threads = []
    for i, d in enumerate(drones):
        t = threading.Thread(target=drone_task, args=(d, vizs[i], quadrants_bounds[i]), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All drones finished exploration!")
    for index in range(min([len(i.snapshots) for i in drones])):
        visualize_exploration_i(controllers=drones, index=index)
    visualize_exploration(controllers=drones)

if __name__ == "__main__":
    main()
