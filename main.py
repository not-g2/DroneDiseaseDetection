# full script (modified with fixes described)
import time
import math
import threading
import argparse
import numpy as np
import airsim
from astar import astar 
import matplotlib.pyplot as plt
import os

CELL_SIZE_PIXELS = 5
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
OBSTACLE_BUFFER_CELLS = 1  # Add buffer around detected obstacles

import sys
import datetime

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)   # show live in console
        self.log.write(message)        # save to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class AirSimDroneController:
    def __init__(self, grid_size, drone_start, vehicle_name, lidar_name, visualize=True, ip="127.0.0.1"):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()
        # enable API control and arm for specific vehicle
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

        # Initialize grid map (drone's knowledge) - stored as [y, x]
        self.drone_map = np.full((grid_size, grid_size), fill_value=UNKNOWN_CELL, dtype=np.int8)
        # set start cell free
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

        # north = origin_ned.x + dy, east = origin_ned.y + dx
        ned_x = self.origin_ned[0] + dy  # north
        ned_y = self.origin_ned[1] + dx  # east
        ned_z = -TAKEOFF_ALT
        return ned_x, ned_y, ned_z

    def ned_to_grid(self, north, east):
        """Convert world NED coords to grid coords anchored at origin_ned and start_cell"""
        if self.origin_ned is None:
            raise RuntimeError("origin_ned not set; call takeoff() first to set origin")
        # compute offset from origin (in meters)
        dx_m = east - self.origin_ned[1]   # east offset
        dy_m = north - self.origin_ned[0]  # north offset

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
        # Set origin for conversions (anchor)
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
            # no points returned
            #print(f"{self.vehicle_name}: LiDAR returned 0 points!")
            return []

        obstacles = set()
        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position

        for point in points:
            horiz_dist = math.hypot(point[0], point[1])
            if horiz_dist < 1e-3 or horiz_dist > LIDAR_RANGE:
                continue

            # interpret sensor local z as relative down offset; combine per earlier discussion
            # point[2] is sensor-local down; height relative to drone is -point[2] (if needed)
            # We'll use point[2] for a simple pass-through check along with configured min/max
            if not (LIDAR_MIN_HEIGHT < point[2] < LIDAR_MAX_HEIGHT):
                continue

            world_north = pos.x_val + point[0]
            world_east = pos.y_val + point[1]

            grid_x, grid_y = self.ned_to_grid(world_north, world_east)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                if horiz_dist < OBSTACLE_THRESHOLD:
                    # add cell + buffer
                    for dx in range(-OBSTACLE_BUFFER_CELLS, OBSTACLE_BUFFER_CELLS + 1):
                        for dy in range(-OBSTACLE_BUFFER_CELLS, OBSTACLE_BUFFER_CELLS + 1):
                            bx, by = grid_x + dx, grid_y + dy
                            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                                obstacles.add((bx, by))
        #print(f"{self.vehicle_name} LiDAR detected {len(obstacles)} obstacle cells at {current_cell}")
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
        #print(f"{self.vehicle_name} LiDAR detected {len(obstacles)} obstacle cells at position {current_cell}")
        #print(obstacles)
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
        # compute current grid cell for obstacle check
        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        current_grid = self.ned_to_grid(pos.x_val, pos.y_val)
        obstacles = self.detect_obstacles_from_lidar(current_grid)

        if target_cell in obstacles or self.drone_map[target_cell[1], target_cell[0]] == OBSTACLE_CELL:
            print(f"{self.vehicle_name}: WARNING: Target cell {target_cell} is blocked!")
            return False

        north, east, down = self.grid_to_ned(target_cell)
        #print(f"{self.vehicle_name}: Moving to cell {target_cell} -> NED({north:.2f}, {east:.2f}, {down:.2f})")
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

        grid_h, grid_w = drone_map.shape  # rows(y), cols(x)
        scored = []

        disease_positions = np.argwhere(drone_map == DISEASE_CELL)  # (y,x) pairs

        for f in frontiers:
            fx, fy = f
            # Manhattan distance
            dist = abs(fx - cur_pos[0]) + abs(fy - cur_pos[1])

            # unexplored neighbors (4-connected)
            unexplored = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    if drone_map[ny, nx] == UNKNOWN_CELL:
                        unexplored += 1

            # disease proximity: compute Manhattan distance from current pos to nearest disease cell
            if disease_positions.size > 0:
                dxs = np.abs(disease_positions[:, 1] - cur_pos[0])  # x differences
                dys = np.abs(disease_positions[:, 0] - cur_pos[1])  # y differences
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
        #print(controller.drone_map)
        if not valid_frontiers:
            break

        target = PathFinder.select_best_frontier(current_pos, valid_frontiers, controller.drone_map)
        if target is None:
            break

        path = astar(current_pos, target, controller.drone_map, controller.grid_size)
        #print(path, target, current_pos)
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
                # safe: viz is None for multi-drone mode to avoid Tkinter threading issues
                viz.move_drone(node[0], node[1])
                viz.mark_visited(node[0], node[1])
                time.sleep(0.02)

            current_pos = node
            visited.add(node)
            frontiers.discard(node)

            if steps_taken % controller.snap_freq == 0:
                controller.snapshots.append(controller.drone_map.copy())

                # Track coverage progress
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

    print(controller.drone_map)

class MatplotlibVisualizer:
    def __init__(self, grid_size, start_pos=None):
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title(f"Drone Map (start {start_pos})")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.img = self.ax.imshow(np.ones((grid_size, grid_size)) * -1,
                                  cmap="viridis", vmin=-1, vmax=3)
        plt.ion(); plt.show(block=False)

    def update(self, grid, drone_pos=None):
        display = grid.copy()
        if drone_pos is not None:
            x, y = drone_pos
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                display[x, y] = 4
        self.img.set_data(display)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

import numpy as np
import matplotlib.pyplot as plt

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
        [1.0, 0.0, 0.0],   # Drone 1 - Red
        [0.0, 0.0, 1.0],   # Drone 2 - Blue
        [0.0, 1.0, 0.0],   # Drone 3 - Green
        [1.0, 1.0, 0.0]    # Drone 4 - Yellow
    ]

    # Each drone map is 1/4 of total area (assume all same size)
    local_size = controllers[0].drone_map.shape[0]
    global_size = local_size * 2  # 2x2 layout

    # Initialize RGB and combined maps
    rgb_map = np.ones((global_size, global_size, 3))  # start as white (unknown)
    combined_map = np.full((global_size, global_size), UNKNOWN_CELL)

    # Position offsets for 4 quadrants: (y_offset, x_offset)
    offsets = [
        (0, 0),                 # Drone 1 - bottom-left
        (local_size, 0),        # Drone 2 - bottom-right
        (local_size, local_size),# Drone 3 - top-right
        (0, local_size)         # Drone 4 - top-left
    ]

    # Blend each drone map into the global RGB map
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

    # Compute coverage %
    known_cells = np.sum(combined_map != UNKNOWN_CELL)
    coverage_percent = known_cells / combined_map.size * 100

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_map, origin="lower")
    plt.title(f"🛰️ Multi-Drone Exploration Map (4 Quadrants) — Coverage: {coverage_percent:.1f}%")
    plt.xlabel("Global Grid X")
    plt.ylabel("Global Grid Y")
    plt.grid(True, color="gray", linestyle=":", linewidth=0.5)

    # Legend
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
        [1.0, 0.0, 0.0],   # Drone 1 - Red
        [0.0, 0.0, 1.0],   # Drone 2 - Blue
        [0.0, 1.0, 0.0],   # Drone 3 - Green
        [1.0, 1.0, 0.0]    # Drone 4 - Yellow
    ]

    # Each drone map is 1/4 of total area (assume all same size)
    local_size = controllers[0].drone_map.shape[0]
    global_size = local_size * 2  # 2x2 layout

    # Initialize RGB and combined maps
    rgb_map = np.ones((global_size, global_size, 3))  # start as white (unknown)
    combined_map = np.full((global_size, global_size), UNKNOWN_CELL)

    # Position offsets for 4 quadrants: (y_offset, x_offset)
    offsets = [
        (0, 0),                 # Drone 1 - bottom-left
        (local_size, 0),        # Drone 2 - bottom-right
        (local_size, local_size),# Drone 3 - top-right
        (0, local_size)         # Drone 4 - top-left
    ]

    # Blend each drone map into the global RGB map
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

    # Compute coverage %
    known_cells = np.sum(combined_map != UNKNOWN_CELL)
    coverage_percent = known_cells / combined_map.size * 100

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_map, origin="lower")
    plt.title(f"🛰️ Multi-Drone Exploration Map (4 Quadrants) — Coverage: {coverage_percent:.1f}%")
    plt.xlabel("Global Grid X")
    plt.ylabel("Global Grid Y")
    plt.grid(True, color="gray", linestyle=":", linewidth=0.5)

    # Legend
    for i, color in enumerate(drone_colors, 1):
        plt.scatter([], [], color=color, label=f"Drone {i}")
    plt.scatter([], [], color='black', label='Obstacle')
    plt.scatter([], [], color='magenta', label='Disease')
    plt.legend(loc='upper right')
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/temp{index}.png")

def main(args):
    GLOBAL_GRID_SIZE = 40       # total area (20m × 20m)
    LOCAL_GRID_SIZE = 20  # each drone’s map (10m × 10m)

    # Quadrant bounds in global coordinates (x_min, x_max, y_min, y_max)
    quadrants_bounds = [
        (0, 19, 0, 19),  # top-left
        (0, 19, 0, 19),  # top-right
        (0, 19, 0, 19),  # bottom-left
        (0, 19, 0, 19),  # bottom-right
    ]

    drones = []
    vizs = []

    # Each drone has its own 20×20 grid, centered at (10,10) in its local map
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

    drones[0].drone_map[10, 5] = DISEASE_CELL
    drones[0].drone_map[11, 5] = DISEASE_CELL
    drones[0].drone_map[10, 4] = DISEASE_CELL

    drones[1].drone_map[5, 10] = DISEASE_CELL
    drones[1].drone_map[6, 10] = DISEASE_CELL

    drones[2].drone_map[15, 15] = DISEASE_CELL

    # Disable per-drone visualization (Tkinter thread safety)
    if not args.no_visualize:
        print("Visualization disabled for multi-drone mode (turtle/matplotlib threading limits).")

    # Takeoff all drones
    for d in drones:
        d.takeoff()

    # Start threads for each quadrant
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
    parser = argparse.ArgumentParser(description="AirSim Multi-Drone LiDAR Frontier Exploration")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization (recommended for multi-drone)")
    args = parser.parse_args()
    main(args)
