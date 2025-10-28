import time
import math
import threading
import random
import argparse
import turtle
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from astar import astar 

CELL_SIZE_PIXELS = 15
METERS_PER_CELL = 1.0
DRONEKIT_CONN = "udp:127.0.0.1:14550"  # change to your SITL output
TAKEOFF_ALT = 1.0
GOTO_TIMEOUT = 30
REACH_THRESH_M = 2.0
UNKNOWN_CELL = -1
FREE_CELL = 0
DISEASE_CELL = 1
OBSTACLE_CELL = 2
DISEASE_CELL_CORD = [
    # Bottom-left patch (near south-west)
    (5, 8), (6, 8), (6, 9), (7, 9),

    # Center-left patch
    (18, 15), (18, 16), (19, 16), (19, 17),

    # Top-right patch (north-east)
    (38, 40), (39, 40), (39, 41), (40, 41),

    # Slight scattered infection line (like crop row disease)
    (30, 25), (31, 26), (32, 27)
]

OBSTACLE_CELL_CORD = [
    # Fence along top edge
    *[(x, 48) for x in range(5, 45, 3)],

    # Fence along right edge
    *[(48, y) for y in range(5, 45, 3)],

    # Small shed or obstacle cluster mid-field
    (22, 22), (23, 22), (22, 23), (23, 23),

    # Trees or poles scattered
    (10, 35), (12, 36), (14, 34),
    (25, 10), (26, 11),
    (40, 15), (41, 15), (42, 16)
]


class Visualizer:
    def __init__(self, grid_size, obstacles, diseases, drone_start):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.diseases = set(diseases)
        self.drone_pos = drone_start

        self.screen = turtle.Screen()
        self.screen.title("Drone Exploration (North ↑, East →)")
        self.screen.setup(width=900, height=900)
        self.screen.tracer(0)

        self.drawer = turtle.Turtle()
        self.drawer.hideturtle()
        self.drawer.speed(0)
        self.drawer.penup()

        self._draw_grid_background()

        for (x, y) in self.obstacles:
            self.draw_cell(x, y, "black")
        for (x, y) in self.diseases:
            self.draw_cell(x, y, "purple")
        self.draw_cell(drone_start[0], drone_start[1], "blue")

        self.drone_t = turtle.Turtle()
        self.drone_t.shape("square")
        scale = CELL_SIZE_PIXELS / 20  # default turtle square size is 20px
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

    def mark_disease(self, x, y):
        self.draw_cell(x, y, "green")

def start_turtle_loop():
    turtle.mainloop()

class PathFinder:
    def __init__(self, grid, drone_start, num_of_diseases, num_of_obstacles, w_dist, w_unexplored, visualize):
        self.grid_size = grid
        self.drone_start = drone_start
        self.num_of_diseases = num_of_diseases
        self.num_of_obstacles = num_of_obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.diseases_loc = set()
        self.obstacles_loc = set()
        self.w_dist = w_dist
        self.w_unexplored = w_unexplored
        self.visualize = visualize

        for x, y in DISEASE_CELL_CORD:
            self.grid[x][y] = DISEASE_CELL
            self.diseases_loc.add((x, y))

        for x, y in OBSTACLE_CELL_CORD:
            self.grid[x][y] = OBSTACLE_CELL
            self.obstacles_loc.add((x, y))

        self.drone_map = np.full_like(self.grid, fill_value=UNKNOWN_CELL)
        self.drone_map[drone_start] = 0

    def grid_to_gps(self, cell, home_gps, meters_per_cell=METERS_PER_CELL):
        lat0, lon0, alt0 = home_gps
        x, y = cell
        dNorth = y * meters_per_cell
        dEast = x * meters_per_cell
        earth_radius = 6378137.0
        dLat = dNorth / earth_radius
        dLon = dEast / (earth_radius * math.cos(math.radians(lat0)))
        newlat = lat0 + (dLat * 180.0 / math.pi)
        newlon = lon0 + (dLon * 180.0 / math.pi)
        return (newlat, newlon, alt0)

    def neighbour(self, point):
        x, y = point
        nei = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<self.grid_size and 0<=ny<self.grid_size and self.drone_map[nx][ny]==UNKNOWN_CELL:
                if self.grid[nx][ny]==OBSTACLE_CELL:
                    self.drone_map[nx][ny]=OBSTACLE_CELL
                else:
                    self.drone_map[nx][ny]=FREE_CELL
                    nei.append((nx,ny))
        return nei

    def check_disease(self, point):
        if self.grid[point[0]][point[1]]==DISEASE_CELL:
            self.drone_map[point[0]][point[1]]=DISEASE_CELL
            return True
        return False

    def next_step(self, cur_pos, frontiers):
        if not frontiers:
            return None, None, []

        scored = []
        for f in frontiers:
            dist = abs(f[0] - cur_pos[0]) + abs(f[1] - cur_pos[1])
            unexplored = sum(
                1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= f[0] + dx < len(self.drone_map)
                and 0 <= f[1] + dy < len(self.drone_map)
                and self.drone_map[f[0] + dx][f[1] + dy] == UNKNOWN_CELL
            )
            scored.append((dist, unexplored, f))

        front = PathFinder.pareto_front(scored)
        chosen = PathFinder.knee_by_utopia(front)
        best_frontier = chosen[2]

        path = astar(cur_pos, best_frontier, self.drone_map, self.grid_size)
        return best_frontier, path

    @staticmethod
    def knee_by_utopia(pareto):
        # pareto is [(dist, unexplored, frontier_cell)]
        min_dist = min(p[0] for p in pareto)
        max_unexploded = max(p[1] for p in pareto)

        # utopia = (min dist, max unexplored)
        utopia = (min_dist, -max_unexploded)  # convert unexplored to negative

        best = None
        best_d = float('inf')
        for dist, unexploded, f in pareto:
            d = math.hypot(dist - utopia[0], (-unexploded) - utopia[1])
            if d < best_d:
                best_d = d
                best = (dist, unexploded, f)
        return best

    @staticmethod
    def pareto_front(points):
        pts = [(d, -u, f) for d, u, f in points]
        front = []
        for i, (d1, u1, f1) in enumerate(pts):
            dominated = False
            for j, (d2, u2, _) in enumerate(pts):
                if j == i:
                    continue
                # j dominates i if <= on both and < on at least one
                if d2 <= d1 and u2 <= u1 and (d2 < d1 or u2 < u1):
                    dominated = True
                    break
            if not dominated:
                front.append(points[i])
        return front


def wait_for_gps(vehicle, timeout=30):
    t0=time.time()
    while True:
        fix = getattr(vehicle,'gps_0',None)
        fix_type = fix.fix_type if fix is not None and hasattr(fix,'fix_type') else 0
        if fix_type>=2:
            return True
        if time.time()-t0>timeout:
            return False
        print("Waiting for GPS fix (fix_type=%s)..."%str(fix_type))
        time.sleep(1)

def set_guided_mode(vehicle, timeout=10):
    vehicle.mode=VehicleMode("GUIDED")
    t0=time.time()
    while vehicle.mode.name!="GUIDED":
        if time.time()-t0>timeout: return False
        print(" Waiting for GUIDED mode...")
        time.sleep(0.5)
    return True

def arm_and_takeoff(vehicle, target_alt):
    print("Basic pre-arm checks")
    t0=time.time()
    while not vehicle.is_armable:
        if time.time()-t0>30: raise RuntimeError("Vehicle not armable after 30s")
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)
    if not set_guided_mode(vehicle):
        raise RuntimeError("Could not set GUIDED mode")
    print("Arming motors")
    vehicle.armed=True
    t0=time.time()
    while not vehicle.armed:
        if time.time()-t0>10: raise RuntimeError("Timeout arming")
        time.sleep(0.5)
    print("Taking off!")
    vehicle.simple_takeoff(target_alt)
    t0=time.time()
    while True:
        alt=vehicle.location.global_relative_frame.alt
        print(" Altitude: %.2f m"%alt)
        if alt>=target_alt*0.95: break
        if time.time()-t0>30: break
        time.sleep(1)

def distance_meters(a_lat,a_lon,b_lat,b_lon):
    R=6371000.0
    phi1=math.radians(a_lat)
    phi2=math.radians(b_lat)
    dphi=math.radians(b_lat-a_lat)
    dlambda=math.radians(b_lon-a_lon)
    hav=math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(hav))

def goto_and_wait(vehicle, lat, lon, alt, timeout=GOTO_TIMEOUT, reach_thresh=REACH_THRESH_M):
    target=LocationGlobalRelative(lat,lon,alt)
    vehicle.simple_goto(target)
    t0=time.time()
    while True:
        cur=vehicle.location.global_relative_frame
        if cur is None: time.sleep(0.5); continue
        dist=distance_meters(cur.lat,cur.lon,lat,lon)
        if dist<=reach_thresh: return True
        if time.time()-t0>timeout:
            print("WARN: goto timeout, remaining %.1f m"%dist)
            return False
        time.sleep(0.1)

def main(args):
    print("Connecting to vehicle on:", DRONEKIT_CONN)
    vehicle=connect(DRONEKIT_CONN, wait_ready=True, timeout=120)

    grid_size=args.grid
    droneStart=tuple(map(int,args.droneStart.split(',')))
    pf=PathFinder(grid_size, droneStart, args.numOfDisease, args.numOfObstacles,
                  args.weightDistance, args.weightUnexplored, visualize= True)
    visualize = True
    if visualize:
        viz=Visualizer(grid_size, pf.obstacles_loc, pf.diseases_loc, droneStart)
    t=threading.Thread(target=start_turtle_loop, daemon=True)
    t.start()
    time.sleep(0.5)

    print("Waiting for GPS...")
    if not wait_for_gps(vehicle, timeout=30):
        print("WARNING: No GPS fix after timeout; continuing (SITL may be OK)")

    print("Arming & takeoff to", TAKEOFF_ALT)
    arm_and_takeoff(vehicle, TAKEOFF_ALT)
    home_gps=(vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, TAKEOFF_ALT)
    print("Home GPS:", home_gps)

    start_time = time.time()
    steps_taken = 0

    cur_pos=droneStart
    visited= {cur_pos}
    frontiers=set(pf.neighbour(cur_pos))

    while frontiers:
        target, path = pf.next_step(cur_pos, frontiers)
        if target is None or path is None:
            print("No reachable frontier left.")
            break
        for node in path[1:]:
            lat, lon, alt=pf.grid_to_gps(node, home_gps, METERS_PER_CELL)
            print(f"Flying to {node} -> {lat:.6f},{lon:.6f}")
            steps_taken += 1
            if visualize:
                viz.move_drone(node[0], node[1])
                viz.screen.update()
                time.sleep(0.1)
                viz.mark_visited(node[0], node[1])
                if pf.grid[node[0]][node[1]] == 1:
                    viz.mark_disease(node[0], node[1])
            goto_and_wait(vehicle, lat, lon, alt, timeout=GOTO_TIMEOUT)
            pf.check_disease(node)
            frontiers.update(pf.neighbour(node))
            cur_pos=node
            visited.add(node)
            if target in frontiers:
                frontiers.remove(target)

    end_time = time.time()
    print(f"Time Taken {(end_time - start_time):.2f} s")
    print(f"Steps Taken {steps_taken}")
    print("Exploration finished, landing...")
    vehicle.mode=VehicleMode("LAND")
    time.sleep(6)
    vehicle.close()
    print("Mission complete. Close the turtle window manually or stop the script.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="")
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--droneStart", type=str, default="0,0")
    parser.add_argument("--numOfDisease", type=int, default=3)
    parser.add_argument("--numOfObstacles", type=int, default=5)
    parser.add_argument("--weightDistance", type=float, default=2.389346)
    parser.add_argument("--weightUnexplored", type=float, default=2.358536)
    parser.add_argument("--no-visualize", action="store_true", default=False, help="Disable visualization")
    args=parser.parse_args()
    main(args)

