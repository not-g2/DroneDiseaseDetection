#hi
import heapq

def astar(start, goal, grid, grid_size):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))  # (f, g, node, path)
    visited = set()

    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        x, y = node
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if grid[nx][ny] != 2:  # avoid obstacles/disease
                    new_cost = g + 1
                    heapq.heappush(open_set, (new_cost + heuristic((nx, ny), goal), new_cost, (nx, ny), path + [(nx, ny)]))
    return None

