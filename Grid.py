import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def generate_uav_positions(grid_size, num_uavs, obstacles, depots, goals):
    """
    Generate random UAV starting positions on the grid,
    avoiding obstacles, depots, and goal cells.
    """
    excluded_set = set(obstacles) | set(depots) | set(goals)
    positions = []
    tries = 0
    max_tries = 1000  # to prevent infinite loop in extreme cases

    while len(positions) < num_uavs and tries < max_tries:
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        if (x, y) not in excluded_set and (x, y) not in positions:
            positions.append((x, y))
        tries += 1

    if len(positions) < num_uavs:
        print(f"WARNING: Only placed {len(positions)}/{num_uavs} UAVs after {max_tries} tries.")
    return positions

def bfs_find_path(grid_size, start, goal, blocked_cells):
    """
    Performs a simple BFS from 'start' to 'goal' on a grid of size 'grid_size',
    avoiding 'blocked_cells'.
    Returns a list of (x, y) coordinates if a path is found, otherwise None.
    """
    # If start or goal are blocked, return None
    if start in blocked_cells or goal in blocked_cells:
        return None

    # 4-directional moves: up, down, left, right
    moves = [(0,1),(0,-1),(-1,0),(1,0)]

    visited = set([start])
    parent = dict()  # to reconstruct path
    queue = deque([start])

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            # Found the goal; reconstruct the path
            path = []
            cur = (cx, cy)
            while cur in parent or cur == start:
                path.append(cur)
                if cur == start:
                    break
                cur = parent[cur]
            path.reverse()
            return path

        # Explore neighbors
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            # Check boundaries
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if (nx, ny) not in visited and (nx, ny) not in blocked_cells:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    return None  # no path found

scenarios = {
    "Small (2 UAVs)": {
        "grid_size": 10,
        "obstacles": [(3, 3), (4, 5)],
        "depots": [(0, 0)],
        "goals": [(8, 8), (2, 7)],
        "num_uavs": 2
    },
    "Medium (4 UAVs)": {
        "grid_size": 10,
        "obstacles": [(3, 3), (4, 5), (7, 2)],
        "depots": [(0, 0)],
        "goals": [(8, 8), (2, 7), (1, 1), (9, 0)],
        "num_uavs": 4
    },
    "Large (6 UAVs)": {
        "grid_size": 10,
        "obstacles": [(3, 3), (4, 5), (7, 2), (5, 8)],
        "depots": [(0, 0)],
        "goals": [(8, 8), (2, 7), (1, 1), (9, 0), (0, 9), (7, 7)],
        "num_uavs": 6
    }
}

# Create 3 subplots, one per scenario
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.suptitle("2D Grid Environment Visualization", fontsize=16)

# Reserve space at the bottom so legends won't overlap or get cut off
plt.subplots_adjust(bottom=0.2, wspace=0.3)

scenario_list = list(scenarios.keys())

for idx, scenario_name in enumerate(scenario_list):
    config = scenarios[scenario_name]
    grid_size = config["grid_size"]
    obstacles = config["obstacles"]
    depots = config["depots"]
    goals = config["goals"]
    num_uavs = config["num_uavs"]

    # Generate UAV positions excluding obstacles, depots, goals
    np.random.seed(idx)  # different seed per scenario
    uav_positions = generate_uav_positions(grid_size, num_uavs, obstacles, depots, goals)

    ax = axs[idx]
    ax.set_title(scenario_name, fontsize=12)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.grid(True, linestyle='--', linewidth=0.5)

    # Convert lists to sets for faster 'in' checks
    obstacles_set = set(obstacles)
    depots_set = set(depots)
    # We'll treat depots as blocked for path planning so the route won't pass over them
    blocked_for_path = obstacles_set | depots_set

    # First, plot obstacles as red squares
    if obstacles:
        ox, oy = zip(*obstacles)
        ax.scatter(ox, oy, marker='s', s=200, color='red', label='Obstacle')

    # Plot depots as blue circles
    if depots:
        dx, dy = zip(*depots)
        ax.scatter(dx, dy, marker='o', s=150, color='blue', label='Depot')

    # Plot goals as green triangles
    if goals:
        gx, gy = zip(*goals)
        ax.scatter(gx, gy, marker='^', s=200, color='green', label='Delivery Point')

    # Plot UAV positions as purple circles
    if uav_positions:
        ux, uy = zip(*uav_positions)
        ax.scatter(ux, uy, marker='o', s=200, color='purple', label='UAV')

    ax.set_aspect('equal', adjustable='box')

    # For each UAV, try to find a path to the corresponding goal
    # We'll skip if we have fewer goals than UAVs
    for i in range(min(num_uavs, len(goals))):
        start = uav_positions[i]
        end = goals[i]

        # Also block out "other UAV positions" so the route won't pass over them.
        # Exclude the UAV's own start, obviously.
        # So let's create a set of all UAV positions except i-th UAV.
        other_uavs = set(uav_positions[:i] + uav_positions[i+1:])
        # Combine with obstacles/depots
        route_blocked = blocked_for_path | other_uavs

        # BFS route
        route = bfs_find_path(grid_size, start, end, route_blocked)
        if route is not None:
            # route is a list of (x,y) we can connect with lines
            rx = [p[0] for p in route]
            ry = [p[1] for p in route]
            ax.plot(rx, ry, linestyle='--', color='black', linewidth=1)

    # Legend: place outside axis, bottom-center
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0,
        ncol=2,
        fontsize=8,
        labelspacing=1.5
    )

plt.savefig("grid_environment_visual.png", dpi=300, bbox_inches='tight')
plt.show()