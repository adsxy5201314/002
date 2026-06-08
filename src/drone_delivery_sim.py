from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    x: float
    y: float
    z: float
    w: float
    d: float
    h: float
    kind: str = "building"

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        px, py, pz = point
        return (
            self.x - margin <= px <= self.x + self.w + margin
            and self.y - margin <= py <= self.y + self.d + margin
            and self.z - margin <= pz <= self.z + self.h + margin
        )

    def distance_to(self, point: np.ndarray) -> float:
        px, py, pz = point
        dx = max(self.x - px, 0.0, px - (self.x + self.w))
        dy = max(self.y - py, 0.0, py - (self.y + self.d))
        dz = max(self.z - pz, 0.0, pz - (self.z + self.h))
        outside = math.sqrt(dx * dx + dy * dy + dz * dz)
        if outside > 0:
            return outside
        return -min(
            px - self.x,
            self.x + self.w - px,
            py - self.y,
            self.y + self.d - py,
            pz - self.z,
            self.z + self.h - pz,
        )


@dataclass(frozen=True)
class DynamicObstacle:
    name: str
    center: tuple[float, float, float]
    amplitude: tuple[float, float, float]
    period: float
    radius: float
    phase: float = 0.0

    def position(self, t: float) -> np.ndarray:
        angle = 2.0 * math.pi * t / self.period + self.phase
        return np.array(
            [
                self.center[0] + self.amplitude[0] * math.sin(angle),
                self.center[1] + self.amplitude[1] * math.cos(angle),
                self.center[2] + self.amplitude[2] * math.sin(angle * 0.7 + self.phase),
            ],
            dtype=float,
        )


@dataclass
class Scenario:
    width: int
    depth: int
    ceiling: int
    min_altitude: float
    cruise_altitude: float
    start: np.ndarray
    goal: np.ndarray
    static_obstacles: list[BoxObstacle]
    dynamic_obstacles: list[DynamicObstacle]
    resolution: float = 2.0
    safety_margin: float = 2.4


def build_scenario() -> Scenario:
    return Scenario(
        width=120,
        depth=80,
        ceiling=42,
        min_altitude=4.0,
        cruise_altitude=28.0,
        start=np.array([8.0, 8.0, 6.0], dtype=float),
        goal=np.array([112.0, 70.0, 6.0], dtype=float),
        static_obstacles=[
            BoxObstacle("mall block", 22, 13, 0, 17, 22, 24),
            BoxObstacle("residential towers", 48, 8, 0, 14, 25, 32),
            BoxObstacle("office campus", 76, 13, 0, 17, 18, 22),
            BoxObstacle("school no-fly column", 32, 50, 0, 22, 17, 42, "no_fly"),
            BoxObstacle("high-rise cluster", 66, 47, 0, 22, 19, 35),
            BoxObstacle("radio mast protected column", 96, 38, 0, 11, 17, 42, "no_fly"),
        ],
        dynamic_obstacles=[
            DynamicObstacle(
                "inspection drone corridor",
                center=(61.0, 38.0, 18.0),
                amplitude=(16.0, 0.0, 3.0),
                period=31.0,
                radius=3.2,
            ),
            DynamicObstacle(
                "temporary crane hook",
                center=(82.0, 42.0, 24.0),
                amplitude=(0.0, 8.0, 5.0),
                period=27.0,
                radius=3.0,
                phase=0.8,
            ),
            DynamicObstacle(
                "medical helicopter approach",
                center=(50.0, 64.0, 22.0),
                amplitude=(9.0, 4.0, 0.0),
                period=36.0,
                radius=3.4,
                phase=1.7,
            ),
        ],
    )


def in_bounds(point: np.ndarray, scenario: Scenario) -> bool:
    x, y, z = point
    return (
        0 <= x <= scenario.width
        and 0 <= y <= scenario.depth
        and scenario.min_altitude <= z <= scenario.ceiling
    )


def blocked(point: np.ndarray, scenario: Scenario, margin: float | None = None) -> bool:
    if margin is None:
        margin = scenario.safety_margin
    if not in_bounds(point, scenario):
        return True
    return any(obs.contains(point, margin=margin) for obs in scenario.static_obstacles)


def to_node(point: np.ndarray, scenario: Scenario) -> tuple[int, int, int]:
    return (
        int(round(point[0] / scenario.resolution)),
        int(round(point[1] / scenario.resolution)),
        int(round(point[2] / scenario.resolution)),
    )


def to_world(node: tuple[int, int, int], scenario: Scenario) -> np.ndarray:
    return np.array(
        [
            node[0] * scenario.resolution,
            node[1] * scenario.resolution,
            node[2] * scenario.resolution,
        ],
        dtype=float,
    )


def astar(scenario: Scenario) -> list[np.ndarray]:
    start = to_node(scenario.start, scenario)
    goal = to_node(scenario.goal, scenario)
    max_x = int(scenario.width / scenario.resolution)
    max_y = int(scenario.depth / scenario.resolution)
    max_z = int(scenario.ceiling / scenario.resolution)

    def valid(node: tuple[int, int, int]) -> bool:
        if not (0 <= node[0] <= max_x and 0 <= node[1] <= max_y and 0 <= node[2] <= max_z):
            return False
        return not blocked(to_world(node, scenario), scenario)

    def heuristic(node: tuple[int, int, int]) -> float:
        world = to_world(node, scenario)
        return float(np.linalg.norm(world - scenario.goal))

    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    frontier: list[tuple[float, tuple[int, int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for dx, dy, dz in neighbors:
            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not valid(nxt):
                continue
            step_cost = scenario.resolution * math.sqrt(dx * dx + dy * dy + dz * dz)
            altitude = to_world(nxt, scenario)[2]
            cruise_penalty = 0.06 * abs(altitude - scenario.cruise_altitude)
            new_cost = cost_so_far[current] + step_cost + cruise_penalty
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        raise RuntimeError("3D A* failed to find a path. Adjust the obstacle layout.")

    path_nodes = []
    node: tuple[int, int, int] | None = goal
    while node is not None:
        path_nodes.append(node)
        node = came_from[node]
    path_nodes.reverse()
    return [to_world(node, scenario) for node in path_nodes]


def segment_clear(a: np.ndarray, b: np.ndarray, scenario: Scenario) -> bool:
    distance = np.linalg.norm(b - a)
    samples = max(2, int(distance / (scenario.resolution * 0.5)))
    for t in np.linspace(0.0, 1.0, samples):
        if blocked(a + (b - a) * t, scenario, margin=scenario.safety_margin):
            return False
    return True


def simplify_path(path: list[np.ndarray], scenario: Scenario) -> list[np.ndarray]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_clear(path[i], path[j], scenario):
            j -= 1
        simplified.append(path[j])
        i = j
    return simplified


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros_like(vector)
    return vector / norm


def lidar_directions() -> list[np.ndarray]:
    directions: list[np.ndarray] = []
    for elevation_deg, count in [(-35, 28), (0, 72), (35, 28)]:
        elevation = math.radians(elevation_deg)
        for azimuth in np.linspace(0, 2 * math.pi, count, endpoint=False):
            directions.append(
                np.array(
                    [
                        math.cos(elevation) * math.cos(azimuth),
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation),
                    ],
                    dtype=float,
                )
            )
    directions.append(np.array([0.0, 0.0, 1.0], dtype=float))
    directions.append(np.array([0.0, 0.0, -1.0], dtype=float))
    return directions


LIDAR_DIRECTIONS = lidar_directions()


def ray_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    scenario: Scenario,
    dyn_centers: list[np.ndarray],
    lidar_range: float,
    step: float = 0.65,
) -> tuple[np.ndarray, float] | None:
    distance = step
    while distance <= lidar_range:
        point = origin + direction * distance
        if blocked(point, scenario, margin=0.25):
            return point, distance
        for center, dyn in zip(dyn_centers, scenario.dynamic_obstacles):
            if np.linalg.norm(point - center) <= dyn.radius + 0.35:
                return point, distance
        distance += step
    return None


def lidar_repulsion(
    position: np.ndarray,
    scenario: Scenario,
    dyn_centers: list[np.ndarray],
    lidar_range: float = 13.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    repulsion = np.zeros(3, dtype=float)
    hits: list[np.ndarray] = []
    for direction in LIDAR_DIRECTIONS:
        hit = ray_hit(position, direction, scenario, dyn_centers, lidar_range)
        if hit is None:
            continue
        point, distance = hit
        hits.append(point)
        away = unit(position - point)
        strength = max(0.0, (1.0 / max(distance, 0.75) - 1.0 / lidar_range))
        repulsion += away * strength * strength * 42.0

    for center, dyn in zip(dyn_centers, scenario.dynamic_obstacles):
        offset = position - center
        clearance_to_body = np.linalg.norm(offset) - dyn.radius
        if clearance_to_body < lidar_range:
            repulsion += unit(offset) * ((lidar_range - clearance_to_body) / lidar_range) ** 2 * 2.8
    return repulsion, hits


def clearance(position: np.ndarray, scenario: Scenario, dyn_centers: list[np.ndarray]) -> float:
    static_clearance = min(
        obs.distance_to(position) - scenario.safety_margin for obs in scenario.static_obstacles
    )
    dynamic_clearance = min(
        np.linalg.norm(position - center) - dyn.radius - 1.2
        for center, dyn in zip(dyn_centers, scenario.dynamic_obstacles)
    )
    border_clearance = min(
        position[0],
        position[1],
        position[2] - scenario.min_altitude,
        scenario.width - position[0],
        scenario.depth - position[1],
        scenario.ceiling - position[2],
    )
    return float(min(static_clearance, dynamic_clearance, border_clearance))


def clip_position(position: np.ndarray, scenario: Scenario) -> np.ndarray:
    return np.clip(
        position,
        [0.0, 0.0, scenario.min_altitude],
        [scenario.width, scenario.depth, scenario.ceiling],
    )


def choose_safe_velocity(
    position: np.ndarray,
    desired_velocity: np.ndarray,
    attraction: np.ndarray,
    repulsion: np.ndarray,
    scenario: Scenario,
    dyn_centers: list[np.ndarray],
    dt: float,
) -> np.ndarray:
    desired_speed = float(np.linalg.norm(desired_velocity))
    if desired_speed < 1e-6:
        return desired_velocity

    candidate_dirs = [
        unit(desired_velocity),
        unit(desired_velocity + repulsion * 0.8),
        unit(repulsion + np.array([0.0, 0.0, 0.55])),
        unit(attraction + np.array([0.0, 0.0, 0.8])),
        unit(np.array([0.0, 0.0, 1.0]) if position[2] < scenario.ceiling - 5 else np.array([0.0, 0.0, -1.0])),
    ]

    best_velocity = desired_velocity
    best_clearance = -float("inf")
    for direction in candidate_dirs:
        if np.linalg.norm(direction) < 1e-6:
            continue
        for scale in (1.0, 0.65, 0.35):
            velocity = direction * desired_speed * scale
            trial = clip_position(position + velocity * dt, scenario)
            trial_clearance = clearance(trial, scenario, dyn_centers)
            if not blocked(trial, scenario, margin=0.0) and trial_clearance > best_clearance:
                best_velocity = velocity
                best_clearance = trial_clearance
            if trial_clearance > 0.15:
                return velocity
    return best_velocity * 0.35


def simulate(scenario: Scenario, waypoints: list[np.ndarray]) -> dict:
    dt = 0.26
    max_speed = 6.6
    reach_radius = 2.6
    lookahead_nodes = 2
    position = scenario.start.copy()
    active_idx = 0
    history = [position.copy()]
    active_waypoints = [active_idx]
    min_clearance = float("inf")
    lidar_log: list[list[list[float]]] = []
    dyn_log: list[list[list[float]]] = []
    speed_log: list[float] = []
    success = False
    collision = False

    for step in range(1500):
        t = step * dt
        dyn_centers = [dyn.position(t) for dyn in scenario.dynamic_obstacles]
        dyn_log.append([center.tolist() for center in dyn_centers])
        current_clearance = clearance(position, scenario, dyn_centers)
        min_clearance = min(min_clearance, current_clearance)
        if current_clearance < -0.12:
            collision = True
            break

        if np.linalg.norm(position - scenario.goal) <= reach_radius:
            success = True
            break

        window_end = min(len(waypoints), active_idx + 12)
        if active_idx < window_end:
            local = waypoints[active_idx:window_end]
            nearest_offset = int(np.argmin([np.linalg.norm(position - point) for point in local]))
            active_idx += nearest_offset
        while active_idx < len(waypoints) - 1 and np.linalg.norm(position - waypoints[active_idx]) < reach_radius:
            active_idx += 1

        target_idx = min(active_idx + lookahead_nodes, len(waypoints) - 1)
        target = waypoints[target_idx]
        attraction = unit(target - position)
        repulsion, hits = lidar_repulsion(position, scenario, dyn_centers)
        repulsion_gain = 0.10 if current_clearance > 7.0 else 0.64
        altitude_trim = unit(np.array([0.0, 0.0, target[2] - position[2]], dtype=float))
        command = (
            1.75 * attraction
            + repulsion_gain * unit(repulsion) * min(np.linalg.norm(repulsion), 2.5)
            + 0.18 * altitude_trim
        )
        if np.linalg.norm(command) < 1e-6:
            command = attraction

        local_clearance = max(0.0, current_clearance)
        speed_scale = 0.38 + 0.62 * min(local_clearance / 8.0, 1.0)
        velocity = unit(command) * max_speed * speed_scale
        next_position = clip_position(position + velocity * dt, scenario)

        next_clearance = clearance(next_position, scenario, dyn_centers)
        if blocked(next_position, scenario, margin=0.0) or next_clearance < 0.65:
            velocity = choose_safe_velocity(
                position,
                velocity,
                attraction,
                repulsion,
                scenario,
                dyn_centers,
                dt,
            )
            next_position = clip_position(position + velocity * dt, scenario)

        position = next_position
        history.append(position.copy())
        active_waypoints.append(active_idx)
        lidar_log.append([point.tolist() for point in hits[:34]])
        speed_log.append(float(np.linalg.norm(velocity)))

    path_length = float(sum(np.linalg.norm(b - a) for a, b in zip(history, history[1:])))
    return {
        "success": success,
        "collision": collision,
        "history": np.array(history),
        "active_waypoints": active_waypoints,
        "lidar_log": lidar_log,
        "dyn_log": dyn_log[: len(history)],
        "speed_log": speed_log,
        "min_clearance": float(min_clearance),
        "path_length": path_length,
        "flight_time_s": float((len(history) - 1) * dt),
        "dt": dt,
    }


def cuboid_faces(x: float, y: float, z: float, w: float, d: float, h: float) -> list[list[tuple[float, float, float]]]:
    p = [
        (x, y, z),
        (x + w, y, z),
        (x + w, y + d, z),
        (x, y + d, z),
        (x, y, z + h),
        (x + w, y, z + h),
        (x + w, y + d, z + h),
        (x, y + d, z + h),
    ]
    return [
        [p[0], p[1], p[2], p[3]],
        [p[4], p[5], p[6], p[7]],
        [p[0], p[1], p[5], p[4]],
        [p[2], p[3], p[7], p[6]],
        [p[1], p[2], p[6], p[5]],
        [p[0], p[3], p[7], p[4]],
    ]


def draw_box_edges(
    ax,
    x: float,
    y: float,
    z: float,
    w: float,
    d: float,
    h: float,
    color: str,
    alpha: float = 0.55,
    linestyle: str = "--",
) -> None:
    vertices = [
        (x, y, z),
        (x + w, y, z),
        (x + w, y + d, z),
        (x, y + d, z),
        (x, y, z + h),
        (x + w, y, z + h),
        (x + w, y + d, z + h),
        (x, y + d, z + h),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [vertices[a][0], vertices[b][0]],
            [vertices[a][1], vertices[b][1]],
            [vertices[a][2], vertices[b][2]],
            color=color,
            linewidth=0.55,
            alpha=alpha,
            linestyle=linestyle,
        )


def add_box(ax, obs: BoxObstacle, face: str, edge: str, alpha: float) -> None:
    poly = Poly3DCollection(
        cuboid_faces(obs.x, obs.y, obs.z, obs.w, obs.d, obs.h),
        facecolors=face,
        edgecolors=edge,
        linewidths=0.55,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def draw_base(ax, scenario: Scenario) -> None:
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.depth)
    ax.set_zlim(0, scenario.ceiling)
    ax.set_box_aspect((scenario.width, scenario.depth, scenario.ceiling * 1.35))
    ax.view_init(elev=27, azim=-62)
    ax.set_xlabel("east / m")
    ax.set_ylabel("north / m")
    ax.set_zlabel("altitude / m")
    ax.set_facecolor("#f8fafc")

    for x in range(0, scenario.width + 1, 10):
        ax.plot([x, x], [0, scenario.depth], [0, 0], color="#e2e8f0", linewidth=0.45, zorder=0)
    for y in range(0, scenario.depth + 1, 10):
        ax.plot([0, scenario.width], [y, y], [0, 0], color="#e2e8f0", linewidth=0.45, zorder=0)

    colors = {
        "building": ("#64748b", "#334155", 0.66),
        "no_fly": ("#fb923c", "#b45309", 0.44),
    }
    for obs in scenario.static_obstacles:
        face, edge, alpha = colors[obs.kind]
        add_box(ax, obs, face, edge, alpha)
        margin = scenario.safety_margin
        z0 = max(0.0, obs.z - margin)
        top = min(float(scenario.ceiling), obs.z + obs.h + margin)
        draw_box_edges(
            ax,
            obs.x - margin,
            obs.y - margin,
            z0,
            obs.w + 2 * margin,
            obs.d + 2 * margin,
            max(0.1, top - z0),
            edge,
        )
        ax.text(obs.x + obs.w / 2, obs.y + obs.d / 2, min(obs.z + obs.h + 1.4, scenario.ceiling), obs.name, fontsize=6)

    ax.scatter(*scenario.start, marker="o", s=62, c="#16a34a", label="hospital start", depthshade=True)
    ax.scatter(*scenario.goal, marker="*", s=145, c="#dc2626", label="delivery target", depthshade=True)


def plot_sphere(ax, center: np.ndarray, radius: float, color: str, edge: str, alpha: float = 0.38) -> None:
    u = np.linspace(0, 2 * math.pi, 18)
    v = np.linspace(0, math.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, edgecolor=edge, linewidth=0.2, alpha=alpha, shade=True)


def plot_final(scenario: Scenario, global_path: list[np.ndarray], waypoints: list[np.ndarray], result: dict) -> None:
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    draw_base(ax, scenario)
    path = np.array(global_path)
    simplified = np.array(waypoints)
    history = result["history"]
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#94a3b8", linewidth=1.0, linestyle=":", label="3D A* grid path")
    ax.plot(
        simplified[:, 0],
        simplified[:, 1],
        simplified[:, 2],
        color="#475569",
        linewidth=1.7,
        linestyle="--",
        label="3D smoothed waypoints",
    )
    ax.plot(history[:, 0], history[:, 1], history[:, 2], color="#2563eb", linewidth=2.6, label="executed trajectory")

    final_t = result["flight_time_s"]
    for dyn in scenario.dynamic_obstacles:
        center = dyn.position(final_t)
        plot_sphere(ax, center, dyn.radius, "#facc15", "#a16207")

    status = "success" if result["success"] else "failed"
    ax.set_title(
        f"3D urban delivery auto-obstacle-avoidance - {status}, "
        f"min clearance {result['min_clearance']:.2f} m"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUTPUT_DIR / "final_trajectory.png", dpi=180)
    plt.close(fig)


def save_animation(scenario: Scenario, global_path: list[np.ndarray], waypoints: list[np.ndarray], result: dict) -> None:
    history = result["history"]
    frame_indices = np.unique(np.linspace(0, len(history) - 1, min(140, len(history)), dtype=int))
    fig = plt.figure(figsize=(9.6, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    draw_base(ax, scenario)

    path = np.array(global_path)
    simplified = np.array(waypoints)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#cbd5e1", linewidth=0.8, linestyle=":")
    ax.plot(simplified[:, 0], simplified[:, 1], simplified[:, 2], color="#64748b", linewidth=1.1, linestyle="--")
    (trail,) = ax.plot([], [], [], color="#2563eb", linewidth=2.2)
    drone = ax.scatter(
        [history[0, 0]],
        [history[0, 1]],
        [history[0, 2]],
        s=58,
        c="#2563eb",
        edgecolors="#1e3a8a",
        depthshade=True,
        zorder=8,
    )
    lidar_lines = [ax.plot([], [], [], color="#38bdf8", linewidth=0.5, alpha=0.45)[0] for _ in range(34)]
    dyn_scatter = ax.scatter([], [], [], s=180, c="#facc15", edgecolors="#a16207", alpha=0.72, depthshade=True)
    title = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=10, va="top")

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])
        pos = history[idx]
        trail.set_data(history[: idx + 1, 0], history[: idx + 1, 1])
        trail.set_3d_properties(history[: idx + 1, 2])
        drone._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        dyn_centers = np.array(result["dyn_log"][min(idx, len(result["dyn_log"]) - 1)])
        dyn_scatter._offsets3d = (dyn_centers[:, 0], dyn_centers[:, 1], dyn_centers[:, 2])

        hits = result["lidar_log"][min(idx, len(result["lidar_log"]) - 1)] if result["lidar_log"] else []
        for line, hit in zip(lidar_lines, hits):
            line.set_data([pos[0], hit[0]], [pos[1], hit[1]])
            line.set_3d_properties([pos[2], hit[2]])
        for line in lidar_lines[len(hits) :]:
            line.set_data([], [])
            line.set_3d_properties([])

        ax.view_init(elev=27, azim=-62 + frame_number * 0.11)
        title.set_text(f"t={idx * result['dt']:5.1f}s | altitude={pos[2]:4.1f}m | step={idx:03d}")
        return [trail, drone, dyn_scatter, title, *lidar_lines]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=85, blit=False)
    gif_path = OUTPUT_DIR / "drone_delivery_demo.gif"
    mp4_path = OUTPUT_DIR / "drone_delivery_demo.mp4"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    try:
        import imageio.v3 as iio

        frames = []
        for frame in iio.imiter(gif_path):
            frame = np.asarray(frame)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            pad_h = frame.shape[0] % 2
            pad_w = frame.shape[1] % 2
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            frames.append(frame)
        iio.imwrite(
            mp4_path,
            frames,
            fps=12,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
    except Exception as exc:  # pragma: no cover - depends on optional video backend.
        (OUTPUT_DIR / "mp4_export_error.txt").write_text(str(exc), encoding="utf-8")


def save_summary(scenario: Scenario, global_path: list[np.ndarray], waypoints: list[np.ndarray], result: dict) -> None:
    history = result["history"]
    summary = {
        "scenario": "urban_low_altitude_3d_delivery",
        "airspace_size_m": [scenario.width, scenario.depth, scenario.ceiling],
        "start_m": scenario.start.tolist(),
        "goal_m": scenario.goal.tolist(),
        "static_obstacle_count": len(scenario.static_obstacles),
        "dynamic_obstacle_count": len(scenario.dynamic_obstacles),
        "global_grid_nodes": len(global_path),
        "smoothed_waypoints": len(waypoints),
        "success": result["success"],
        "collision": result["collision"],
        "flight_time_s": round(result["flight_time_s"], 2),
        "executed_path_length_m": round(result["path_length"], 2),
        "minimum_clearance_m": round(result["min_clearance"], 2),
        "mean_speed_mps": round(float(np.mean(result["speed_log"])) if result["speed_log"] else 0.0, 2),
        "min_altitude_m": round(float(np.min(history[:, 2])), 2),
        "max_altitude_m": round(float(np.max(history[:, 2])), 2),
        "mean_altitude_m": round(float(np.mean(history[:, 2])), 2),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    scenario = build_scenario()
    global_path = astar(scenario)
    waypoints = simplify_path(global_path, scenario)
    result = simulate(scenario, waypoints)

    plot_final(scenario, global_path, waypoints, result)
    save_animation(scenario, global_path, waypoints, result)
    save_summary(scenario, global_path, waypoints, result)

    print("3D simulation complete")
    print(f"success: {result['success']}")
    print(f"collision: {result['collision']}")
    print(f"flight time: {result['flight_time_s']:.2f}s")
    print(f"path length: {result['path_length']:.2f}m")
    print(f"minimum clearance: {result['min_clearance']:.2f}m")
    print(f"altitude range: {result['history'][:, 2].min():.2f}m - {result['history'][:, 2].max():.2f}m")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
