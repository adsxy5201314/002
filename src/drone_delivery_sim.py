from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class RectObstacle:
    name: str
    x: float
    y: float
    w: float
    h: float
    kind: str = "building"

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        px, py = point
        return (
            self.x - margin <= px <= self.x + self.w + margin
            and self.y - margin <= py <= self.y + self.h + margin
        )

    def distance_to(self, point: np.ndarray) -> float:
        px, py = point
        dx = max(self.x - px, 0.0, px - (self.x + self.w))
        dy = max(self.y - py, 0.0, py - (self.y + self.h))
        outside = math.hypot(dx, dy)
        if outside > 0:
            return outside
        return -min(px - self.x, self.x + self.w - px, py - self.y, self.y + self.h - py)


@dataclass(frozen=True)
class DynamicObstacle:
    name: str
    center: tuple[float, float]
    amplitude: tuple[float, float]
    period: float
    radius: float
    phase: float = 0.0

    def position(self, t: float) -> np.ndarray:
        angle = 2.0 * math.pi * t / self.period + self.phase
        return np.array(
            [
                self.center[0] + self.amplitude[0] * math.sin(angle),
                self.center[1] + self.amplitude[1] * math.cos(angle),
            ],
            dtype=float,
        )


@dataclass
class Scenario:
    width: int
    height: int
    start: np.ndarray
    goal: np.ndarray
    static_obstacles: list[RectObstacle]
    dynamic_obstacles: list[DynamicObstacle]
    resolution: float = 1.0
    safety_margin: float = 2.4


def build_scenario() -> Scenario:
    return Scenario(
        width=120,
        height=80,
        start=np.array([8.0, 9.0], dtype=float),
        goal=np.array([112.0, 70.0], dtype=float),
        static_obstacles=[
            RectObstacle("mall block", 22, 13, 17, 22),
            RectObstacle("residential towers", 48, 8, 14, 25),
            RectObstacle("office campus", 76, 13, 17, 18),
            RectObstacle("school no-fly zone", 32, 50, 22, 17, "no_fly"),
            RectObstacle("high-rise cluster", 66, 47, 22, 19),
            RectObstacle("radio mast buffer", 96, 38, 11, 17, "no_fly"),
        ],
        dynamic_obstacles=[
            DynamicObstacle(
                "inspection drone corridor",
                center=(61.0, 38.0),
                amplitude=(16.0, 0.0),
                period=31.0,
                radius=3.2,
            ),
            DynamicObstacle(
                "temporary crane hook",
                center=(82.0, 42.0),
                amplitude=(0.0, 8.0),
                period=27.0,
                radius=2.7,
                phase=0.8,
            ),
        ],
    )


def in_bounds(point: np.ndarray, scenario: Scenario) -> bool:
    x, y = point
    return 0 <= x <= scenario.width and 0 <= y <= scenario.height


def blocked(point: np.ndarray, scenario: Scenario, margin: float | None = None) -> bool:
    if margin is None:
        margin = scenario.safety_margin
    if not in_bounds(point, scenario):
        return True
    return any(obs.contains(point, margin=margin) for obs in scenario.static_obstacles)


def to_node(point: np.ndarray, scenario: Scenario) -> tuple[int, int]:
    return (
        int(round(point[0] / scenario.resolution)),
        int(round(point[1] / scenario.resolution)),
    )


def to_world(node: tuple[int, int], scenario: Scenario) -> np.ndarray:
    return np.array([node[0] * scenario.resolution, node[1] * scenario.resolution], dtype=float)


def astar(scenario: Scenario) -> list[np.ndarray]:
    start = to_node(scenario.start, scenario)
    goal = to_node(scenario.goal, scenario)
    max_x = int(scenario.width / scenario.resolution)
    max_y = int(scenario.height / scenario.resolution)

    def valid(node: tuple[int, int]) -> bool:
        if not (0 <= node[0] <= max_x and 0 <= node[1] <= max_y):
            return False
        return not blocked(to_world(node, scenario), scenario)

    def heuristic(node: tuple[int, int]) -> float:
        return math.hypot(node[0] - goal[0], node[1] - goal[1])

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    frontier: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not valid(nxt):
                continue
            step_cost = math.hypot(dx, dy)
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        raise RuntimeError("A* failed to find a path. Adjust the scenario obstacles.")

    path_nodes = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path_nodes.append(node)
        node = came_from[node]
    path_nodes.reverse()
    return [to_world(node, scenario) for node in path_nodes]


def segment_clear(a: np.ndarray, b: np.ndarray, scenario: Scenario) -> bool:
    distance = np.linalg.norm(b - a)
    samples = max(2, int(distance / scenario.resolution))
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


def ray_hit(
    origin: np.ndarray,
    angle: float,
    scenario: Scenario,
    dyn_centers: list[np.ndarray],
    lidar_range: float,
    step: float = 0.45,
) -> tuple[np.ndarray, float] | None:
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
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
    lidar_range: float = 11.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    repulsion = np.zeros(2, dtype=float)
    hits: list[np.ndarray] = []
    for angle in np.linspace(0, 2 * math.pi, 72, endpoint=False):
        hit = ray_hit(position, angle, scenario, dyn_centers, lidar_range)
        if hit is None:
            continue
        point, distance = hit
        hits.append(point)
        away = unit(position - point)
        strength = max(0.0, (1.0 / max(distance, 0.6) - 1.0 / lidar_range))
        repulsion += away * strength * strength * 36.0

    for center, dyn in zip(dyn_centers, scenario.dynamic_obstacles):
        offset = position - center
        clearance = np.linalg.norm(offset) - dyn.radius
        if clearance < lidar_range:
            repulsion += unit(offset) * ((lidar_range - clearance) / lidar_range) ** 2 * 2.0
    return repulsion, hits


def clearance(position: np.ndarray, scenario: Scenario, dyn_centers: list[np.ndarray]) -> float:
    static_clearance = min(
        obs.distance_to(position) - scenario.safety_margin for obs in scenario.static_obstacles
    )
    dynamic_clearance = min(
        np.linalg.norm(position - center) - dyn.radius - 1.2
        for center, dyn in zip(dyn_centers, scenario.dynamic_obstacles)
    )
    border_clearance = min(position[0], position[1], scenario.width - position[0], scenario.height - position[1])
    return float(min(static_clearance, dynamic_clearance, border_clearance))


def simulate(scenario: Scenario, waypoints: list[np.ndarray]) -> dict:
    dt = 0.28
    max_speed = 6.2
    reach_radius = 2.0
    lookahead_nodes = 3
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

    for step in range(1300):
        t = step * dt
        dyn_centers = [dyn.position(t) for dyn in scenario.dynamic_obstacles]
        dyn_log.append([center.tolist() for center in dyn_centers])
        current_clearance = clearance(position, scenario, dyn_centers)
        min_clearance = min(min_clearance, current_clearance)
        if current_clearance < -0.05:
            collision = True
            break

        if np.linalg.norm(position - scenario.goal) <= reach_radius:
            success = True
            break

        window_end = min(len(waypoints), active_idx + 28)
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
        repulsion_gain = 0.10 if current_clearance > 6.0 else 0.42
        command = 1.65 * attraction + repulsion_gain * unit(repulsion) * min(np.linalg.norm(repulsion), 2.2)
        if np.linalg.norm(command) < 1e-6:
            command = attraction

        local_clearance = max(0.0, current_clearance)
        speed_scale = 0.45 + 0.55 * min(local_clearance / 8.0, 1.0)
        velocity = unit(command) * max_speed * speed_scale
        next_position = position + velocity * dt

        if blocked(next_position, scenario, margin=0.0):
            velocity *= 0.35
            next_position = position + velocity * dt

        position = np.clip(next_position, [0.0, 0.0], [scenario.width, scenario.height])
        history.append(position.copy())
        active_waypoints.append(active_idx)
        lidar_log.append([point.tolist() for point in hits[:28]])
        speed_log.append(float(np.linalg.norm(velocity)))

    path_length = float(
        sum(np.linalg.norm(b - a) for a, b in zip(history, history[1:]))
    )
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
    }


def draw_base(ax, scenario: Scenario) -> None:
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east / m")
    ax.set_ylabel("north / m")
    ax.grid(True, color="#d7dde5", linewidth=0.6)
    ax.set_facecolor("#f8fafc")

    colors = {
        "building": ("#58677b", "#334155"),
        "no_fly": ("#e7a06f", "#b45309"),
    }
    for obs in scenario.static_obstacles:
        face, edge = colors[obs.kind]
        patch = Rectangle((obs.x, obs.y), obs.w, obs.h, facecolor=face, edgecolor=edge, alpha=0.88)
        ax.add_patch(patch)
        margin = Rectangle(
            (obs.x - scenario.safety_margin, obs.y - scenario.safety_margin),
            obs.w + 2 * scenario.safety_margin,
            obs.h + 2 * scenario.safety_margin,
            facecolor="none",
            edgecolor=edge,
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
        )
        ax.add_patch(margin)

    ax.scatter(*scenario.start, marker="o", s=80, c="#16a34a", label="hospital start", zorder=5)
    ax.scatter(*scenario.goal, marker="*", s=180, c="#dc2626", label="delivery target", zorder=5)


def plot_final(scenario: Scenario, global_path: list[np.ndarray], waypoints: list[np.ndarray], result: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    draw_base(ax, scenario)
    path = np.array(global_path)
    simplified = np.array(waypoints)
    history = result["history"]
    ax.plot(path[:, 0], path[:, 1], color="#94a3b8", linewidth=1.0, linestyle=":", label="A* grid path")
    ax.plot(
        simplified[:, 0],
        simplified[:, 1],
        color="#475569",
        linewidth=1.6,
        linestyle="--",
        label="smoothed waypoints",
    )
    ax.plot(history[:, 0], history[:, 1], color="#2563eb", linewidth=2.4, label="executed trajectory")

    final_t = result["flight_time_s"]
    for dyn in scenario.dynamic_obstacles:
        center = dyn.position(final_t)
        ax.add_patch(Circle(center, dyn.radius, facecolor="#facc15", edgecolor="#a16207", alpha=0.6))

    status = "success" if result["success"] else "failed"
    ax.set_title(
        f"Low-altitude emergency delivery simulation - {status}, "
        f"min clearance {result['min_clearance']:.2f} m"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUTPUT_DIR / "final_trajectory.png", dpi=180)
    plt.close(fig)


def save_animation(scenario: Scenario, global_path: list[np.ndarray], waypoints: list[np.ndarray], result: dict) -> None:
    history = result["history"]
    frame_indices = np.unique(np.linspace(0, len(history) - 1, min(170, len(history)), dtype=int))
    fig, ax = plt.subplots(figsize=(9.12, 6.08), constrained_layout=True)
    draw_base(ax, scenario)

    path = np.array(global_path)
    simplified = np.array(waypoints)
    ax.plot(path[:, 0], path[:, 1], color="#cbd5e1", linewidth=0.8, linestyle=":")
    ax.plot(simplified[:, 0], simplified[:, 1], color="#64748b", linewidth=1.1, linestyle="--")
    trail, = ax.plot([], [], color="#2563eb", linewidth=2.2)
    drone = Circle((history[0, 0], history[0, 1]), 1.4, facecolor="#2563eb", edgecolor="#1e3a8a", zorder=6)
    ax.add_patch(drone)
    lidar_lines = [ax.plot([], [], color="#38bdf8", linewidth=0.45, alpha=0.45)[0] for _ in range(28)]
    dyn_patches = [
        Circle((0, 0), dyn.radius, facecolor="#facc15", edgecolor="#a16207", alpha=0.65, zorder=4)
        for dyn in scenario.dynamic_obstacles
    ]
    for patch in dyn_patches:
        ax.add_patch(patch)
    title = ax.text(0.01, 1.015, "", transform=ax.transAxes, fontsize=10, va="bottom")

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])
        pos = history[idx]
        trail.set_data(history[: idx + 1, 0], history[: idx + 1, 1])
        drone.center = (pos[0], pos[1])
        dyn_centers = result["dyn_log"][min(idx, len(result["dyn_log"]) - 1)]
        for patch, center in zip(dyn_patches, dyn_centers):
            patch.center = center
        hits = result["lidar_log"][min(idx, len(result["lidar_log"]) - 1)] if result["lidar_log"] else []
        for line, hit in zip(lidar_lines, hits):
            line.set_data([pos[0], hit[0]], [pos[1], hit[1]])
        for line in lidar_lines[len(hits) :]:
            line.set_data([], [])
        title.set_text(f"t={idx * 0.32:5.1f}s | step={idx:03d}")
        return [trail, drone, title, *lidar_lines, *dyn_patches]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=True)
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
    summary = {
        "scenario": "urban_low_altitude_emergency_delivery",
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
    result = simulate(scenario, global_path)

    plot_final(scenario, global_path, waypoints, result)
    save_animation(scenario, global_path, waypoints, result)
    save_summary(scenario, global_path, waypoints, result)

    print("Simulation complete")
    print(f"success: {result['success']}")
    print(f"collision: {result['collision']}")
    print(f"flight time: {result['flight_time_s']:.2f}s")
    print(f"path length: {result['path_length']:.2f}m")
    print(f"minimum clearance: {result['min_clearance']:.2f}m")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
