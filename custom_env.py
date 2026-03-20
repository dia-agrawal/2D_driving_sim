# custom_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from heapq import heappush, heappop

from map import generate_city_map, ROAD, BUILDING, GRASS


class DrivingENV(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, size: int = 100, render_mode: str = "none"):
        super().__init__()

        self.size = size
        self.dt = 0.05
        self.v_max = 30.0
        self.max_steps = 5000
        self.render_mode = render_mode

        # dynamics knobs
        self.accel_gain = 10.0  # was effectively 2.0; higher makes learning much easier

        # steering knobs (used now)
        self.turn_gain = 8.0        # try 6–15
        self.turn_eps = 0.05        # base steer even at low speed
        self.v_turn = 5.0           # speed at which turning reaches "full strength"

        # Debug (prints ONCE per episode when episode ends)
        self.debug_reward = False
        self._episode_id = 0
        self._debug_printed = False
        self._debug_totals = {}

        # Heading-alignment reward (turning toward waypoint)
        self.k_align = 0.10         # try 0.05–0.20
        self.align_v_min = 0.5      # only apply when moving

        self.grid, self.x_nodes, self.y_nodes = generate_city_map(
            size=self.size, block=10, road_w=2, n_grass=60, seed=0
        )

        # Unique centerlines (for neighbor lookup)
        self.x_centers = sorted(set(self.x_nodes))
        self.y_centers = sorted(set(self.y_nodes))

        # Continuous agent state
        self.x = 0.0
        self.y = 0.0
        self._agent_yaw = 0.0
        self._agent_velocity = 0.0
        self.target = np.zeros(2, dtype=np.float32)
        self.steps = 0

        # A* / guidance state
        self.path_xy = []             # list of (x,y) waypoints
        self.wp_idx = 0               # current waypoint index
        self.wp_final_bonus = 5.0     # paid ONCE when reaching the last waypoint
        self._final_wp_paid = False

        self.wp_spacing = 1.0
        self.wp_radius = 1.0
        self.wp_reward = 0.5

        # Observation/action spaces
        low_agent = np.array([0.0, 0.0, -np.pi, 0.0], dtype=np.float32)
        high_agent = np.array([self.size - 1, self.size - 1, np.pi, self.v_max], dtype=np.float32)

        # guide = [dx_to_wp, dy_to_wp, heading_error, dist_to_wp]
        # NOTE: dx/dy are in the AGENT frame (forward/lateral), not global world frame.
        low_guide = np.array([-(self.size - 1), -(self.size - 1), -np.pi, 0.0], dtype=np.float32)
        high_guide = np.array([(self.size - 1), (self.size - 1), np.pi, np.sqrt(2.0) * (self.size - 1)],
                              dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=low_agent, high=high_agent, dtype=np.float32),
                "target": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([self.size - 1, self.size - 1], dtype=np.float32),
                    dtype=np.float32,
                ),
                "waypoint": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([self.size - 1, self.size - 1], dtype=np.float32),
                    dtype=np.float32,
                ),
                "guide": spaces.Box(low=low_guide, high=high_guide, dtype=np.float32),
            }
        )

        # action = [turn_intent, accel_intent]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Render state
        self._fig = None
        self._ax = None
        self._im = None
        self._target_dot = None
        self._wp_dot = None
        self._car_patch = None

    # --------------------------
    # Helpers
    # --------------------------
    def _wrap_pi(self, ang: float) -> float:
        while ang > np.pi:
            ang -= 2.0 * np.pi
        while ang < -np.pi:
            ang += 2.0 * np.pi
        return ang

    def _densify_path(self, coarse_xy, spacing=1.0):
        if not coarse_xy or len(coarse_xy) < 2:
            return coarse_xy

        dense = [coarse_xy[0]]
        for (x0, y0), (x1, y1) in zip(coarse_xy[:-1], coarse_xy[1:]):
            dx = x1 - x0
            dy = y1 - y0
            dist = float(np.hypot(dx, dy))
            if dist < 1e-6:
                continue
            n = max(1, int(np.ceil(dist / spacing)))
            for k in range(1, n + 1):
                t = k / n
                dense.append((x0 + t * dx, y0 + t * dy))
        return dense

    def _debug_reset(self):
        self._debug_totals = {
            "total_reward": 0.0,
            "turn_penalty": 0.0,
            "standstill_penalty": 0.0,
            "time_penalty": 0.0,
            "forward_reward": 0.0,
            "align_reward": 0.0,
            "road_reward": 0.0,
            "grass_penalty": 0.0,
            "building_penalty": 0.0,
            "waypoint_reward": 0.0,
            "final_bonus": 0.0,
            "goal_bonus": 0.0,
            "speed_bonus": 0.0,
        }
        self._debug_printed = False

    def _debug_add(self, key, val):
        if self.debug_reward:
            self._debug_totals[key] += float(val)

    def _debug_maybe_print(self, terminated, truncated):
        if (not self.debug_reward) or self._debug_printed:
            return
        if not (terminated or truncated):
            return
        t = self._debug_totals
        print(f"\n=== Episode {self._episode_id} Reward Summary (steps={self.steps}) ===")
        for k in [
            "total_reward", "time_penalty", "standstill_penalty", "turn_penalty",
            "forward_reward", "speed_bonus", "align_reward",
            "road_reward", "grass_penalty", "building_penalty",
            "waypoint_reward", "final_bonus", "goal_bonus"
        ]:
            print(f"{k:18s}: {t[k]:.3f}")
        print("===========================================\n")
        self._debug_printed = True

    def _sample_target(self):
        for _ in range(2000):
            t = self.np_random.uniform(0, self.size - 1, size=2).astype(np.float32)
            cx, cy = int(t[0]), int(t[1])
            if self.grid[cy, cx] == ROAD:
                return t
        return self.np_random.uniform(0, self.size - 1, size=2).astype(np.float32)

    def _nearest_center_ix(self, x: float) -> int:
        arr = np.asarray(self.x_centers, dtype=np.float32)
        return int(np.argmin(np.abs(arr - x)))

    def _nearest_center_iy(self, y: float) -> int:
        arr = np.asarray(self.y_centers, dtype=np.float32)
        return int(np.argmin(np.abs(arr - y)))

    def _node_xy(self, ix: int, iy: int):
        return float(self.x_centers[ix]), float(self.y_centers[iy])

    def _is_road_node(self, ix: int, iy: int) -> bool:
        x, y = self._node_xy(ix, iy)
        cx = int(np.clip(int(round(x)), 0, self.size - 1))
        cy = int(np.clip(int(round(y)), 0, self.size - 1))
        return int(self.grid[cy, cx]) == ROAD

    def _neighbors(self, ix: int, iy: int):
        # Filter neighbors to road nodes to avoid weird guidance paths
        candidates = [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]
        for nx, ny in candidates:
            if 0 <= nx < len(self.x_centers) and 0 <= ny < len(self.y_centers):
                if self._is_road_node(nx, ny):
                    yield (nx, ny)

    def _edge_cost(self, a, b) -> float:
        ax, ay = self._node_xy(a[0], a[1])
        bx, by = self._node_xy(b[0], b[1])
        return abs(bx - ax) + abs(by - ay)

    def _heuristic(self, node, goal) -> float:
        nx, ny = self._node_xy(node[0], node[1])
        gx, gy = self._node_xy(goal[0], goal[1])
        return float(np.hypot(gx - nx, gy - ny))

    def _astar(self, start, goal):
        if start == goal:
            return [start]

        open_heap = []
        heappush(open_heap, (self._heuristic(start, goal), 0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, g, cur = heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            closed.add(cur)
            for nb in self._neighbors(cur[0], cur[1]):
                if nb in closed:
                    continue
                tentative_g = g_score[cur] + self._edge_cost(cur, nb)
                if (nb not in g_score) or (tentative_g < g_score[nb]):
                    came_from[nb] = cur
                    g_score[nb] = tentative_g
                    heappush(open_heap, (tentative_g + self._heuristic(nb, goal), tentative_g, nb))

        return [start]

    def _compute_guidance(self):
        if not self.path_xy:
            wp = (self.x, self.y)
        else:
            self.wp_idx = int(np.clip(self.wp_idx, 0, len(self.path_xy) - 1))
            wp = self.path_xy[self.wp_idx]

        dx_w = float(wp[0] - self.x)
        dy_w = float(wp[1] - self.y)
        dist = float(np.hypot(dx_w, dy_w))

        desired = float(np.arctan2(dy_w, dx_w))
        heading_err = self._wrap_pi(desired - self._agent_yaw)

        # advance waypoint index if already close
        while self.path_xy and dist <= self.wp_radius and self.wp_idx < len(self.path_xy) - 1:
            self.wp_idx += 1
            wp = self.path_xy[self.wp_idx]
            dx_w = float(wp[0] - self.x)
            dy_w = float(wp[1] - self.y)
            dist = float(np.hypot(dx_w, dy_w))
            desired = float(np.arctan2(dy_w, dx_w))
            heading_err = self._wrap_pi(desired - self._agent_yaw)

        # Convert (dx,dy) to agent frame to avoid "always go right" bias
        c = float(np.cos(self._agent_yaw))
        s = float(np.sin(self._agent_yaw))
        dx_a = c * dx_w + s * dy_w          # forward component
        dy_a = -s * dx_w + c * dy_w         # lateral component (left-positive)

        guide = np.array([dx_a, dy_a, heading_err, dist], dtype=np.float32)
        waypoint = np.array([float(wp[0]), float(wp[1])], dtype=np.float32)
        return waypoint, guide

    # --------------------------
    # Gym API
    # --------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        self._episode_id += 1
        self._debug_reset()
        self._final_wp_paid = False

        # Start near middle, snapped to nearest intersection
        self.x = self.size / 2.0
        self.y = self.size / 2.0
        ix0 = self._nearest_center_ix(self.x)
        iy0 = self._nearest_center_iy(self.y)
        self.x, self.y = self._node_xy(ix0, iy0)

        self._agent_yaw = 0.0
        self._agent_velocity = 0.0

        # Target and goal node
        self.target = self._sample_target()
        ixg = self._nearest_center_ix(float(self.target[0]))
        iyg = self._nearest_center_iy(float(self.target[1]))

        start = (ix0, iy0)
        goal = (ixg, iyg)

        # Run A* once per episode
        path_nodes = self._astar(start, goal)
        coarse = [self._node_xy(ix, iy) for (ix, iy) in path_nodes]
        self.path_xy = self._densify_path(coarse, spacing=self.wp_spacing)

        self.wp_idx = 1 if len(self.path_xy) >= 2 else 0

        waypoint, guide = self._compute_guidance()
        obs = {
            "agent": np.array([self.x, self.y, self._agent_yaw, self._agent_velocity], dtype=np.float32),
            "target": self.target,
            "waypoint": waypoint,
            "guide": guide,
        }

        if self.render_mode == "human":
            self.render()

        return obs, {}

    def step(self, action):
        self.steps += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        terminated = False
        truncated = False
        reward = 0.0

        # Save previous for shaping
        prev_x, prev_y = self.x, self.y
        prev_yaw = self._agent_yaw

        wx = wy = None
        if self.path_xy:
            wx, wy = self.path_xy[int(np.clip(self.wp_idx, 0, len(self.path_xy) - 1))]

        head_err_prev = None
        if wx is not None:
            desired_prev = float(np.arctan2(wy - prev_y, wx - prev_x))
            head_err_prev = self._wrap_pi(desired_prev - prev_yaw)

        # --------------------------
        # DYNAMICS
        # --------------------------
        self._agent_velocity = float(np.clip(
            self._agent_velocity + self.accel_gain * action[1] * self.dt,
            0.0, self.v_max
        ))

        # speed-dependent steering strength
        steer_strength = self.turn_gain * (self.turn_eps + (self._agent_velocity / (self._agent_velocity + self.v_turn)))
        self._agent_yaw = self._wrap_pi(self._agent_yaw + steer_strength * action[0] * self.dt)

        self.x += self._agent_velocity * np.cos(self._agent_yaw) * self.dt
        self.y += self._agent_velocity * np.sin(self._agent_yaw) * self.dt
        self.x = float(np.clip(self.x, 0.0, self.size - 1.0))
        self.y = float(np.clip(self.y, 0.0, self.size - 1.0))

        # time penalty
        time_penalty = -0.001
        reward += time_penalty
        self._debug_add("time_penalty", time_penalty)

        # --------------------------
        # TURN PENALTY
        # --------------------------
        yaw_delta = self._wrap_pi(self._agent_yaw - prev_yaw)
        turn_penalty = -0.01 * abs(yaw_delta)
        reward += turn_penalty
        self._debug_add("turn_penalty", turn_penalty)

        # discourage standing still (make it meaningful)
        if self._agent_velocity < 0.5:
            standstill = -0.05
            reward += standstill
            self._debug_add("standstill_penalty", standstill)

        # --------------------------
        # FORWARD PROGRESS TOWARD WAYPOINT
        # --------------------------
        speed_bonus = 0.0  # IMPORTANT: avoid UnboundLocalError
        if wx is not None:
            to_wp = np.array([wx - prev_x, wy - prev_y], dtype=np.float32)
            u = to_wp / (np.linalg.norm(to_wp) + 1e-6)

            step_vec = np.array([self.x - prev_x, self.y - prev_y], dtype=np.float32)
            forward = float(np.dot(step_vec, u))  # >0 toward waypoint, <0 away

            forward_reward = 1.0 * forward
            reward += forward_reward
            self._debug_add("forward_reward", forward_reward)

            # only reward speed if actually progressing
            if forward > 0:
                speed_bonus = 0.02 * float(self._agent_velocity)
                reward += speed_bonus
            self._debug_add("speed_bonus", speed_bonus)

        # Heading alignment shaping (only if moving)
        if (wx is not None) and (head_err_prev is not None) and (self._agent_velocity > self.align_v_min):
            desired_new = float(np.arctan2(wy - self.y, wx - self.x))
            head_err_new = self._wrap_pi(desired_new - self._agent_yaw)
            align_reward = self.k_align * (abs(head_err_prev) - abs(head_err_new))
            reward += align_reward
            self._debug_add("align_reward", align_reward)

        # --------------------------
        # TILE REWARD / TERMINATION
        # --------------------------
        cx = int(self.x)
        cy = int(self.y)
        cell = int(self.grid[cy, cx])

        if cell == BUILDING:
            building = -10.0
            reward += building
            self._debug_add("building_penalty", building)
            terminated = True
        elif cell == GRASS:
            grass = -0.05
            reward += grass
            self._debug_add("grass_penalty", grass)
        else:
            road = 0.0
            reward += road
            self._debug_add("road_reward", road)

        # --------------------------
        # GOAL CONDITION
        # --------------------------
        dist_to_target = float(np.linalg.norm(np.array([self.x, self.y], dtype=np.float32) - self.target))
        if dist_to_target < 1.5:
            goal_bonus = 50.0
            reward += goal_bonus
            self._debug_add("goal_bonus", goal_bonus)
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        # --------------------------
        # WAYPOINT REWARD (no farming)
        # --------------------------
        if self.path_xy:
            wx2, wy2 = self.path_xy[int(np.clip(self.wp_idx, 0, len(self.path_xy) - 1))]
            dist_to_wp = float(np.hypot(wx2 - self.x, wy2 - self.y))

            if dist_to_wp <= self.wp_radius:
                if self.wp_idx < len(self.path_xy) - 1:
                    reward += self.wp_reward
                    self._debug_add("waypoint_reward", self.wp_reward)
                    self.wp_idx += 1
                else:
                    # last waypoint: pay once only
                    if not self._final_wp_paid:
                        reward += self.wp_reward
                        self._debug_add("waypoint_reward", self.wp_reward)
                        reward += self.wp_final_bonus
                        self._debug_add("final_bonus", self.wp_final_bonus)
                        self._final_wp_paid = True

        waypoint, guide = self._compute_guidance()
        obs = {
            "agent": np.array([self.x, self.y, self._agent_yaw, self._agent_velocity], dtype=np.float32),
            "target": self.target,
            "waypoint": waypoint,
            "guide": guide,
        }

        info = {
            "cell": cell,
            "dist_to_target": dist_to_target,
            "wp_idx": self.wp_idx,
            "path_len": len(self.path_xy),
            "speed": float(self._agent_velocity),
            "action_turn": float(action[0]),
            "action_accel": float(action[1]),
        }

        self._debug_add("total_reward", reward)
        self._debug_maybe_print(terminated, truncated)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots()
            self._im = self._ax.imshow(self.grid, origin="lower")
            self._car_patch = Polygon([[0, 0], [0, 0], [0, 0]], closed=True)
            self._ax.add_patch(self._car_patch)
            self._target_dot, = self._ax.plot([], [], marker="x")
            self._wp_dot, = self._ax.plot([], [], linestyle="None", marker="o", markersize=3)
            self._ax.set_title("DrivingENV")

        # Car triangle
        L = 1.25
        W = 0.625
        dx = np.cos(self._agent_yaw)
        dy = np.sin(self._agent_yaw)

        nose = (self.x + L * dx, self.y + L * dy)
        left = (self.x - 0.5 * L * dx - W * dy, self.y - 0.5 * L * dy + W * dx)
        right = (self.x - 0.5 * L * dx + W * dy, self.y - 0.5 * L * dy - W * dx)
        self._car_patch.set_xy([nose, left, right])

        # Target + current waypoint
        self._target_dot.set_data([float(self.target[0])], [float(self.target[1])])

        if self.path_xy:
            wx, wy = self.path_xy[int(np.clip(self.wp_idx, 0, len(self.path_xy) - 1))]
        else:
            wx, wy = self.x, self.y
        self._wp_dot.set_data([float(wx)], [float(wy)])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._im = None
            self._target_dot = None
            self._wp_dot = None
            self._car_patch = None