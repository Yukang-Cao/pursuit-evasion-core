"""Common route-guided MPPI approach outside the local finishing region.

The MPPI noise and softmax update are inherited from the existing C-Free-Uniform
adaptation. The route is a navigation heuristic; it is not a pursuit game solver.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from ..reversible.mppi import ReversibleMPPI, MPPISettings
from ..reversible.model import vehicle_step
from .native import GeometryEngine


class RouteGuide:
    def __init__(self, arena, role, resolution=0.15, lookahead=2.0):
        self.arena = arena
        self.role = role
        self.resolution = resolution
        self.lookahead = lookahead
        self.version = None
        n = int(np.ceil(2 * arena.half_width / resolution))
        self.axis = -arena.half_width + (np.arange(n) + 0.5) * resolution
        x, y = np.meshgrid(self.axis, self.axis)
        self.xy = np.stack([x, y], -1)
        self.n = n
        self.cache = {}

    def update(self, memory, *, ideal=False):
        version = (
            ("ideal",)
            if ideal
            else ("observed", int(np.count_nonzero(memory.cells == 2)))
        )
        if version == self.version:
            return
        b = getattr(self.arena, self.role)
        if ideal:
            free = self.arena.point_clearance(self.xy) > b.width / 2 + 1e-7
        else:
            occ = memory.cells == 2
            dist = distance_transform_edt(~occ) * memory.settings.memory_resolution
            ix = np.floor(
                (self.xy[..., 0] - memory.origin[0]) / memory.settings.memory_resolution
            ).astype(int)
            iy = np.floor(
                (self.xy[..., 1] - memory.origin[1]) / memory.settings.memory_resolution
            ).astype(int)
            free = (dist[iy, ix] > b.width / 2) & (
                abs(self.xy).max(-1) < self.arena.half_width - b.width / 2
            )
        self.free = free
        self.nodes = np.flatnonzero(free)
        self.positions = self.xy.reshape(-1, 2)[self.nodes]
        if not len(self.nodes):
            raise RuntimeError("No navigation cells")
        self.tree = cKDTree(self.positions)
        self.node_index = np.full(self.n * self.n, -1, int)
        self.node_index[self.nodes] = np.arange(len(self.nodes))
        components, _ = label(free)
        self.components = components.ravel()[self.nodes]
        rows = []
        cols = []
        weights = []
        for dy, dx in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            y, x = np.divmod(self.nodes, self.n)
            ny, nx = y + dy, x + dx
            valid = (ny >= 0) & (nx >= 0) & (ny < self.n) & (nx < self.n)
            dest = np.clip(ny, 0, self.n - 1) * self.n + np.clip(nx, 0, self.n - 1)
            valid &= self.node_index[dest] >= 0
            if dx and dy:
                valid &= (
                    free[y, np.clip(nx, 0, self.n - 1)]
                    & free[np.clip(ny, 0, self.n - 1), x]
                )
            rows.extend(np.flatnonzero(valid))
            cols.extend(self.node_index[dest[valid]])
            weights.extend([self.resolution * np.hypot(dx, dy)] * int(valid.sum()))
        self.graph = csr_matrix(
            (weights, (rows, cols)), shape=(len(self.nodes), len(self.nodes))
        )
        self.cache = {}
        self.version = version

    def nearest(self, xy):
        return int(self.tree.query(np.asarray(xy))[1])

    def distances(self, goal):
        if goal not in self.cache:
            if len(self.cache) > 12:
                self.cache.clear()
            self.cache[goal] = dijkstra(
                self.graph, directed=False, indices=goal, return_predecessors=True
            )
        return self.cache[goal]

    def pursuit(self, pose, evader):
        start = self.nearest(pose[:2])
        component = self.components[start]
        eligible = np.flatnonzero(self.components == component)
        goal = int(
            eligible[
                np.argmin(np.linalg.norm(self.positions[eligible] - evader[:2], axis=1))
            ]
        )
        distance, predecessor = self.distances(goal)
        path = [start]
        for _ in range(len(self.nodes)):
            nxt = int(predecessor[path[-1]])
            if nxt < 0:
                break
            path.append(nxt)
            if nxt == goal:
                break
        xy = self.positions[path]
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        k = min(int(np.searchsorted(length, self.lookahead)), len(xy) - 1)
        target = xy[k]
        direction = xy[k + 1] - target if k + 1 < len(xy) else evader[:2] - target
        if np.linalg.norm(direction) < 1e-8:
            direction = target - pose[:2]
        heading = float(np.arctan2(direction[1], direction[0]))
        return target, heading, xy

    def escape(self, pose, pursuer):
        start = self.nearest(pose[:2])
        eligible = np.flatnonzero(self.components == self.components[start])
        origin = int(
            eligible[
                np.argmin(
                    np.linalg.norm(self.positions[eligible] - pursuer[:2], axis=1)
                )
            ]
        )
        dp, _ = self.distances(origin)
        de, _ = self.distances(start)
        candidates = eligible[
            (de[eligible] <= self.lookahead) & np.isfinite(dp[eligible])
        ]
        if not len(candidates):
            return pose[:2].copy(), float(pose[2]), np.asarray([pose[:2]])
        radial = np.linalg.norm(self.positions[candidates] - pursuer[:2], axis=1)
        chosen = int(
            candidates[np.lexsort((de[candidates], -radial, -dp[candidates]))[0]]
        )
        target = self.positions[chosen]
        d = target - pose[:2]
        return (
            target,
            (
                float(np.arctan2(d[1], d[0]))
                if np.linalg.norm(d) > 1e-8
                else float(pose[2])
            ),
            np.asarray([pose[:2], target]),
        )


class FootprintMPPI(ReversibleMPPI):
    def __init__(self, config, arena, role="pursuer", seed=0, settings=None):
        settings = settings or MPPISettings(
            samples=512,
            horizon=2.0,
            sampler="log_mppi",
            noise_std=(3.0, getattr(config, role).max_curvature),
        )
        super().__init__(
            config,
            settings,
            seed,
            role=role,
            clearance=lambda x: np.full(np.shape(x)[:-1], np.inf),
        )
        self.role = role
        self.goal_heading = 0.0
        self.pose_for_sampling = None
        self.require_forward = False
        self.update_geometry(arena)

    def update_geometry(self, arena):
        self.engine = GeometryEngine(
            arena, self.limits, self.role, threads=4, control_dt=self.config.control_dt
        )

    def sample_noise(self, shape):
        noise = super().sample_noise(shape)
        seeds = []
        a = self.limits.acceleration
        k = self.limits.max_curvature
        for sign in [-1, 1]:
            for fraction in [0.25, 0.5, 0.75]:
                split = round(self.steps * fraction)
                u = np.tile([a, sign * k], (self.steps, 1))
                u[split:, 1] = 0
                seeds.append(u)
                u = np.tile([-a, -sign * k], (self.steps, 1))
                u[split:] = [a, sign * k]
                seeds.append(u)
        for curvature in [-k, 0, k]:
            u = np.zeros((self.steps, 2))
            s = self.pose_for_sampling.copy()
            for t in range(self.steps):
                u[t] = [
                    -np.sign(s[3]) * min(a, abs(s[3]) / self.config.control_dt),
                    curvature,
                ]
                s = vehicle_step(s, u[t], self.config.control_dt, self.limits)
            seeds.append(u)
        for i, u in enumerate(seeds, 10):
            if i >= shape[1]:
                break
            noise[:, i] = u - self.nominal
        return noise

    def _rollout(self, pose, commands, target, *, return_paths=True):
        commands = np.asarray(commands)
        n = len(commands)
        paths = np.empty((self.steps + 1, n, 4))
        paths[0] = pose
        blocked = np.zeros((self.steps + 1, n), bool)
        goal = np.zeros_like(blocked)
        seen_bad = np.zeros(n, bool)
        seen_goal = np.zeros(n, bool)
        radius = self.limits.minimum_turn_radius
        target_front = target + radius * np.array(
            [np.cos(self.goal_heading), np.sin(self.goal_heading)]
        )
        distance = np.empty((self.steps + 1, n))
        goal_step = np.full(n, self.steps + 1)
        blocked_step = np.full(n, self.steps + 1)
        for t in range(self.steps + 1):
            if t:
                after, safe = self.engine.matched_steps(
                    paths[t - 1], commands[:, t - 1], self.config.control_dt
                )
                paths[t] = after
                active = ~seen_bad & (~seen_goal if t > 1 else True)
                hit = active & ~safe
                blocked[t] = hit
                blocked_step[hit] = np.minimum(blocked_step[hit], t)
                seen_bad |= hit
            s = paths[t]
            center = np.sum((s[:, :2] - target) ** 2, axis=1)
            front = s[:, :2] + radius * np.stack([np.cos(s[:, 2]), np.sin(s[:, 2])], -1)
            front_error = np.sum((front - target_front) ** 2, axis=1)
            distance[t] = center + front_error
            reached = (
                (center <= self.settings.goal_tolerance**2)
                & (front_error <= self.settings.goal_tolerance**2)
                & ~seen_bad
            )
            newly = reached & ~seen_goal
            goal_step[newly] = t
            seen_goal |= reached
            goal[t] = seen_goal
        bad = np.maximum.accumulate(blocked, axis=0)
        terminated = bad | goal
        costs = np.where(terminated, 0, self.settings.distance_weight * distance).sum(0)
        costs += (bad * self.settings.obstacle_penalty).sum(0)
        costs += np.where(
            ~(seen_bad | seen_goal), self.settings.terminal_weight * distance[-1], 0
        )
        valid = ~seen_bad
        safe_steps = np.minimum(blocked_step - 1, self.steps)
        final = paths[np.minimum(goal_step, self.steps), np.arange(n)]
        valid &= self.engine.backup_safe(final)
        forward = (final[:, 3] >= -1e-8) & (np.cos(final[:, 2] - self.goal_heading) > 0)
        if n > 1:
            self.require_forward = bool(np.any(valid & forward))
        if self.require_forward:
            valid &= forward
        return costs, valid, safe_steps, np.swapaxes(paths, 0, 1)

    def command_pose(self, pose, target, heading):
        self.goal_heading = float(heading)
        self.pose_for_sampling = np.asarray(pose, float).copy()
        self.require_forward = False
        command = super().command(pose, np.asarray(target), diagnostics=False)
        self.last.update(
            geometry_backend="oriented rectangles against observed surface points",
            pose_goal="center and a forward reference point at minimum turn radius",
            prefer_forward_facing=self.require_forward,
        )
        return command
