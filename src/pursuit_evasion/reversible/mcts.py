"""Small adversarial UCT prototype with numerical-DP leaf guidance.

P chooses a current command, E observes it and chooses its response, then both
move for one control interval. The next P decision can depend on the observed
successor. There is no future open-loop P sequence and no nested MPPI rollout.

This follows the P/E node separation and confidence-bound signs of the earlier
research/tree_capture/engine.cpp implementation, with a small inspectable Python
tree and the new shared world-state physics. See docs/reversible-uct.md.
"""

from dataclasses import dataclass, field
import time
import numpy as np

from .controllers import action_catalogue, braking_safe_mask, table_states
from .game import WorldMap, advance


@dataclass(frozen=True)
class SearchStep:
    state: object
    cost: float
    terminal: bool = False
    event: str | None = None


@dataclass
class _Edge:
    visits: int = 0
    total: float = 0.0
    cost: float = 0.0
    child: object = None
    terminal: bool = False

    @property
    def mean(self):
        return self.total / self.visits if self.visits else np.nan


@dataclass
class _Node:
    state: object
    depth: int
    player: str
    actions: tuple | None
    p_action: int | None = None
    index: int = 0
    visits: int = 0
    edges: dict = field(default_factory=dict)
    initial_value: float | None = None
    exact_value: float | None = None


class AdversarialUCT:
    """Finite-action deterministic tree used by the vehicle adapter and tests.

    The callbacks are deliberately small: legal P actions, legal E actions,
    one joint transition, and a frontier value. `depth` counts complete joint
    command intervals, not individual P/E choices. Exactness, when reported,
    concerns only this finite-depth game with the supplied leaf heuristic.
    """

    def __init__(
        self,
        state,
        p_actions,
        e_actions,
        transition,
        leaf_value,
        depth=6,
        exploration=0.25,
        seed=0,
        state_key=None,
    ):
        if depth < 1 or exploration < 0 or not np.isfinite(exploration):
            raise ValueError("Depth must be positive and exploration nonnegative.")
        self.p_actions = p_actions
        self.e_actions = e_actions
        self.transition = transition
        self.leaf_value = leaf_value
        self.depth = int(depth)
        self.exploration = float(exploration)
        self.rng = np.random.default_rng(seed)
        self.state_key = state_key or self._default_key
        self.nodes = []
        self.p_cache = {}
        self.transitions = 0
        self.leaf_queries = 0
        self.terminal_events = {}
        self.deepest = 0
        self.root, _ = self._p_node(state, 0)
        self._ensure_actions(self.root)

    @staticmethod
    def _default_key(state):
        if isinstance(state, np.ndarray):
            # Do not merge merely nearby states across sharp value boundaries.
            return (state.shape, state.dtype.str, state.tobytes())
        return state

    def _leaf(self, state):
        value = float(self.leaf_value(state))
        self.leaf_queries += 1
        if not np.isfinite(value):
            raise ValueError(
                "UCT leaf callback must return a finite declared estimate."
            )
        return value

    def _p_node(self, state, depth):
        key = (depth, self.state_key(state))
        if key in self.p_cache:
            return self.p_cache[key], False
        # A newly reached frontier needs a leaf value, not yet a full set of
        # braking checks for actions that this simulation will not execute.
        actions = None if depth < self.depth else ()
        node = _Node(state, depth, "P", actions, index=len(self.nodes))
        node.initial_value = self._leaf(state)
        if depth >= self.depth:
            node.exact_value = node.initial_value
        self.nodes.append(node)
        self.p_cache[key] = node
        self.deepest = max(self.deepest, depth)
        return node, True

    def _ensure_actions(self, node):
        if node.actions is None:
            node.actions = tuple(self.p_actions(node.state))
            if not node.actions:
                node.exact_value = node.initial_value

    def _e_node(self, parent, p_action):
        actions = tuple(self.e_actions(parent.state, p_action))
        if not actions:
            raise ValueError("An E response node requires at least one action.")
        node = _Node(
            parent.state,
            parent.depth,
            "E",
            actions,
            p_action=p_action,
            index=len(self.nodes),
        )
        self.nodes.append(node)
        return node

    def _select(self, node):
        unvisited = [
            a for a in node.actions if a not in node.edges or not node.edges[a].visits
        ]
        if unvisited:
            return unvisited[int(self.rng.integers(len(unvisited)))]
        log_visits = np.log(node.visits + 1.0)
        means = np.array([node.edges[a].mean for a in node.actions])
        bonus = self.exploration * np.sqrt(
            log_visits / np.array([node.edges[a].visits for a in node.actions])
        )
        estimates = means - bonus if node.player == "P" else means + bonus
        index = int(
            np.argmin(estimates) if node.player == "P" else np.argmax(estimates)
        )
        return node.actions[index]

    @staticmethod
    def _refresh_exact(node):
        if (
            node.actions is None
            or node.exact_value is not None
            or len(node.edges) != len(node.actions)
        ):
            return
        values = []
        for action in node.actions:
            edge = node.edges[action]
            if edge.terminal:
                values.append(edge.cost)
            elif edge.child is not None and edge.child.exact_value is not None:
                values.append(edge.cost + edge.child.exact_value)
            else:
                return
        if values:
            node.exact_value = min(values) if node.player == "P" else max(values)

    def simulate(self):
        node = self.root
        path = []
        while True:
            self._ensure_actions(node)
            self._refresh_exact(node)
            if node.exact_value is not None:
                result = node.exact_value
                break
            action = self._select(node)
            edge = node.edges.get(action)
            if node.player == "P":
                if edge is None:
                    edge = _Edge(child=self._e_node(node, action))
                    node.edges[action] = edge
                path.append((node, edge))
                node = edge.child
                continue
            created_child = False
            if edge is None:
                step = self.transition(node.state, node.p_action, action)
                self.transitions += 1
                if not np.isfinite(step.cost):
                    raise ValueError("Transition cost must be finite.")
                edge = _Edge(cost=float(step.cost), terminal=bool(step.terminal))
                node.edges[action] = edge
                if step.terminal:
                    event = step.event or "terminal"
                    self.terminal_events[event] = self.terminal_events.get(event, 0) + 1
                else:
                    edge.child, created_child = self._p_node(step.state, node.depth + 1)
            path.append((node, edge))
            if edge.terminal:
                result = 0.0
                break
            if created_child:
                result = edge.child.initial_value
                break
            node = edge.child
        for parent, edge in reversed(path):
            result = edge.cost + result
            edge.visits += 1
            edge.total += result
            parent.visits += 1
            self._refresh_exact(parent)
        return result

    def search(self, simulations=256, max_seconds=None):
        if simulations < 0 or (max_seconds is not None and max_seconds < 0):
            raise ValueError("Budgets must be nonnegative.")
        begin = time.perf_counter()
        completed = 0
        for _ in range(simulations):
            self._refresh_exact(self.root)
            if self.root.exact_value is not None:
                break
            if max_seconds is not None and time.perf_counter() - begin >= max_seconds:
                break
            self.simulate()
            completed += 1
        if self.root.exact_value is not None and self.root.edges:
            action = min(
                self.root.actions,
                key=lambda a: self.root.edges[a].cost
                + self.root.edges[a].child.exact_value,
            )
            rule = "exact_truncated_tree"
        elif self.root.edges:
            most = max(edge.visits for edge in self.root.edges.values())
            tied = [a for a, edge in self.root.edges.items() if edge.visits == most]
            action = min(tied, key=lambda a: self.root.edges[a].mean)
            rule = "most_visited_then_lowest_mean"
        else:
            action = self.root.actions[0] if self.root.actions else None
            rule = "no_simulation_first_legal_action"
        return dict(
            action=action,
            selection_rule=rule,
            simulations=completed,
            seconds=time.perf_counter() - begin,
            nodes=len(self.nodes),
            transition_calls=self.transitions,
            leaf_queries=self.leaf_queries,
            deepest_joint_steps=self.deepest,
            finite_tree_exhausted=self.root.exact_value is not None,
            exact_truncated_value=self.root.exact_value,
            initial_leaf_estimate=self.root.initial_value,
            terminal_events=dict(self.terminal_events),
            root_actions=self.root_statistics(),
        )

    def root_statistics(self):
        rows = []
        for action in self.root.actions:
            edge = self.root.edges.get(action)
            responses = []
            if edge is not None:
                for e_action in edge.child.actions:
                    response = edge.child.edges.get(e_action)
                    responses.append(
                        dict(
                            action=int(e_action),
                            visits=0 if response is None else response.visits,
                            mean_cost=(
                                None if response is None else float(response.mean)
                            ),
                        )
                    )
            observed = [r["mean_cost"] for r in responses if r["visits"]]
            rows.append(
                dict(
                    action=int(action),
                    visits=0 if edge is None else edge.visits,
                    mean_cost=None if edge is None else float(edge.mean),
                    observed_worst_response_mean=max(observed) if observed else None,
                    responses_tested=len(observed),
                    responses=responses,
                )
            )
        return rows

    def snapshot(self, max_nodes=200):
        """Compact tree for an algorithm walkthrough; not a complete policy export."""
        return [
            dict(
                index=n.index,
                player=n.player,
                depth=n.depth,
                state=np.asarray(n.state).tolist(),
                p_action=n.p_action,
                visits=n.visits,
                initial_leaf=n.initial_value,
                exact_truncated_value=n.exact_value,
                edges=[
                    dict(
                        action=int(a),
                        visits=e.visits,
                        mean_cost=float(e.mean),
                        immediate_cost=e.cost,
                        terminal=e.terminal,
                        child=None if e.child is None else e.child.index,
                    )
                    for a, e in n.edges.items()
                ],
            )
            for n in self.nodes[:max_nodes]
        ]


@dataclass(frozen=True)
class UCTSettings:
    depth: int = 6
    simulations: int = 256
    exploration: float = 0.25
    max_seconds: float | None = None
    pursuer_fault_cost: float = 1000.0


class UCTController:
    """DP-leaf UCT prototype, not the planned B1-continuation baseline.

    Known leaf values are numerical remaining times. Unknown in-domain values
    use horizon + positive signed field / speed scale. Outside-domain or absent
    tables use horizon + straight-line closing-time guidance. These are declared
    heuristics, not capture-time bounds. Every node uses the actual map physics.
    """

    mode = "uct_dp_leaf_prototype"

    def __init__(
        self,
        config,
        world=None,
        table=None,
        settings=UCTSettings(),
        seed=0,
        p_actions=None,
        e_actions=None,
    ):
        self.config = config
        self.world = WorldMap(config.arena_half_width) if world is None else world
        self.table = table
        self.settings = settings
        self.seed = int(seed)
        self.P = (
            action_catalogue(config.pursuer)
            if p_actions is None
            else np.asarray(p_actions, float)
        )
        self.E = (
            action_catalogue(config.evader)
            if e_actions is None
            else np.asarray(e_actions, float)
        )
        for actions, limits in [(self.P, config.pursuer), (self.E, config.evader)]:
            if (
                actions.ndim != 2
                or actions.shape[1] != 2
                or not len(actions)
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    "Action catalogues must be finite nonempty arrays [n,2]."
                )
            if (abs(actions[:, 0]) > limits.acceleration).any() or (
                abs(actions[:, 1]) > limits.max_curvature
            ).any():
                raise ValueError("Catalogue exceeds a physical control bound.")
        if table is not None and hasattr(table, "assert_compatible"):
            table.assert_compatible(config)
        if table is not None and getattr(table, "state_dimension", 5) == 8:
            tc = table.config
            if (
                tc.pursuer != config.pursuer
                or tc.evader != config.evader
                or tc.capture_radius != config.capture_radius
                or self.world.half_width != tc.arena_half_width
                or self.world.obstacles
            ):
                raise ValueError(
                    "Fixed-map leaf table does not match the physical game/map."
                )
        if settings.pursuer_fault_cost <= 0 or not np.isfinite(
            settings.pursuer_fault_cost
        ):
            raise ValueError("P fault penalty must be positive and finite.")
        if settings.max_seconds is not None and (
            not np.isfinite(settings.max_seconds) or settings.max_seconds < 0
        ):
            raise ValueError("Wall-time budget must be finite and nonnegative.")
        self.mode = (
            "uct_dp_leaf_prototype"
            if table is not None
            else "uct_distance_leaf_prototype"
        )
        self.reset(None)

    def reset(self, state=None):
        self.rng = np.random.default_rng(self.seed)
        self.last = {}
        self.tree = None
        self.guard_failures = 0
        self.leaf_modes = dict(
            capture=0, DP_time=0, signed_field_heuristic=0, distance_heuristic=0
        )
        self.root_guard_failure = False
        self._root_guard_pending = True

    def _legal_p(self, state):
        safe = braking_safe_mask(state[:4], self.P, self.config, self.world)
        # A game ending during the first command does not require a braking
        # continuation afterward. This is the same exception used by B1.
        close = np.linalg.norm(state[:2] - state[4:6]) <= (
            self.config.capture_radius
            + (self.config.pursuer.max_abs_speed + self.config.evader.max_abs_speed)
            * self.config.control_dt
        )
        e_near_wall = (
            float(self.world.clearance(state[4:6]))
            <= self.config.evader.max_abs_speed * self.config.control_dt
        )
        if close or e_near_wall:
            for i in np.flatnonzero(~safe):
                outcomes = [
                    advance(state, self.P[i], e, self.config, world=self.world).event
                    for e in self.E
                ]
                safe[i] = all(
                    x in ("capture", "evader_boundary_fault") for x in outcomes
                )
        if self._root_guard_pending:
            self.root_guard_failure = not safe.any()
            self._root_guard_pending = False
        if not safe.any():
            self.guard_failures += 1
            return tuple(range(len(self.P)))
        # A deterministic neutral-first order is used only if no simulation fits.
        return tuple(
            sorted(
                np.flatnonzero(safe).tolist(),
                key=lambda i: (abs(self.P[i, 1]), abs(self.P[i, 0]), i),
            )
        )

    def _transition(self, state, p_index, e_index):
        result = advance(
            state, self.P[p_index], self.E[e_index], self.config, world=self.world
        )
        cost = result.elapsed
        if result.event == "pursuer_boundary_fault":
            cost += self.settings.pursuer_fault_cost
        return SearchStep(result.state, cost, result.terminal, result.event)

    def _leaf(self, state):
        distance = max(
            0.0,
            float(np.linalg.norm(state[:2] - state[4:6])) - self.config.capture_radius,
        )
        if distance <= 0:
            self.leaf_modes["capture"] += 1
            return 0.0
        speed_scale = max(
            self.config.pursuer.speed_max - self.config.evader.speed_max, 0.1
        )
        horizon = 0.0
        if self.table is not None:
            q = table_states(self.table, state)
            value = float(self.table.value(q))
            horizon = float(self.table.times[-1])
            if np.isfinite(value):
                self.leaf_modes["DP_time"] += 1
                return max(0.0, value)
            signed = float(self.table.field(q, horizon))
            if np.isfinite(signed):
                self.leaf_modes["signed_field_heuristic"] += 1
                return horizon + max(0.0, signed) / speed_scale
        self.leaf_modes["distance_heuristic"] += 1
        return horizon + distance / speed_scale

    def command(self, state, t=0.0, diagnostics=False):
        begin = time.perf_counter()
        state = np.asarray(state, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite eight-dimensional state.")
        self.guard_failures = 0
        self.root_guard_failure = False
        self._root_guard_pending = True
        self.leaf_modes = dict(
            capture=0, DP_time=0, signed_field_heuristic=0, distance_heuristic=0
        )
        self.tree = AdversarialUCT(
            state.copy(),
            self._legal_p,
            lambda s, p: tuple(range(len(self.E))),
            self._transition,
            self._leaf,
            depth=self.settings.depth,
            exploration=self.settings.exploration,
            seed=int(self.rng.integers(2**32)),
        )
        remaining = (
            None
            if self.settings.max_seconds is None
            else max(0.0, self.settings.max_seconds - (time.perf_counter() - begin))
        )
        result = self.tree.search(self.settings.simulations, remaining)
        if result["action"] is None:
            command = np.zeros(2)
        else:
            command = self.P[result["action"]].copy()
        for row in result["root_actions"]:
            row["command"] = self.P[row["action"]].tolist()
            for response in row["responses"]:
                response["command"] = self.E[response["action"]].tolist()
        result.update(
            requested_joint_steps=self.settings.depth,
            requested_horizon=self.settings.depth * self.config.control_dt,
            deepest_continuing_time=self.tree.deepest * self.config.control_dt,
            explicit_map_time_reached=self.tree.deepest * self.config.control_dt,
            leaf_table_horizon=(
                None if self.table is None else float(self.table.times[-1])
            ),
            leaf_modes=dict(self.leaf_modes),
            guard_failures=self.guard_failures,
            budget_policy="checked between simulations; one simulation can overrun",
            P_fault_penalty=self.settings.pursuer_fault_cost,
        )
        self.last = dict(
            control_owner=(
                "uct_dp_leaf" if self.table is not None else "uct_distance_leaf"
            ),
            search=result,
            prototype="DP-leaf UCT; no B1 rollout continuation",
            guard_failure=bool(self.root_guard_failure),
            seconds=time.perf_counter() - begin,
        )
        if diagnostics:
            self.last["tree"] = self.tree.snapshot()
            self.last["tree_snapshot_limit"] = 200
            self.last["tree_snapshot_complete"] = len(self.tree.nodes) <= 200
        return command
