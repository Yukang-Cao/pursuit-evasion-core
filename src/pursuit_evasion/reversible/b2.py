"""B2: adversarial UCT outside DP coverage, with B1 feedback continuations.

Each new tree frontier simulates a bounded B1 rollout against the declared
reactive E policy. If B1 takes over with DP, its one-step DP cost closes the
rollout. Otherwise a fixed terminal heuristic closes the bounded simulation.
This is an approximate search evaluation, not an exact worst-response value.
"""

from dataclasses import dataclass
import copy
import time
import numpy as np
from .controllers import Baseline1, action_catalogue
from .mcts import AdversarialUCT, UCTController, UCTSettings
from .mppi import MPPISettings
from .opponents import ReactiveEvader
from .game import WorldMap, advance


@dataclass(frozen=True)
class B2Settings:
    depth: int = 6
    simulations: int = 128
    rollout_steps: int = 3
    exploration: float = 0.5
    max_seconds: float | None = None


class B2Controller:
    """Same DP switch as B1; UCT chooses primitive controls in approach mode.

    A B1 command is included in the root catalogue alongside the nine bounded
    acceleration/curvature commands. Each hypothetical B1 controller gets an
    isolated copy of the current approach planner's warm start, RNG and anchor.
    The warm-start snapshot is a declared simulation initialization; it is not
    silently advanced by running additional optimizers along the searched prefix.
    """

    mode = "b2_uct_b1"

    def __init__(
        self,
        config,
        world=None,
        table=None,
        settings=B2Settings(),
        seed=0,
        mppi_settings=None,
        lookahead=1.0,
    ):
        if settings.depth < 1 or settings.simulations < 1 or settings.rollout_steps < 0:
            raise ValueError("Invalid B2 search/rollout budget")
        self.config = config
        self.world = WorldMap(None) if world is None else world
        self.table = table
        self.settings = settings
        self.seed = int(seed)
        self.baseline = Baseline1(
            config,
            self.world,
            table,
            mode="hybrid",
            seed=seed,
            mppi_settings=mppi_settings,
            lookahead=lookahead,
            takeover="successor",
        )
        self.adapter = UCTController(
            config,
            self.world,
            table,
            UCTSettings(
                depth=settings.depth,
                simulations=settings.simulations,
                exploration=settings.exploration,
                max_seconds=settings.max_seconds,
            ),
            seed=seed,
        )
        self.last = {}
        self.tree = None
        self.ready = False

    def reset(self, state):
        self.baseline.reset(state)
        self.adapter.reset(state)
        self.rng = np.random.default_rng(self.seed)
        self.last = {}
        self.tree = None
        self.ready = True

    def fork_b1(self, state):
        """Share only immutable model/map/table; isolate every mutable policy field."""
        base = self.baseline
        fork = Baseline1(
            self.config,
            self.world,
            self.table,
            mode="hybrid",
            seed=self.seed,
            mppi_settings=base.tracker.settings,
            lookahead=base.lion.step_length,
            takeover="successor",
        )
        fork.reset(state)
        fork.lion.anchor = base.lion.anchor.copy()
        fork.tracker.nominal = base.tracker.nominal.copy()
        fork.tracker.rng.bit_generator.state = copy.deepcopy(
            base.tracker.rng.bit_generator.state
        )
        return fork

    def _continuation(self, state):
        start = time.perf_counter()
        self.rollout_calls += 1
        s = np.asarray(state, float).copy()
        key = s.tobytes()
        self.dp_frontiers[key] = False
        elapsed = 0.0
        if np.linalg.norm(s[:2] - s[4:6]) <= self.config.capture_radius:
            self.rollout_endings["capture"] += 1
            self.dp_frontiers[key] = True
            return 0.0
        policy = self.fork_b1(s)
        evader = ReactiveEvader(self.config, self.world, mode="separation", horizon=1.0)
        evader.reset()
        for k in range(self.settings.rollout_steps):
            u = policy.command(s, k * self.config.control_dt, diagnostics=False)
            self.rollout_b1_calls += 1
            if policy.last["control_owner"].startswith("dp"):
                if k == 0:
                    self.dp_frontiers[key] = True
                self.rollout_endings["DP_takeover"] += 1
                self.rollout_compute_seconds += time.perf_counter() - start
                return elapsed + float(policy.last["dp"]["estimated_cost"])
            v = evader.command(s, u)
            result = advance(s, u, v, self.config, world=self.world)
            elapsed += result.elapsed
            self.rollout_motion_steps += 1
            s = result.state
            if result.terminal:
                self.rollout_endings[result.event] = (
                    self.rollout_endings.get(result.event, 0) + 1
                )
                self.rollout_compute_seconds += time.perf_counter() - start
                return elapsed + (
                    1000.0 if result.event == "pursuer_boundary_fault" else 0.0
                )
        self.rollout_endings["truncated"] += 1
        value = elapsed + self.adapter._leaf(s)
        self.rollout_compute_seconds += time.perf_counter() - start
        return value

    def _tree_p_actions(self, state):
        # The real hierarchical controller commits to DP at these states.
        # Do not search alternative future P choices after that takeover.
        if self.dp_frontiers.get(np.asarray(state, float).tobytes(), False):
            return ()
        return self.adapter._legal_p(state)

    def command(self, state, t=0.0, diagnostics=False):
        begin = time.perf_counter()
        state = np.asarray(state, float)
        if not self.ready:
            self.reset(state)
        proposal = self.baseline.command(state, t, diagnostics=False)
        if self.baseline.last["control_owner"].startswith("dp"):
            self.last = dict(self.baseline.last)
            self.last["search"] = {"skipped_for_DP": True, "simulations": 0}
            self.last["seconds"] = time.perf_counter() - begin
            return proposal
        commands = [proposal.copy()]
        for u in action_catalogue(self.config.pursuer):
            if not any(np.array_equal(u, v) for v in commands):
                commands.append(u.copy())
        self.adapter.P = np.asarray(commands)
        self.adapter.guard_failures = 0
        self.adapter.root_guard_failure = False
        self.adapter._root_guard_pending = True
        self.adapter.leaf_modes = dict(
            capture=0, DP_time=0, signed_field_heuristic=0, distance_heuristic=0
        )
        self.dp_frontiers = {}
        self.rollout_calls = 0
        self.rollout_b1_calls = 0
        self.rollout_motion_steps = 0
        self.rollout_compute_seconds = 0.0
        self.rollout_endings = {"capture": 0, "DP_takeover": 0, "truncated": 0}
        self.tree = AdversarialUCT(
            state.copy(),
            self._tree_p_actions,
            lambda s, p: tuple(range(len(self.adapter.E))),
            self.adapter._transition,
            self._continuation,
            depth=self.settings.depth,
            exploration=self.settings.exploration,
            seed=int(self.rng.integers(2**32)),
        )
        available = (
            None
            if self.settings.max_seconds is None
            else max(0.0, self.settings.max_seconds - (time.perf_counter() - begin))
        )
        seeded = 0
        pairs = [
            (p, e) for p in self.tree.root.actions for e in range(len(self.adapter.E))
        ]
        # In this small finite-action game, first inspect every root E response
        # when the declared total budget permits it. Extra visits then use UCT.
        if self.settings.simulations >= len(pairs):
            for pair in pairs:
                if (
                    self.settings.max_seconds is not None
                    and time.perf_counter() - begin >= self.settings.max_seconds
                ):
                    break
                if self.tree.root.exact_value is not None:
                    break
                self.tree.simulate(root_pair=pair)
                seeded += 1
        remaining = (
            None
            if self.settings.max_seconds is None
            else max(0.0, self.settings.max_seconds - (time.perf_counter() - begin))
        )
        search = self.tree.search(self.settings.simulations - seeded, remaining)
        search["simulations"] += seeded
        search["root_initialization_simulations"] = seeded
        search["seconds"] = time.perf_counter() - begin
        if not search["finite_tree_exhausted"]:
            covered = [
                r
                for r in search["root_actions"]
                if r["responses_tested"] == len(self.adapter.E)
            ]
            if covered:
                search["action"] = min(
                    covered,
                    key=lambda r: (r["observed_worst_response_mean"], r["action"]),
                )["action"]
                search["selection_rule"] = "minimum_worst_mean_over_fully_tested_E_rows"
            else:
                search["action"] = 0
                search["selection_rule"] = "B1_proposal_until_a_full_E_row_is_evaluated"
        index = search["action"]
        command = proposal if index is None else self.adapter.P[index].copy()
        search.update(
            rollout_calls=self.rollout_calls,
            rollout_b1_calls=self.rollout_b1_calls,
            rollout_motion_steps=self.rollout_motion_steps,
            rollout_compute_seconds=self.rollout_compute_seconds,
            rollout_max_time=self.settings.rollout_steps * self.config.control_dt,
            rollout_endings=self.rollout_endings,
            tail_heuristic_counts=self.adapter.leaf_modes,
            requested_simulations=self.settings.simulations,
            root_B1_action=proposal.tolist(),
            root_selected_action=command.tolist(),
            root_selected_B1_proposal=bool(index == 0),
        )
        for row in search["root_actions"]:
            row["command"] = self.adapter.P[row["action"]].tolist()
        self.last = {
            "control_owner": "uct_b1",
            "takeover_rule": "successor",
            "dp_value": self.baseline.last.get("dp_value"),
            "search": search,
            "seconds": time.perf_counter() - begin,
            "guard_failure": bool(self.adapter.root_guard_failure),
        }
        if diagnostics:
            self.last["tree"] = self.tree.snapshot()
            self.last["B1_proposal"] = copy.deepcopy(self.baseline.last)
        return command
