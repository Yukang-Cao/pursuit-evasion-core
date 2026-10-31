from types import SimpleNamespace
import numpy as np
from pursuit_evasion.reversible.map_game_policy import select_time_or_margin


class Values:
    def __init__(self, times, fields):
        self.times = np.asarray(times)
        self.fields = np.asarray(fields)

    def time_value(self, states):
        return self.times

    def field(self, states):
        return self.fields


def test_one_unresolved_evader_response_disqualifies_an_action_row():
    commands = np.array([[3, 0], [0, 0]])
    records = [[SimpleNamespace(elapsed=0.05) for _ in range(2)] for _ in range(2)]
    codes = np.full((2, 2), "continuing", dtype="U20")
    value = Values([[0.1, np.nan], [0.8, 0.9]], [[-1, 1], [-0.1, -0.1]])
    index, mode = select_time_or_margin(commands, records, None, codes, value)
    assert (index, mode) == (1, "time")
    # Capture ends that pair, so it must not require a continuation lookup.
    codes[0, 1] = "capture"
    index, mode = select_time_or_margin(commands, records, None, codes, value)
    assert (index, mode) == (0, "time")


def test_margin_fallback_keeps_minimax_and_collision_priority():
    commands = np.array([[3, 0], [0, 0]])
    records = [[SimpleNamespace(elapsed=0.05) for _ in range(2)] for _ in range(2)]
    codes = np.full((2, 2), "continuing", dtype="U20")
    value = Values(np.full((2, 2), np.nan), [[-1, 0.8], [0.1, 0.2]])
    assert select_time_or_margin(commands, records, None, codes, value) == (
        1,
        "margin_fallback",
    )
    codes[1, 0] = "pursuer_collision"
    assert select_time_or_margin(commands, records, None, codes, value) == (
        0,
        "margin_fallback",
    )
