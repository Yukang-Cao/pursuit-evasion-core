# Read the code in this order

1. `model.py`: state order, limits, exact arcs and the continuous Hamiltonian.
2. `game.py`: first-event stopping and capture versus boundary-fault outcomes.
3. `configs/square6d.json`: the experiment's physical parameters.
4. `tests/test_model.py`: independent checks of the shared physical contract.

The numerical backends and controller modules are added separately. They must
consume this contract rather than redefine the vehicle model internally.

## Intended ownership

- `dp/`: offline numerical adapters, portable field metadata, and online queries.
- `controllers/`: geometric targets, MPPI, and the DP/approach handoff.
- `search/`: later UCT integration; no dependency from the base game to search.
- `learning/`: later cloning/PPO; labels include targets and actual commands.
- `scripts/`: thin reproducible experiment entrypoints, not another physics copy.

Only implemented modules should appear in the working examples. Generated
fields and replays belong in an explicitly supplied output directory outside Git.
Offline backends may have separate environments; online import should remain
lightweight and portable to Jetson Orin.

