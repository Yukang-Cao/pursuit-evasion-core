# Code and numerical provenance

The shared model was extracted and simplified from the local pursuit-evasion
research implementation, with an explicit world-state and unit contract. The
new package deliberately does not import private report builders or hardcoded
paths to earlier experiment directories.

External solvers remain external dependencies, pinned in experiment records:

- OptimizedDP: <https://github.com/SFU-MARS/optimized_dp>
- hj_reachability: <https://github.com/StanfordASL/hj_reachability>

The planned MPPI implementation follows the weighted-sampling structure used
in the C-Free-Uniform navigation code. Dynamics, costs and units are checked
again for this game. Do not treat matching method names as matching models.

License for newly authored project code has not yet been selected for public
release. External dependency licenses remain with their respective authors.

