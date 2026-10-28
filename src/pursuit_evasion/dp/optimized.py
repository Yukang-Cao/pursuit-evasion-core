"""Subprocess boundary for the legacy HeteroCL environment."""

import json
import os
from pathlib import Path
import subprocess
from .options import validate_options


def solve(
    config,
    grid,
    output,
    *,
    horizon=2.0,
    output_dt=0.1,
    order=2,
    cfl=0.8,
    threads=8,
    history=True,
):
    validate_options(horizon, output_dt, order, cfl, threads)
    if abs(cfl - 0.8) > 1e-12:
        raise ValueError("The unmodified OptimizedDP graph uses CFL 0.8.")
    python = os.environ.get("PE_ODP_PYTHON")
    if not python:
        raise RuntimeError(
            "Set PE_ODP_PYTHON to the isolated Python executable with OptimizedDP/HeteroCL."
        )
    output = Path(output).resolve()
    if (output / "meta.json").exists():
        raise FileExistsError(output)
    output.mkdir(exist_ok=True, parents=True)
    config_path = output / "physical_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    options = dict(
        config=config_path,
        output=output,
        positions=grid.positions,
        headings=grid.headings,
        extent=grid.extent,
        horizon=horizon,
        order=order,
        threads=threads,
        history=int(history),
        **{"output-dt": output_dt},
    )
    args = [python, str(Path(__file__).with_name("optimized_runner.py"))] + [
        str(x) for k, v in options.items() for x in ("--" + k, v)
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    if env.get("PE_ODP_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = (
            env["PE_ODP_LIBRARY_PATH"] + ":" + env.get("LD_LIBRARY_PATH", "")
        )
    with (output / "runner.log").open("w") as log:
        result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'OptimizedDP runner failed; inspect {output/"runner.log"}')
    return json.loads((output / "meta.json").read_text())
