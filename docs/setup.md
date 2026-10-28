# Install and run a small CPU check

The online model, controllers and table reader need only NumPy. Start with the
base CPU environment; install an offline numerical backend separately when
needed. Commands below assume a checkout of this repository and a Bash shell.

## Base package

Python 3.10 or newer is declared by the package. The fresh-install check used
Python 3.12.3, NumPy 2.5.4 and pytest 9.2.0 on Ubuntu 24.04 x86-64. These pins
describe that check; other supported Python versions may need older dependency
versions.

From the repository root:

```bash
export PE_CORE_DIR="$PWD"
python3.12 -m venv .venv
.venv/bin/python -m pip install 'numpy==2.5.4' 'pytest==9.2.0' .
.venv/bin/python -m pytest -q
```

For editable development use `python -m pip install -e '.[test]'` in your
chosen environment instead. The standard test suite needs no JAX or HeteroCL.
Its native test skips when `g++` is unavailable; a compiler present without
OpenMP support is not sufficient. The tested compiler was GCC/g++ 13.3.0 with
OpenMP. On Ubuntu, the usual `build-essential` package supplies the compiler.

Run a small native solve in a new directory outside the checkout:

```bash
export PE_RUN_DIR="$(mktemp -d /tmp/pe-check.XXXXXX)"
"$PE_CORE_DIR/.venv/bin/python" -m pursuit_evasion solve \
  --config "$PE_CORE_DIR/configs/square6d.json" \
  --backend native --positions 9 --headings 8 --horizon 0.2 \
  --threads 2 --output "$PE_RUN_DIR/native"
```

This grid has 419,904 states and is an installation check, not a useful accuracy
claim. A completed solve writes `meta.json`, `final.f32`, `arrival.f32` and,
unless `--no-history` is used, `fields.f32`. Only a completed solve retaining
signed history can be loaded as a `LocalTable` for DP finishing. Do not use
`arrival.f32` as an interpolated capture-time table.

The native binary is compiled on first use and cached under
`~/.cache/pursuit-evasion-core`; `PE_BUILD_CACHE` can choose another directory.
Use a new cache directory when comparing cold compilation or changing compiler
toolchains. Output directories with existing metadata are preserved, including
failed runs; choose a fresh output name to retry.

Run the small `front` development case without needing a DP table:

```bash
"$PE_CORE_DIR/.venv/bin/python" -m pursuit_evasion chase \
  --config "$PE_CORE_DIR/configs/square6d.json" \
  --cases "$PE_CORE_DIR/configs/development_cases.json" --case front \
  --without-dp --seed 0 --max-time 1 --diagnostic-stride 0 \
  --output "$PE_RUN_DIR/front.json"
```

The release check exercised a noneditable installation, the tests, a native
solve with a cold build cache, and this chase from another working directory.
It also verified that importing the controller does not import JAX or HeteroCL.

## Optional JAX/hj_reachability environment

Use another Python 3.12 environment. The following versions were used by the
recorded numerical runs; they are not a complete transitive dependency lock.

```bash
python3.12 -m venv .venv-jax
.venv-jax/bin/python -m pip install \
  'numpy==2.5.4' 'jax==0.12.0.1' 'jaxlib==0.12.0.1' \
  'hj-reachability==0.7.0' 'psutil==7.2.2' .
JAX_PLATFORMS=cpu .venv-jax/bin/python -m pursuit_evasion solve \
  --config configs/square6d.json --backend jax \
  --positions 9 --headings 8 --horizon 0.2 \
  --output "$PE_RUN_DIR/jax-cpu"
```

The [upstream installation instructions](https://github.com/StanfordASL/hj_reachability#installation)
explain installing an accelerator-enabled JAX first. GPU installation depends
on the actual device and driver; it is not required for this CPU example.
Record the selected device in every comparison. `--threads` does not set JAX's
CPU thread pool and must not be interpreted as matching OpenMP thread counts.

## Optional isolated OptimizedDP environment

OptimizedDP's tested HeteroCL stack uses Python 3.8, while the main package
requires Python 3.10+. Keep them separate. The adapter launches a standalone
runner with `PE_ODP_PYTHON`; do not install the full core package into Python 3.8.

The recorded Linux setup used the following sequence with Micromamba already
installed. Choose a new dependency folder outside the checkout:

```bash
export PE_DEPS_DIR="$(mktemp -d /tmp/pe-deps.XXXXXX)"
micromamba create -y -p "$PE_DEPS_DIR/odp-env" -c conda-forge \
  python=3.8.19 pip numpy=1.24.4 scipy=1.10.1 pytest psutil
micromamba install -y -p "$PE_DEPS_DIR/odp-env" \
  -c cornell-zhang -c conda-forge -c defaults heterocl=0.3 plotly
git clone https://github.com/SFU-MARS/optimized_dp.git "$PE_DEPS_DIR/optimized_dp"
git -C "$PE_DEPS_DIR/optimized_dp" checkout 341b20c49813ea8ec1f84638bae60a70dd4d6e7b
"$PE_DEPS_DIR/odp-env/bin/python" -m pip install \
  future==1.0.0 cloudpickle==3.1.2 decorator==4.4.2 attrs==25.3.0 pandas==2.0.3
"$PE_DEPS_DIR/odp-env/bin/python" -m pip install -e "$PE_DEPS_DIR/optimized_dp"
export PE_ODP_PYTHON="$PE_DEPS_DIR/odp-env/bin/python"
```

This legacy environment is platform-sensitive, not a portable lock. The
tested Conda HeteroCL artifact was `0.3-py38ha8d69ae_0` from `cornell-zhang`,
with LLVM 6.0.0. Its Python distribution reports `heterocl==0.1`; use the Conda
record as well as `pip freeze` to identify the actual build. A plain pip install
of `heterocl==0.3` did not provide this environment.

If import fails on `libtinfo.so.5`, follow the compatibility note in the
[upstream README](https://github.com/SFU-MARS/optimized_dp#quickstart-ubuntu-18-20-22).
The development setup extracted an official Ubuntu `libtinfo5` package into a
local compatibility directory. For that approach, set `PE_ODP_LIBRARY_PATH` to
the extracted library directory followed by the environment's `lib` directory.
The wrapper adds it only to the child process's library search path. No
compatibility binaries are included here.

After the isolated interpreter can import `heterocl` and `odp.solver`, run from
the base environment:

```bash
.venv/bin/python -m pursuit_evasion solve --config configs/square6d.json \
  --backend optimized_dp --positions 9 --headings 8 --horizon 0.2 \
  --threads 2 --output "$PE_RUN_DIR/optimized-dp"
```

This upstream graph fixes CFL at 0.8. Retaining its full history may allocate
float64 storage internally before conversion to float32; budget memory for the
backend, not only the final files.

## Before increasing the grid

State count is `positions⁴ × headings²`. One float32 field uses four bytes per
state, and retained history multiplies that by the number of output frames.
Solvers also need working arrays. For example, 29 positions and 24 headings
already give 407,393,856 states; 21 frames occupy about 34.2 GB before other
outputs. Check free host memory, device memory and disk first.

Save the core revision, configuration, command, package versions and compiler
version with each experiment. Preserve failed-run metadata. The current tests
and small examples establish an installation path, not convergence or a
Jetson 10 Hz deployment result.
