# Monte Carlo π estimator

This project estimates π by sampling random points in the unit square and
counting those inside the quarter unit circle. The estimate is
`4 * inside / samples`.

## Run on an AMD GPU with HIP

`pi_hip.py` uses AMD's low-level `hip-python` bindings, HIPRTC for compiling
the GPU kernels, and hipRAND for random-number generation. ROCm must be
installed on a supported Linux system with a compatible AMD GPU before
installing the Python package. ROCm is not supported on macOS, so the current
Mac environment cannot run this backend directly.

After installing ROCm, create a HIP environment and install the matching
`hip-python` release. The package version should match the first three numbers
of the installed ROCm version, for example `7.0.0` for ROCm 7.0.x:

```bash
uv venv .venv-hip
source .venv-hip/bin/activate
python -m pip install --upgrade pip
uv sync --extra hip
```

If the automatic version is not compatible with the installed ROCm release,
install the matching version explicitly. For ROCm `7.0.x`, for example:

```bash
python -m pip install "hip-python~=7.0.0.0"
```

Verify the GPU is visible before running:

```bash
python -c "from hip import hip; print(hip.hipGetDeviceCount())"
python pi_hip.py 1000000 --seed 42
python pi_hip.py --grid-size 1000 --seed 42
```

## Run

Create a virtual environment and install the project dependencies with `uv`:

```powershell
uv venv
uv sync
```

This creates `.venv` in the project directory. Then run the program:

```powershell
uv run python pi_serial.py 1000000 --seed 42
```

`samples` is optional and defaults to `1_000_000`. `--seed` is optional; use
it when you want the same random result on every run.

To measure empirical accuracy over repeated, independent runs, use `--trials`:

```powershell
uv run python pi_serial.py 100000 --trials 100 --seed 42
```

The reported empirical RMSE is the root mean square of each trial's error from
π. Use the same sample count and number of trials when comparing this value
with another estimator: a lower RMSE is statistically more efficient.

## Stratified sampling

### Method summary

- Plain Monte Carlo samples points anywhere in the unit square, so points can
  cluster by chance.
- Equal-grid stratification divides the square into an `m x m` grid and takes
  one uniform random point from every cell (`m^2` total points).
- Multiple-sample stratification takes `k` uniform random points from every
  cell (`m^2 * k` total points).

Both stratified methods remain unbiased while usually reducing variance by
ensuring that every part of the square is represented.

Use `--grid-size m` to split the unit square into an `m x m` grid and draw one
uniform point inside every cell. This produces exactly `m^2` points and is an
unbiased estimator that usually has lower variance than plain Monte Carlo:

```powershell
uv run python pi_serial.py --grid-size 1000 --seed 42
```

Use `--samples-per-stratum k` to draw `k` independent points in every cell.
The total sample count is then `m^2 * k`, which allows useful counts beyond a
perfect square:

```powershell
uv run python pi_serial.py --grid-size 300 --samples-per-stratum 3 --seed 42
```

The usual plain-Monte-Carlo standard-error formula is not printed for a
stratified run because it does not account for the reduced, stratum-specific
variance. Use `--trials` to compare empirical RMSE at matched total samples.

## Run with MPI

Each MPI rank receives its own JAX random-number stream by folding the rank
into the base seed. The local inside-circle counts and sample counts are
combined with an MPI `Allreduce` sum; rank 0 prints the global result.

```bash
uv sync
uv run mpiexec -n 4 python pi_mpi.py 1000000 --seed 42
```

The `mpiexec` command must be provided by an installed MPI implementation
such as Open MPI or MPICH. On macOS, if `mpiexec` is not already installed,
install one with Homebrew before running the command.

## Functionality

- Generates Monte Carlo samples with JAX's random-number generator.
- JIT-compiles the array-based sampling and circle-counting operation.
- Reports the number of samples, points inside the circle, the π estimate, and
  its absolute error from the exact value.
- Calculates the theoretical standard error:
  `4 * sqrt((π / 4) * (1 - π / 4) / N)`, approximately `1.64 / sqrt(N)`.
- Calculates empirical RMSE and bias across optional independent trials.
- Supports equal-grid stratified sampling, with one or multiple samples per
  stratum.

More samples generally improve the estimate, but the statistical error falls
only in proportion to `1 / sqrt(N)`.
