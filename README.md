# Monte Carlo π estimator

This project estimates π by sampling random points in the unit square and
counting those inside the quarter unit circle. The estimate is
`4 * inside / samples`.

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

## Functionality

- Generates Monte Carlo samples with JAX's random-number generator.
- JIT-compiles the array-based sampling and circle-counting operation.
- Reports the number of samples, points inside the circle, the π estimate, and
  its absolute error from the exact value.
- Calculates the theoretical standard error:
  `4 * sqrt((π / 4) * (1 - π / 4) / N)`, approximately `1.64 / sqrt(N)`.

More samples generally improve the estimate, but the statistical error falls
only in proportion to `1 / sqrt(N)`.
