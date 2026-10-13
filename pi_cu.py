"""Estimate pi with plain and stratified CuPy Monte Carlo sampling.

All point generation, circle tests, and reductions run on the active NVIDIA
CUDA device. Only scalar results used for printing are copied back to the CPU.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import cupy as cp


EXACT_PI = math.pi
QUARTER_CIRCLE_PROBABILITY = EXACT_PI / 4.0


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo result and its absolute error against the exact value."""

    samples: int
    inside_quarter_circle: int
    pi_estimate: float

    @property
    def absolute_error(self) -> float:
        return abs(self.pi_estimate - EXACT_PI)


@dataclass(frozen=True)
class EmpiricalError:
    """Observed error statistics from repeated independent estimator runs."""

    samples_per_trial: int
    trials: int
    rmse: float
    bias: float


def _sample_inside_quarter_circle(
    samples: int, rng: cp.random.Generator
) -> cp.ndarray:
    """Generate and count points on the current CUDA device."""
    points = rng.random((samples, 2))
    return cp.count_nonzero(cp.sum(points**2, axis=1) <= 1.0)


def _sample_stratified_inside_quarter_circle(
    grid_size: int, samples_per_stratum: int, rng: cp.random.Generator
) -> cp.ndarray:
    """Sample uniformly in each grid cell and count points in the circle."""
    cell_coordinates = cp.stack(
        cp.meshgrid(
            cp.arange(grid_size), cp.arange(grid_size), indexing="ij"
        ),
        axis=-1,
    )
    offsets = rng.random((grid_size, grid_size, samples_per_stratum, 2))
    points = (cell_coordinates[:, :, None, :] + offsets) / grid_size
    return cp.count_nonzero(cp.sum(points**2, axis=-1) <= 1.0)


def estimate_pi(samples: int, *, seed: int | None = None) -> Estimate:
    """Return a CuPy Monte Carlo estimate of pi using ``samples`` points."""
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    rng = cp.random.default_rng(0 if seed is None else seed)
    inside = int(_sample_inside_quarter_circle(samples, rng).get())
    return Estimate(samples, inside, 4.0 * inside / samples)


def estimate_pi_stratified(
    grid_size: int, samples_per_stratum: int = 1, *, seed: int | None = None
) -> Estimate:
    """Estimate pi with uniform samples in every cell of a square grid."""
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be a positive integer")

    samples = grid_size**2 * samples_per_stratum
    rng = cp.random.default_rng(0 if seed is None else seed)
    inside = int(
        _sample_stratified_inside_quarter_circle(
            grid_size, samples_per_stratum, rng
        ).get()
    )
    return Estimate(samples, inside, 4.0 * inside / samples)


def pi_standard_error(samples: int) -> float:
    """Return the theoretical standard error of the plain estimator."""
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    p = QUARTER_CIRCLE_PROBABILITY
    return 4.0 * math.sqrt(p * (1.0 - p) / samples)


def empirical_error(
    samples: int, trials: int, *, seed: int | None = None
) -> EmpiricalError:
    """Measure RMSE and bias using a single parallel GPU batch."""
    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    rng = cp.random.default_rng(0 if seed is None else seed)
    points = rng.random((trials, samples, 2))
    inside_counts = cp.count_nonzero(
        cp.sum(points**2, axis=-1) <= 1.0, axis=1
    )
    estimates = 4.0 * inside_counts / samples
    errors = estimates - EXACT_PI
    return EmpiricalError(
        samples_per_trial=samples,
        trials=trials,
        rmse=float(cp.sqrt(cp.mean(errors**2)).get()),
        bias=float(cp.mean(errors).get()),
    )


def stratified_empirical_error(
    grid_size: int, samples_per_stratum: int, trials: int, *, seed: int | None = None
) -> EmpiricalError:
    """Measure stratified RMSE and bias using a single parallel GPU batch."""
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    samples = grid_size**2 * samples_per_stratum
    rng = cp.random.default_rng(0 if seed is None else seed)
    cell_coordinates = cp.stack(
        cp.meshgrid(
            cp.arange(grid_size), cp.arange(grid_size), indexing="ij"
        ),
        axis=-1,
    )
    offsets = rng.random(
        (trials, grid_size, grid_size, samples_per_stratum, 2)
    )
    points = (cell_coordinates[None, :, :, None, :] + offsets) / grid_size
    inside_counts = cp.count_nonzero(
        cp.sum(points**2, axis=-1).reshape(trials, samples) <= 1.0, axis=1
    )
    estimates = 4.0 * inside_counts / samples
    errors = estimates - EXACT_PI
    return EmpiricalError(
        samples_per_trial=samples,
        trials=trials,
        rmse=float(cp.sqrt(cp.mean(errors**2)).get()),
        bias=float(cp.mean(errors).get()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CuPy Monte Carlo estimate of pi")
    parser.add_argument(
        "samples",
        nargs="?",
        type=int,
        help="plain-Monte-Carlo sample count (default: 1000000)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        help="use an m by m equal-area grid, with m given by this value",
    )
    parser.add_argument(
        "--samples-per-stratum",
        type=int,
        default=1,
        help="independent samples in each grid cell (default: 1)",
    )
    parser.add_argument("--seed", type=int, help="optional reproducible random seed")
    parser.add_argument(
        "--trials",
        type=int,
        help="run independent trials and report empirical RMSE and bias",
    )
    args = parser.parse_args()

    if args.grid_size is None:
        if args.samples_per_stratum != 1:
            parser.error("--samples-per-stratum requires --grid-size")
        samples = 1_000_000 if args.samples is None else args.samples
        result = estimate_pi(samples, seed=args.seed)
        empirical = (
            empirical_error(result.samples, args.trials, seed=args.seed)
            if args.trials is not None
            else None
        )
        standard_error = pi_standard_error(result.samples)
    else:
        if args.samples is not None:
            parser.error("samples cannot be used with --grid-size")
        result = estimate_pi_stratified(
            args.grid_size, args.samples_per_stratum, seed=args.seed
        )
        empirical = (
            stratified_empirical_error(
                args.grid_size, args.samples_per_stratum, args.trials, seed=args.seed
            )
            if args.trials is not None
            else None
        )
        standard_error = None

    print(f"samples:  {result.samples:,}")
    print(f"inside:   {result.inside_quarter_circle:,}")
    print(f"estimate: {result.pi_estimate:.10f}")
    print(f"exact:    {EXACT_PI:.10f}")
    print(f"abs error:{result.absolute_error:.10f}")
    if standard_error is not None:
        print(f"std error:{standard_error:.10f}")
    else:
        print(f"grid:     {args.grid_size} x {args.grid_size}")
        print(f"per cell: {args.samples_per_stratum}")
    if empirical is not None:
        print(f"trials:   {empirical.trials:,}")
        print(f"emp RMSE: {empirical.rmse:.10f}")
        print(f"emp bias: {empirical.bias:.10f}")


if __name__ == "__main__":
    main()
