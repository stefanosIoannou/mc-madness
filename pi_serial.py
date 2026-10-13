"""Estimate pi with plain and stratified JAX Monte Carlo samplers.

Points are sampled uniformly from the unit square.  The fraction lying in the
quarter unit circle estimates its area (pi / 4), so four times that fraction
estimates pi.  The standard Monte Carlo error is O(1 / sqrt(samples)).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax import random


EXACT_PI = float(jnp.pi)
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


@partial(jax.jit, static_argnames=("samples",))
def _sample_inside_quarter_circle(samples: int, key: jax.Array) -> jax.Array:
    """Generate and count points on the JAX backend."""
    points = random.uniform(key, shape=(samples, 2))
    return jnp.count_nonzero(jnp.sum(points**2, axis=1) <= 1.0)


@partial(jax.jit, static_argnames=("grid_size", "samples_per_stratum"))
def _sample_stratified_inside_quarter_circle(
    grid_size: int, samples_per_stratum: int, key: jax.Array
) -> jax.Array:
    """Sample uniformly inside each grid cell and count points in the circle."""
    cell_coordinates = jnp.stack(
        jnp.meshgrid(
            jnp.arange(grid_size), jnp.arange(grid_size), indexing="ij"
        ),
        axis=-1,
    )
    offsets = random.uniform(
        key, shape=(grid_size, grid_size, samples_per_stratum, 2)
    )
    points = (cell_coordinates[:, :, None, :] + offsets) / grid_size
    return jnp.count_nonzero(jnp.sum(points**2, axis=-1) <= 1.0)


def estimate_pi(samples: int, *, seed: int | None = None) -> Estimate:
    """Return a JAX Monte Carlo estimate of pi using ``samples`` points.

    JAX generates the samples and counts them in compiled array operations.
    Supplying ``seed`` makes a run reproducible; omitting it uses seed zero.
    """
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    key = random.PRNGKey(0 if seed is None else seed)
    inside = int(_sample_inside_quarter_circle(samples, key))

    return Estimate(samples, inside, 4.0 * inside / samples)


def estimate_pi_stratified(
    grid_size: int, samples_per_stratum: int = 1, *, seed: int | None = None
) -> Estimate:
    """Estimate pi with uniform samples in every cell of a square grid.

    The unit square is split into ``grid_size`` by ``grid_size`` equally sized
    strata.  Each stratum receives ``samples_per_stratum`` independent uniform
    samples, so the total is ``grid_size ** 2 * samples_per_stratum``.  Since
    every sample is uniform in its own stratum and all strata have equal area,
    this is an unbiased estimate of pi.
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be a positive integer")

    samples = grid_size**2 * samples_per_stratum
    key = random.PRNGKey(0 if seed is None else seed)
    inside = int(
        _sample_stratified_inside_quarter_circle(grid_size, samples_per_stratum, key)
    )
    return Estimate(samples, inside, 4.0 * inside / samples)


def pi_standard_error(samples: int) -> float:
    """Return the theoretical standard error of the Monte Carlo pi estimate.

    A point lands in the quarter circle with probability p = pi / 4.  Since
    the estimator is four times a binomial proportion, its standard error is
    ``4 * sqrt(p * (1 - p) / samples)`` (about ``1.64 / sqrt(samples)``).
    """
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    p = QUARTER_CIRCLE_PROBABILITY
    return float(4.0 * jnp.sqrt(p * (1.0 - p) / samples))


def empirical_error(
    samples: int, trials: int, *, seed: int | None = None
) -> EmpiricalError:
    """Measure the estimator's RMSE and bias over independent trials.

    Compare the returned ``rmse`` with that of another estimator only when it
    uses the same number of samples per trial and the same number of trials.
    """
    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    key = random.PRNGKey(0 if seed is None else seed)
    keys = random.split(key, trials)
    inside_counts = jax.vmap(lambda trial_key: _sample_inside_quarter_circle(samples, trial_key))(
        keys
    )
    estimates = 4.0 * inside_counts / samples
    errors = estimates - EXACT_PI
    return EmpiricalError(
        samples_per_trial=samples,
        trials=trials,
        rmse=float(jnp.sqrt(jnp.mean(errors**2))),
        bias=float(jnp.mean(errors)),
    )


def stratified_empirical_error(
    grid_size: int, samples_per_stratum: int, trials: int, *, seed: int | None = None
) -> EmpiricalError:
    """Measure RMSE and bias for repeated stratified-estimator runs."""
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    samples = grid_size**2 * samples_per_stratum
    key = random.PRNGKey(0 if seed is None else seed)
    keys = random.split(key, trials)
    inside_counts = jax.vmap(
        lambda trial_key: _sample_stratified_inside_quarter_circle(
            grid_size, samples_per_stratum, trial_key
        )
    )(keys)
    estimates = 4.0 * inside_counts / samples
    errors = estimates - EXACT_PI
    return EmpiricalError(
        samples_per_trial=samples,
        trials=trials,
        rmse=float(jnp.sqrt(jnp.mean(errors**2))),
        bias=float(jnp.mean(errors)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX Monte Carlo estimate of pi")
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
