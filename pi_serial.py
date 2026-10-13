"""Estimate pi with a JAX Monte Carlo sampler.

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


@partial(jax.jit, static_argnames=("samples",))
def _sample_inside_quarter_circle(samples: int, key: jax.Array) -> jax.Array:
    """Generate and count points on the JAX backend."""
    points = random.uniform(key, shape=(samples, 2))
    return jnp.count_nonzero(jnp.sum(points**2, axis=1) <= 1.0)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX Monte Carlo estimate of pi")
    parser.add_argument("samples", nargs="?", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, help="optional reproducible random seed")
    args = parser.parse_args()

    result = estimate_pi(args.samples, seed=args.seed)
    print(f"samples:  {result.samples:,}")
    print(f"inside:   {result.inside_quarter_circle:,}")
    print(f"estimate: {result.pi_estimate:.10f}")
    print(f"exact:    {EXACT_PI:.10f}")
    print(f"abs error:{result.absolute_error:.10f}")
    print(f"std error:{pi_standard_error(result.samples):.10f}")


if __name__ == "__main__":
    main()
