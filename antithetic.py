"""Estimate pi using rejection, semicircle, and Cauchy Monte Carlo estimators

with optional antithetic variates, JAX JIT acceleration, and profiling tools.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import time

import jax
from jax import numpy as jnp
from jax import random

EXACT_PI = float(jnp.pi)


@dataclass(frozen=True)
class Estimate:
    samples: int
    pi_estimate: float
    runtime_sec: float

    @property
    def absolute_error(self) -> float:
        return abs(self.pi_estimate - EXACT_PI)

    @property
    def throughput_msamples_per_sec(self) -> float:
        return (self.samples / self.runtime_sec) / 1e6 if self.runtime_sec > 0 else 0.0


@dataclass(frozen=True)
class EmpiricalResult:
    samples_per_trial: int
    trials: int
    rmse: float
    bias: float
    runtime_sec: float

    @property
    def total_samples(self) -> int:
        return self.samples_per_trial * self.trials

    @property
    def throughput_msamples_per_sec(self) -> float:
        return (self.total_samples / self.runtime_sec) / 1e6 if self.runtime_sec > 0 else 0.0


# =============================================================================
# Core Compiled Kernels
# =============================================================================


@partial(jax.jit, static_argnames=("samples", "antithetic"))
def _sample_rejection(samples: int, key: jax.Array, antithetic: bool) -> jax.Array:
    if antithetic:
        pairs = samples // 2
        u = random.uniform(key, shape=(pairs, 2))
        u_anti = 1.0 - u
        hits_orig = jnp.count_nonzero(jnp.sum(u**2, axis=-1) <= 1.0)
        hits_anti = jnp.count_nonzero(jnp.sum(u_anti**2, axis=-1) <= 1.0)
        return 4.0 * (hits_orig + hits_anti) / (2 * pairs)

    u = random.uniform(key, shape=(samples, 2))
    hits = jnp.count_nonzero(jnp.sum(u**2, axis=-1) <= 1.0)
    return 4.0 * hits / samples


@partial(jax.jit, static_argnames=("samples", "antithetic"))
def _sample_semicircle(samples: int, key: jax.Array, antithetic: bool) -> jax.Array:
    if antithetic:
        pairs = samples // 2
        u = random.uniform(key, shape=(pairs,))
        vals_orig = 4.0 * jnp.sqrt(1.0 - u**2)
        vals_anti = 4.0 * jnp.sqrt(1.0 - (1.0 - u) ** 2)
        return jnp.mean(0.5 * (vals_orig + vals_anti))

    u = random.uniform(key, shape=(samples,))
    return jnp.mean(4.0 * jnp.sqrt(1.0 - u**2))


@partial(jax.jit, static_argnames=("samples", "antithetic"))
def _sample_cauchy(samples: int, key: jax.Array, antithetic: bool) -> jax.Array:
    if antithetic:
        pairs = samples // 2
        u = random.uniform(key, shape=(pairs,))
        vals_orig = 4.0 / (1.0 + u**2)
        vals_anti = 4.0 / (1.0 + (1.0 - u) ** 2)
        return jnp.mean(0.5 * (vals_orig + vals_anti))

    u = random.uniform(key, shape=(samples,))
    return jnp.mean(4.0 / (1.0 + u**2))


# =============================================================================
# Vectorized Multi-Trial Dispatch
# =============================================================================


@partial(jax.jit, static_argnames=("method", "samples", "antithetic"))
def _run_trials(
    method: str, samples: int, antithetic: bool, keys: jax.Array
) -> jax.Array:
    if method == "rejection":
        return jax.vmap(lambda k: _sample_rejection(samples, k, antithetic))(keys)
    if method == "semicircle":
        return jax.vmap(lambda k: _sample_semicircle(samples, k, antithetic))(keys)
    if method == "cauchy":
        return jax.vmap(lambda k: _sample_cauchy(samples, k, antithetic))(keys)
    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Benchmarked Driver Routines
# =============================================================================


def estimate_pi_timed(
    method: str,
    samples: int,
    *,
    antithetic: bool = False,
    seed: int = 0,
) -> Estimate:
    """Run a single timed estimate, compiling during a warmup pass."""
    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    if antithetic and samples % 2 != 0:
        raise ValueError("samples must be even when using antithetic sampling")

    key = random.PRNGKey(seed)
    kernel_dispatch = {
        "rejection": _sample_rejection,
        "semicircle": _sample_semicircle,
        "cauchy": _sample_cauchy,
    }
    kernel = kernel_dispatch[method]

    # Warmup to compile kernel for this exact shape/static args
    _ = kernel(samples, key, antithetic).block_until_ready()

    # Timed run
    t0 = time.perf_counter()
    val = kernel(samples, key, antithetic).block_until_ready()
    elapsed = time.perf_counter() - t0

    return Estimate(samples, float(val), elapsed)


def run_empirical_benchmark(
    method: str,
    samples: int,
    trials: int,
    *,
    antithetic: bool = False,
    seed: int = 0,
) -> EmpiricalResult:
    """Measure empirical RMSE, bias, and throughput across vectorized trials."""
    if samples <= 0 or trials <= 0:
        raise ValueError("samples and trials must be positive integers")
    if antithetic and samples % 2 != 0:
        raise ValueError("samples must be even when using antithetic sampling")

    key = random.PRNGKey(seed)
    keys = random.split(key, trials)

    # Warmup
    _ = _run_trials(method, samples, antithetic, keys).block_until_ready()

    # Timed run
    t0 = time.perf_counter()
    estimates = _run_trials(method, samples, antithetic, keys).block_until_ready()
    elapsed = time.perf_counter() - t0

    errors = estimates - EXACT_PI
    rmse = float(jnp.sqrt(jnp.mean(errors**2)))
    bias = float(jnp.mean(errors))

    return EmpiricalResult(samples, trials, rmse, bias, elapsed)


def run_convergence_sweep(
    method: str,
    antithetic: bool,
    trials: int = 30,
    seed: int = 0,
) -> None:
    """Evaluate error across multiple sample sizes to confirm 1/sqrt(N) decay."""
    sample_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 4_000_000]
    rmses: list[float] = []

    mode_str = "Antithetic" if antithetic else "Standard IID"
    print(f"\nConvergence sweep: {method.capitalize()} ({mode_str}), {trials} trials each")
    print(f"{'Samples (N)':>12} | {'RMSE':>14} | {'Throughput (MS/s)':>20}")
    print("-" * 52)

    key = random.PRNGKey(seed)
    for n in sample_sizes:
        key, subkey = random.split(key)
        res = run_empirical_benchmark(
            method, n, trials, antithetic=antithetic, seed=int(subkey[0])
        )
        rmses.append(res.rmse)
        print(f"{n:>12,} | {res.rmse:>14.8f} | {res.throughput_msamples_per_sec:>20.2f}")

    # Compute empirical slope on log-log scale: d(log RMSE) / d(log N)
    log_n = jnp.log(jnp.array(sample_sizes, dtype=jnp.float32))
    log_rmse = jnp.log(jnp.array(rmses, dtype=jnp.float32))
    slope = float(jnp.cov(log_n, log_rmse)[0, 1] / jnp.var(log_n, ddof=1))

    print("-" * 52)
    print(f"Empirical slope d(log RMSE)/d(log N): {slope:.3f} (Theoretical: -0.500)\n")


def run_comparison_table(samples: int = 1_000_000, trials: int = 50, seed: int = 0) -> None:
    """Compare all 6 estimator combinations in an executive summary table."""
    configurations = [
        ("rejection", False),
        ("rejection", True),
        ("semicircle", False),
        ("semicircle", True),
        ("cauchy", False),
        ("cauchy", True),
    ]

    print(f"\nMonte Carlo Pi Estimators Benchmark ({samples:,} samples/trial, {trials} trials)")
    print(f"{'Method':<12} | {'Mode':<12} | {'RMSE':<14} | {'Bias':<14} | {'Throughput (MS/s)':<18}")
    print("-" * 78)

    key = random.PRNGKey(seed)
    for method, anti in configurations:
        key, subkey = random.split(key)
        res = run_empirical_benchmark(
            method, samples, trials, antithetic=anti, seed=int(subkey[0])
        )
        mode_label = "Antithetic" if anti else "Standard IID"
        print(
            f"{method.capitalize():<12} | {mode_label:<12} | {res.rmse:<14.8f} | "
            f"{res.bias:<14.8f} | {res.throughput_msamples_per_sec:<18.2f}"
        )
    print("-" * 78)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-throughput Monte Carlo Pi estimators in JAX"
    )
    parser.add_argument(
        "method",
        nargs="?",
        choices=["rejection", "semicircle", "cauchy"],
        default="rejection",
        help="estimator algorithm (default: rejection)",
    )
    parser.add_argument(
        "pos_samples",
        nargs="?",
        type=int,
        default=None,
        help="sample count per trial positional (default: 1,000,000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="sample count per trial flag (default: 1,000,000)",
    )
    parser.add_argument(
        "--antithetic",
        action="store_true",
        help="enable antithetic pairs (1 - u)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="evaluate multiple independent trials via vmap to report empirical RMSE and bias",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run a sample size scaling sweep to verify 1/sqrt(N) convergence slope",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run head-to-head benchmark comparing all 6 estimator modes",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()

    # Resolve samples from flag or positional argument
    samples = args.samples or args.pos_samples or 1_000_000

    if args.compare:
        run_comparison_table(samples=samples, trials=args.trials or 50, seed=args.seed)
        return

    if args.sweep:
        run_convergence_sweep(args.method, args.antithetic, trials=args.trials or 30, seed=args.seed)
        return

    mode = "Antithetic" if args.antithetic else "Standard IID"

    if args.trials is not None:
        empirical = run_empirical_benchmark(
            args.method,
            samples,
            args.trials,
            antithetic=args.antithetic,
            seed=args.seed,
        )
        print(f"Method:     {args.method.capitalize()} ({mode})")
        print(f"Samples:    {samples:,} x {args.trials} trials ({empirical.total_samples:,} total)")
        print(f"Emp RMSE:   {empirical.rmse:.10f}")
        print(f"Emp Bias:   {empirical.bias:.10f}")
        print(f"Runtime:    {empirical.runtime_sec * 1e3:.2f} ms")
        print(f"Throughput: {empirical.throughput_msamples_per_sec:.2f} MSamples/s")
    else:
        single = estimate_pi_timed(
            args.method,
            samples,
            antithetic=args.antithetic,
            seed=args.seed,
        )
        print(f"Method:     {args.method.capitalize()} ({mode})")
        print(f"Samples:    {single.samples:,}")
        print(f"Estimate:   {single.pi_estimate:.10f}")
        print(f"Exact:      {EXACT_PI:.10f}")
        print(f"Abs Error:  {single.absolute_error:.10f}")
        print(f"Runtime:    {single.runtime_sec * 1e3:.2f} ms")
        print(f"Throughput: {single.throughput_msamples_per_sec:.2f} MSamples/s")

if __name__ == "__main__":
    main()