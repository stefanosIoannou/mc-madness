"""Estimate pi with independent JAX RNG streams and MPI reduction."""

from __future__ import annotations

import argparse

import jax
import numpy as np
from mpi4py import MPI
from jax import random

from pi_serial import EXACT_PI, _sample_inside_quarter_circle, pi_standard_error


def main() -> None:
    parser = argparse.ArgumentParser(description="MPI JAX Monte Carlo estimate of pi")
    parser.add_argument("samples", nargs="?", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0, help="reproducible base seed")
    args = parser.parse_args()

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    world_size = communicator.Get_size()

    if args.samples <= 0:
        raise ValueError("samples must be a positive integer")

    # Distribute the remainder so every requested sample is counted exactly once.
    local_samples = args.samples // world_size + (rank < args.samples % world_size)
    key = random.fold_in(random.PRNGKey(args.seed), rank)
    local_inside = int(_sample_inside_quarter_circle(local_samples, key))

    global_counts = np.zeros(2, dtype=np.int64)
    communicator.Allreduce(
        np.array([local_inside, local_samples], dtype=np.int64),
        global_counts,
        op=MPI.SUM,
    )

    if rank == 0:
        total_inside, total_samples = map(int, global_counts)
        estimate = 4.0 * total_inside / total_samples
        print(f"ranks:    {world_size}")
        print(f"samples:  {total_samples:,}")
        print(f"inside:   {total_inside:,}")
        print(f"estimate: {estimate:.10f}")
        print(f"exact:    {EXACT_PI:.10f}")
        print(f"abs error:{abs(estimate - EXACT_PI):.10f}")
        print(f"std error:{pi_standard_error(total_samples):.10f}")


if __name__ == "__main__":
    main()