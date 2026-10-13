"""Estimate pi with AMD HIP Python bindings and HIPRTC kernels."""

from __future__ import annotations

import argparse
import ctypes
import math
from dataclasses import dataclass

import numpy as np

from hip import hip, hiprand, hiprtc


EXACT_PI = math.pi
QUARTER_CIRCLE_PROBABILITY = EXACT_PI / 4.0

_KERNEL_SOURCE = b"""
extern "C" __global__ void count_points(
    unsigned long long n, const double* points, unsigned long long* count) {
  unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = points[2 * i];
    double y = points[2 * i + 1];
    if (x * x + y * y <= 1.0) atomicAdd(count, 1ULL);
  }
}

extern "C" __global__ void count_stratified(
    unsigned long long n, unsigned long long grid_size,
    unsigned long long samples_per_stratum, const double* offsets,
    unsigned long long* count) {
  unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned long long cell = i / samples_per_stratum;
    unsigned long long row = cell / grid_size;
    unsigned long long column = cell % grid_size;
    double x = (column + offsets[2 * i]) / grid_size;
    double y = (row + offsets[2 * i + 1]) / grid_size;
    if (x * x + y * y <= 1.0) atomicAdd(count, 1ULL);
  }
}
"""


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


def _hip_check(result: tuple, *, roc_rand: bool = False):
    error = result[0]
    if roc_rand:
        success = hiprand.hiprandStatus.HIPRAND_STATUS_SUCCESS
    elif isinstance(error, hiprtc.hiprtcResult):
        success = hiprtc.hiprtcResult.HIPRTC_SUCCESS
    else:
        success = hip.hipError_t.hipSuccess
    if error != success:
        raise RuntimeError(str(error))
    values = result[1:]
    return values[0] if len(values) == 1 else values


def _compile_kernels():
    program = _hip_check(
        hiprtc.hiprtcCreateProgram(_KERNEL_SOURCE, b"pi_kernels", 0, [], [])
    )
    try:
        properties = hip.hipDeviceProp_t()
        _hip_check(hip.hipGetDeviceProperties(properties, 0))
        architecture = properties.gcnArchName
        if isinstance(architecture, str):
            architecture = architecture.encode()
        flags = [b"--offload-arch=" + architecture]
        error = hiprtc.hiprtcCompileProgram(program, len(flags), flags)[0]
        if error != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = _hip_check(hiprtc.hiprtcGetProgramLogSize(program))
            log = bytearray(log_size)
            _hip_check(hiprtc.hiprtcGetProgramLog(program, log))
            raise RuntimeError(log.decode())
        code_size = _hip_check(hiprtc.hiprtcGetCodeSize(program))
        code = bytearray(code_size)
        _hip_check(hiprtc.hiprtcGetCode(program, code))
    finally:
        _hip_check(hiprtc.hiprtcDestroyProgram(program.createRef()))

    module = _hip_check(hip.hipModuleLoadData(code))
    return (
        module,
        _hip_check(hip.hipModuleGetFunction(module, b"count_points")),
        _hip_check(hip.hipModuleGetFunction(module, b"count_stratified")),
    )


_KERNELS = None


def _get_kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _compile_kernels()
    return _KERNELS


def _new_generator(seed: int | None):
    generator = _hip_check(
        hiprand.hiprandCreateGenerator(
            hiprand.hiprandRngType.HIPRAND_RNG_PSEUDO_DEFAULT
        ),
        roc_rand=True,
    )
    _hip_check(
        hiprand.hiprandSetPseudoRandomGeneratorSeed(
            generator, 0 if seed is None else seed
        ),
        roc_rand=True,
    )
    return generator


def _count_on_device(
    samples: int,
    generator,
    *,
    grid_size: int | None = None,
    samples_per_stratum: int = 1,
) -> int:
    point_count = 2 * samples
    points = _hip_check(hip.hipMalloc(point_count * np.dtype(np.float64).itemsize))
    count = _hip_check(hip.hipMalloc(np.dtype(np.uint64).itemsize))
    host_count = np.zeros(1, dtype=np.uint64)
    try:
        _hip_check(
            hiprand.hiprandGenerateUniformDouble(generator, points, point_count),
            roc_rand=True,
        )
        _hip_check(hip.hipMemset(count, 0, host_count.nbytes))
        _, plain_kernel, stratified_kernel = _get_kernels()
        kernel = plain_kernel if grid_size is None else stratified_kernel
        block = hip.dim3(x=256)
        grid = hip.dim3(x=math.ceil(samples / block.x))
        arguments = [ctypes.c_ulonglong(samples)]
        if grid_size is not None:
            arguments.extend(
                [
                    ctypes.c_ulonglong(grid_size),
                    ctypes.c_ulonglong(samples_per_stratum),
                ]
            )
        arguments.extend([points, count])
        _hip_check(
            hip.hipModuleLaunchKernel(
                kernel,
                *grid,
                *block,
                sharedMemBytes=0,
                stream=None,
                kernelParams=None,
                extra=tuple(arguments),
            )
        )
        _hip_check(hip.hipDeviceSynchronize())
        _hip_check(
            hip.hipMemcpy(
                host_count,
                count,
                host_count.nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToHost,
            )
        )
        return int(host_count[0])
    finally:
        _hip_check(hip.hipFree(points))
        _hip_check(hip.hipFree(count))


def estimate_pi(samples: int, *, seed: int | None = None) -> Estimate:
    """Return a HIP Monte Carlo estimate of pi using device-side counting."""
    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    generator = _new_generator(seed)
    try:
        inside = _count_on_device(samples, generator)
    finally:
        _hip_check(hiprand.hiprandDestroyGenerator(generator), roc_rand=True)
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
    generator = _new_generator(seed)
    try:
        inside = _count_on_device(
            samples,
            generator,
            grid_size=grid_size,
            samples_per_stratum=samples_per_stratum,
        )
    finally:
        _hip_check(hiprand.hiprandDestroyGenerator(generator), roc_rand=True)
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
    """Measure RMSE and bias over independent HIP trials."""
    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")
    generator = _new_generator(seed)
    try:
        estimates = np.array(
            [
                4.0 * _count_on_device(samples, generator) / samples
                for _ in range(trials)
            ]
        )
    finally:
        _hip_check(hiprand.hiprandDestroyGenerator(generator), roc_rand=True)
    errors = estimates - EXACT_PI
    return EmpiricalError(
        samples, trials, float(np.sqrt(np.mean(errors**2))), float(np.mean(errors))
    )


def stratified_empirical_error(
    grid_size: int, samples_per_stratum: int, trials: int, *, seed: int | None = None
) -> EmpiricalError:
    """Measure stratified RMSE and bias over independent HIP trials."""
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")
    samples = grid_size**2 * samples_per_stratum
    generator = _new_generator(seed)
    try:
        estimates = np.array(
            [
                4.0
                * _count_on_device(
                    samples,
                    generator,
                    grid_size=grid_size,
                    samples_per_stratum=samples_per_stratum,
                )
                / samples
                for _ in range(trials)
            ]
        )
    finally:
        _hip_check(hiprand.hiprandDestroyGenerator(generator), roc_rand=True)
    errors = estimates - EXACT_PI
    return EmpiricalError(
        samples, trials, float(np.sqrt(np.mean(errors**2))), float(np.mean(errors))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HIP Python Monte Carlo estimate of pi")
    parser.add_argument("samples", nargs="?", type=int, help="sample count (default: 1000000)")
    parser.add_argument("--grid-size", type=int, help="use an m by m equal-area grid")
    parser.add_argument("--samples-per-stratum", type=int, default=1)
    parser.add_argument("--seed", type=int, help="optional reproducible random seed")
    parser.add_argument("--trials", type=int, help="report empirical RMSE and bias")
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
