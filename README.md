# cuda-perception-kernels

CUDA C++ implementations of image processing kernels, benchmarked against OpenCV CPU baselines.

## Kernels

| Kernel | Description |
|---|---|
| Bilateral filter | Edge-preserving depth map smoothing using shared memory tiling |

## Performance

Benchmarked on **1920×1080 float32** images. OpenCV runs with default threading on a 30-core CPU.

| Kernel | CUDA (A10 GPU) | OpenCV CPU | Speedup |
|---|---|---|---|
| Bilateral filter (radius=5, σ_s=3, σ_r=0.5) | 0.668 ms | 1.736 ms | **2.6×** |

**Test hardware:** NVIDIA A10 (24 GB PCIe), 30-core CPU @ 2.59 GHz · Lambda Labs · 2026-06-28

## Implementation notes

- Shared memory tiling with halo loading to minimise global memory traffic
- Border handling via clamping (equivalent to OpenCV `BORDER_REPLICATE`)
- Unnormalised Gaussians — normalisation coefficients cancel in the weighted average and are omitted

## Environment setup

Tested on Ubuntu 24.04 with CUDA pre-installed (Lambda Labs Lambda Stack image).

```bash
sudo apt update
sudo apt install -y cmake build-essential libopencv-dev libgtest-dev libbenchmark-dev
```

## Build

**Requirements:** CMake ≥ 3.18, CUDA toolkit, OpenCV ≥ 4.5, Google Benchmark, GoogleTest

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Tests

```bash
./run_tests
```

## Benchmarks

```bash
./run_benchmarks
```
