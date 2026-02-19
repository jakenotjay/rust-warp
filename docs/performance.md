# Performance

rust-warp is designed to be faster than GDAL/rasterio for raster reprojection while maintaining pixel-level correctness. This document summarizes benchmark results from all implementation phases.

## Single-Chunk Reprojection

### rust-warp vs rasterio/GDAL (1024x1024, UTM 33N to EPSG:4326)

| Kernel | rust-warp | rasterio/GDAL | Speedup |
|--------|-----------|---------------|---------|
| nearest | 1.56 ms | 10.35 ms | **6.6x** |
| bilinear | 1.92 ms | 41.91 ms | **21.8x** |
| cubic | 3.31 ms | 68.27 ms | **20.6x** |
| lanczos | 8.84 ms | 87.20 ms | **9.9x** |
| average (4x down) | 1.02 ms | 4.38 ms | **4.3x** |

*Apple Silicon, release build with `target-cpu=native`.*

### Warp Engine Scaling (bilinear, Rayon default threads)

| Size | Time | Throughput |
|------|------|------------|
| 256x256 | 74 us | 886 MP/s |
| 512x512 | 298 us | 880 MP/s |
| 1024x1024 | 796 us | 1,317 MP/s |
| 2048x2048 | 1.21 ms | 3,467 MP/s |
| 4096x4096 | 13.2 ms | 1,271 MP/s |

Throughput peaks at 2048x2048 where data fits in cache while Rayon has enough rows to saturate all cores.

### Thread Scaling (bilinear, 1024x1024)

| Threads | Time | Speedup |
|---------|------|---------|
| 1 | 2.36 ms | 1.0x |
| 2 | 1.42 ms | 1.66x |
| 4 | 1.07 ms | 2.21x |
| 8 | 884 us | 2.67x |

Scaling is sub-linear due to memory bandwidth saturation on bilinear (which is memory-bound). Compute-bound kernels like cubic and lanczos scale better.

### Multi-Band (256x256, bilinear)

| Bands | rust-warp | rasterio | Speedup |
|-------|-----------|----------|---------|
| 3 | 0.32 ms | 8.93 ms | **27.8x** |
| 64 | 9.41 ms | 193.0 ms | **20.5x** |

## Dask Graph Construction

### HighLevelGraph vs dask.delayed (16384x16384 int8 source)

| Engine | Graph Build Time |
|--------|-----------------|
| rust-warp (HighLevelGraph) | **0.024s** |
| odc-geo | 0.265s |
| rust-warp (old dask.delayed) | 272s |

The HighLevelGraph rewrite (Phase 8) achieved an **11,000x speedup** in graph construction by referencing source blocks by key rather than embedding subgraphs.

### Distributed Compute (8 workers, 16384x16384)

| Engine | Compute Time |
|--------|-------------|
| rust-warp | **1.39s** |
| odc-geo | 1.95s |

### Single-Process Compute (synchronous scheduler)

| Engine | Compute Time | Peak Memory |
|--------|-------------|-------------|
| rust-warp | **0.66s** | +3 MiB |
| odc-geo | 5.65s | +49 MiB |

**8.5x faster, 16x less memory** on single-threaded dask compute.

### Raw Single-Process (no dask overhead)

| Engine | Time | Peak Memory |
|--------|------|-------------|
| rust-warp | **0.17s** | +1 MiB |
| rasterio/GDAL | 1.10s | +169 MiB |

**6.5x faster, 169x less memory.**

## Projection Throughput

| Projection | Points/sec |
|------------|------------|
| Web Mercator (native) | 189M |
| UTM 33 Transverse Mercator (native) | 5.4M |

Native projections are significantly faster than the proj4rs fallback due to avoiding PROJ string parsing and dispatch overhead.

## Chunk Planning Overhead

| Tiles | Planning Time |
|-------|--------------|
| 4 | 39 us |
| 16 | 153 us |
| 64 | 612 us |
| 256 | 2.46 ms |

Planning overhead is negligible relative to compute time for all practical tile counts.

## Optimization Details

### What makes rust-warp fast

1. **Rayon parallelism** — the warp loop distributes output rows across all available CPU cores. Each row is independent (shared source read, disjoint output write), so there's no synchronization overhead.

2. **Linear approximation** — instead of computing exact projection transforms for every pixel, `LinearApprox` computes 3 exact points per scanline and linearly interpolates between them. For near-linear transforms (UTM, Web Mercator), this reduces projection calls by ~50-100x per row. Recursive subdivision ensures accuracy never exceeds 0.125 pixels.

3. **Native projection math** — for common CRSes (all UTM zones, Web Mercator, Equirectangular), rust-warp uses hand-written Rust implementations that avoid the overhead of PROJ string parsing and generic dispatch.

4. **Polynomial sin approximation** — the Lanczos kernel replaces `sin()` calls with a degree-11 Taylor polynomial (max error ~6e-8), avoiding expensive transcendental function calls in the inner loop.

5. **Weight precomputation** — cubic and average kernels precompute 1D weight arrays before the 2D convolution loop, reducing redundant `cubic_weight()` calls from 16 to 8 (cubic) and eliminating redundant overlap calculations (average).

6. **`target-cpu=native`** — enables NEON (Apple Silicon) or AVX2 (x86_64) auto-vectorization for the inner loop. LLVM vectorizes bilinear's 4-point weighted sum and the linear interpolation path.

7. **HighLevelGraph construction** — the dask graph builder references source blocks by key rather than embedding task subgraphs, making graph construction O(tiles) instead of O(tiles * source_graph_size).

8. **GIL release** — all computation runs outside the Python GIL, so multiple Python threads can reproject concurrently and dask's threaded scheduler works efficiently.

### Network-bound regime

When data must be fetched over the network (e.g., remote Zarr stores, COGs on S3), both rust-warp and odc-geo converge to the same performance (~14s for 16384x16384) because I/O dominates compute. The rust-warp advantage is most pronounced for in-memory or locally-cached data.

## Running Benchmarks

### Rust benchmarks (Criterion)

```bash
cargo bench
```

Results are saved to `target/criterion/` with HTML reports.

### Python benchmarks (pytest-benchmark)

```bash
uv run pytest tests/test_benchmark.py --benchmark-enable -v
```

### Dask graph build benchmarks

```bash
uv run pytest tests/test_dask_graph_bench.py --benchmark-enable -v
```
