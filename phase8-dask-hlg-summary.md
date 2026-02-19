# Phase 8: Dask HighLevelGraph Rewrite

## Problem

The original `dask_graph.py` used `dask.delayed(func)(src_chunk)` where `src_chunk` was a dask array slice. Each `delayed()` call embedded the full source task subgraph into the delayed function's metadata. With hundreds of destination tiles, this caused:

- **272s graph build** for a 16384x16384 source (vs odc-geo's 0.086s)
- **281 MiB graph** that triggered dask's "Sending large graph" warning
- Crash on int8 dtype (since added in Phase 7)

## Root Cause

`dask.delayed` serialises its arguments eagerly. When passed a dask array slice (`src_data[r0:r1, c0:c1]`), it captures the entire upstream task graph for that slice. With ~600 tiles and 4 bands, this duplicated the source graph ~2,400 times.

## Solution

Replaced `dask.delayed` + `da.concatenate` with `HighLevelGraph.from_collections`, following odc-geo's approach:

1. **`_slice_to_block_indices`** — maps pixel-space source ROI to dask chunk block indices using pre-computed cumulative chunk boundaries
2. **`_reproject_from_blocks`** — worker function that assembles overlapping source blocks into a contiguous array, then calls `reproject_array`. Static args frozen via `functools.partial`
3. **Raw task dict** — `{(name, ri, ci): (proc, plan_dict, *src_block_keys)}` where dask resolves block keys before calling the worker. Zero graph duplication
4. **`HighLevelGraph.from_collections`** — wraps the task dict with an explicit dependency on the source array

## Benchmark Results

Source: 16384x16384 int8 (AEF embeddings from Source Cooperative)
Target: 16385x12956 EPSG:3857 (same pixel count, CRS change only)
Hardware: Apple M-series, 8 dask workers

### Graph Build

| Engine | Time |
|--------|------|
| odc-geo | 0.265s |
| rust-warp (before) | 272s |
| **rust-warp (after)** | **0.024s** |

**11,000x faster** graph build.

### Distributed Compute (8 workers, in-memory data on workers)

| Engine | Time |
|--------|------|
| odc-geo | 1.95s |
| rust-warp | 1.39s |

### Single-threaded Compute (dask synchronous scheduler)

| Engine | Time | Memory |
|--------|------|--------|
| odc-geo | 5.65s | +49 MiB |
| rust-warp | 0.66s | +3 MiB |

**8.5x faster, 16x less memory.**

### Raw Single-Process (no dask, rasterio vs rust-warp)

| Engine | Time | Memory |
|--------|------|--------|
| rasterio (GDAL) | 1.10s | +169 MiB |
| rust-warp | 0.17s | +1 MiB |

**6.5x faster, 169x less memory.**

### Network I/O Bound (Source Cooperative via VirtualiZarr)

| Engine | Time |
|--------|------|
| odc-geo | 13.7s |
| rust-warp | 15.7s |

Both engines are I/O bound at ~14s when data must be fetched over the network. Performance is equivalent in this regime.

## Numerical Accuracy

Comparison against odc-geo (212M pixels, nearest-neighbor, int8):

- **99.9987% exact match**
- ~1,100 off-by-1 pixels per band
- ~1,500 off-by->1 pixels per band (max diff 37–58)

The >1 differences occur at sharp edges in the AEF embedding space where adjacent source pixels have very different values. The two engines compute slightly different sub-pixel source coordinates (proj4rs vs PROJ), causing different nearest-neighbor pixel selection at boundaries. This is expected when using different projection libraries.

Direct comparison against rasterio/GDAL (same transform, 212M pixels):

- **99.9939% exact match**
- 5,426 off-by-1 pixels
- 7,438 off-by->1 pixels (max diff 52)

The GDAL comparison test suite passes with exact match (atol=0) at 64x64 and 256x256 sizes. The mismatches at 16K scale are from accumulated floating-point differences in the projection coordinate computation between proj4rs and PROJ.

## Files Changed

| File | Change |
|------|--------|
| `python/rust_warp/dask_graph.py` | Full rewrite: `dask.delayed` → `HighLevelGraph` |
| `tests/test_dask_graph_bench.py` | New: pytest-benchmark for graph build speed |
