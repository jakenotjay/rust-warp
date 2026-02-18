# rust-warp — Implementation Plan

## Rust-Powered Raster Reprojection for Xarray/Dask

**Project codename:** `rust-warp`
**Target:** A Python library backed by a Rust core that reprojects chunked raster data without GDAL, integrating natively with xarray and dask.

---

## Table of Contents

1. [Development Environment Setup (macOS)](#1-development-environment-setup-macos)
2. [Project Structure and Build System](#2-project-structure-and-build-system)
3. [Phase 1 — Rust Core: Projection Math](#3-phase-1--rust-core-projection-math)
4. [Phase 2 — Rust Core: Warp Engine and Resampling Kernels](#4-phase-2--rust-core-warp-engine-and-resampling-kernels)
5. [Phase 3 — Python Bindings via PyO3](#5-phase-3--python-bindings-via-pyo3)
6. [Phase 4 — Chunk Planner and Dask Integration](#6-phase-4--chunk-planner-and-dask-integration)
7. [Phase 5 — Xarray Accessor and High-Level API](#7-phase-5--xarray-accessor-and-high-level-api)
8. [Phase 6 — SIMD Optimisation and Parallelism](#8-phase-6--simd-optimisation-and-parallelism)
9. [Phase 7 — Production Hardening](#9-phase-7--production-hardening)
10. [Testing Plan](#10-testing-plan)
11. [Benchmark Plan](#11-benchmark-plan)
12. [Appendix A — Key Mathematical References](#appendix-a--key-mathematical-references)
13. [Appendix B — File-by-File Manifest](#appendix-b--file-by-file-manifest)

---

## 1. Development Environment Setup (macOS)

### 1.1 Prerequisites Installation

Run these commands in order. Each step depends on the previous.

```bash
# ── Step 1: Homebrew (skip if already installed) ──
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# ── Step 2: Rust toolchain via rustup ──
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# When prompted, choose: 1) Proceed with standard installation
# Then reload your shell:
source "$HOME/.cargo/env"

# Verify:
rustc --version   # Should be >= 1.75.0
cargo --version

# ── Step 3: Add useful Rust components ──
rustup component add clippy rustfmt
cargo install cargo-nextest   # Better test runner
cargo install cargo-criterion # Benchmarking

# ── Step 4: Python via uv ──
# uv is the modern Python package manager from Astral (the ruff people)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"   # uv installs to ~/.cargo/bin

# Verify:
uv --version  # Should be >= 0.5.x

# ── Step 5: Install maturin as a uv tool ──
uv tool install maturin

# Verify:
maturin --version  # Should be >= 1.7.x

# ── Step 6: System dependencies for benchmarking ──
# We need GDAL and PROJ for the comparison benchmarks (not for rust-warp itself)
brew install gdal proj
```

### 1.2 IDE Setup (Recommended)

```bash
# VS Code with Rust and Python extensions:
# - rust-analyzer (Rust language server)
# - Even Better TOML (for Cargo.toml)
# - Python + Pylance
# - Ruff (linting)

# For CLion / RustRover users:
# - The Python plugin + PyO3-aware inspection is excellent
```

### 1.3 Apple Silicon Note

If on Apple Silicon (M1/M2/M3/M4), Rust compiles natively for `aarch64-apple-darwin`. NEON SIMD intrinsics are available via `std::arch::aarch64`. The project will use `cfg` attributes to select NEON vs AVX2:

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
```

---

## 2. Project Structure and Build System

### 2.1 Build System Decision

> **Important:** The user requested hatchling, but for Rust+Python projects, **maturin IS the build backend**. Hatchling cannot compile Rust code. The standard 2025 stack is:
> - **uv** — Python project/dependency management (frontend)
> - **maturin** — Build backend (compiles Rust, packages wheels)
>
> The Python-side packaging (version, metadata, dependencies) is handled by maturin's pyproject.toml support, which covers everything hatchling does for pure-Python projects. We use a "mixed" layout where Python source lives alongside the Rust crate.

### 2.2 Project Initialisation Script

Run this to create the entire project skeleton:

```bash
# ── Create and enter project directory ──
mkdir rust-warp && cd rust-warp
git init

# ── Initialise with uv (creates pyproject.toml and .python-version) ──
uv init --lib --build-backend maturin --name rust-warp

# This creates:
#   rust-warp/
#   ├── .python-version
#   ├── Cargo.toml
#   ├── pyproject.toml
#   ├── README.md
#   └── src/
#       └── lib.rs

# ── Restructure into the mixed layout we need ──
# The mixed layout has Python code in python/rust-warp/ and Rust in src/
mkdir -p python/rust-warp
mv src/lib.rs src/lib.rs.bak  # We'll rewrite this

# Create the Python package init
mkdir -p python/rust-warp
```

### 2.3 pyproject.toml (Full Version)

This is the central configuration file. **Create this exactly:**

```toml
[project]
name = "rust-warp"
version = "0.1.0"
description = "High-performance raster reprojection engine in Rust with xarray/dask integration"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
keywords = ["geospatial", "reprojection", "raster", "xarray", "dask", "rust"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: GIS",
    "Programming Language :: Rust",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
xarray = [
    "xarray>=2023.1",
    "dask[array]>=2023.1",
    "pyproj>=3.4",
]
all = [
    "rust-warp[xarray]",
    "odc-geo>=0.4",
]
dev = [
    "rust-warp[all]",
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "rioxarray>=0.15",
    "rasterio>=1.3",
    "odc-geo>=0.4",
    "pyproj>=3.4",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
# "mixed" layout: Python in python/, Rust in src/
python-source = "python"
module-name = "rust-warp._rust"
features = ["pyo3/extension-module"]

[tool.uv]
# Rebuild when Rust sources change
cache-keys = [
    { file = "pyproject.toml" },
    { file = "Cargo.toml" },
    { file = "src/**/*.rs" },
    { file = "crates/**/*.rs" },
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "benchmark: marks tests as benchmarks (deselect with '-m \"not benchmark\"')",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

### 2.4 Cargo.toml (Workspace Root)

```toml
[package]
name = "rust-warp"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[lib]
name = "_rust"
crate-type = ["cdylib", "rlib"]
# cdylib = shared library for Python; rlib = for Rust tests

[dependencies]
# ── Python bindings ──
pyo3 = { version = "0.23", features = ["abi3-py310"] }
numpy = "0.23"          # rust-numpy: PyO3 ↔ ndarray bridge

# ── Core computation ──
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.10"           # Data parallelism

# ── Projection math ──
proj4rs = "0.1"          # Pure-Rust PROJ.4 implementation

# ── Utilities ──
thiserror = "2.0"        # Error types
log = "0.4"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"           # Float comparison in tests

[[bench]]
name = "warp_bench"
harness = false

[profile.release]
opt-level = 3
lto = "fat"              # Link-time optimisation for maximum speed
codegen-units = 1        # Better optimisation, slower compile
strip = true             # Smaller binary
```

### 2.5 Directory Layout (Target State)

```
rust-warp/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Rust tests + Python tests + linting
│       └── release.yml             # Build wheels for all platforms
├── .python-version                 # e.g. "3.12"
├── Cargo.toml                      # Rust workspace root
├── Cargo.lock
├── pyproject.toml                  # Python project config + maturin config
├── README.md
├── LICENSE
│
├── src/                            # ── Rust source ──
│   ├── lib.rs                      # PyO3 module definition
│   ├── error.rs                    # Error types
│   ├── affine.rs                   # Affine transform (6-param geotransform)
│   ├── geobox.rs                   # GeoBox: CRS + affine + shape
│   │
│   ├── proj/                       # Projection math
│   │   ├── mod.rs
│   │   ├── ellipsoid.rs            # WGS84, GRS80 constants
│   │   ├── common.rs               # Shared helpers (meridian_distance, etc.)
│   │   ├── transverse_mercator.rs  # Krüger n-series, 6th order
│   │   ├── mercator.rs             # Normal Mercator + Web Mercator (3857)
│   │   ├── lambert_conformal.rs    # 1SP and 2SP variants
│   │   ├── albers_equal_area.rs
│   │   ├── stereographic.rs        # Polar + oblique
│   │   ├── sinusoidal.rs
│   │   ├── equirectangular.rs      # Plate Carrée
│   │   ├── pipeline.rs             # CRS_A → geodetic → CRS_B chain
│   │   └── approx.rs              # Linear approximation of transforms
│   │
│   ├── resample/                   # Resampling kernels
│   │   ├── mod.rs
│   │   ├── nearest.rs
│   │   ├── bilinear.rs
│   │   ├── cubic.rs
│   │   ├── lanczos.rs
│   │   └── average.rs
│   │
│   ├── warp/                       # Warp engine
│   │   ├── mod.rs
│   │   ├── engine.rs               # Main inverse-mapping loop
│   │   ├── nodata.rs               # NaN/nodata mask propagation
│   │   └── simd.rs                 # Platform-specific SIMD paths
│   │
│   ├── chunk/                      # Chunk planning for dask
│   │   ├── mod.rs
│   │   ├── planner.rs              # Source↔dest overlap computation
│   │   └── halo.rs                 # Kernel-radius padding
│   │
│   └── py/                         # PyO3 binding layer
│       ├── mod.rs
│       ├── reproject.rs            # reproject_array() function
│       ├── plan.rs                 # plan_reproject() function
│       └── types.rs                # Python-visible types
│
├── python/                         # ── Python source ──
│   └── rust-warp/
│       ├── __init__.py             # Public API
│       ├── _rust.pyi               # Type stubs for the Rust module
│       ├── geobox.py               # GeoBox class (Python side)
│       ├── reproject.py            # High-level reproject() function
│       ├── dask_graph.py           # Dask graph builder
│       └── xarray_accessor.py      # .rust-warp xarray accessor
│
├── tests/                          # ── Test suite ──
│   ├── conftest.py                 # Shared fixtures
│   ├── rust/                       # Pure Rust tests (cargo test)
│   │   └── (inline in src/*.rs)
│   ├── test_projections.py         # Projection accuracy vs pyproj
│   ├── test_resampling.py          # Kernel output vs scipy/GDAL
│   ├── test_warp.py                # Full reprojection correctness
│   ├── test_dask.py                # Chunked reprojection
│   ├── test_xarray.py             # Accessor integration
│   └── benchmarks/
│       ├── bench_projections.py    # Projection throughput
│       ├── bench_warp.py           # Per-chunk warp time
│       ├── bench_e2e.py            # End-to-end vs GDAL/odc-geo
│       └── bench_parallel.py       # Multi-thread scaling
│
├── benches/                        # ── Rust benchmarks (Criterion) ──
│   └── warp_bench.rs
│
└── scripts/
    ├── generate_test_data.py       # Create synthetic rasters for testing
    └── run_benchmarks.py           # Full benchmark suite runner
```

### 2.6 Development Workflow Commands

```bash
# ── First-time setup ──
cd rust-warp
uv sync --all-extras          # Creates venv, installs all deps + compiles Rust

# ── After editing Rust code ──
uv run maturin develop --release   # Recompile and reinstall into venv
# OR for debug builds (faster compile, slower runtime):
uv run maturin develop

# ── Run Python tests ──
uv run pytest tests/ -x -v

# ── Run Rust tests ──
cargo test

# ── Run Rust benchmarks ──
cargo bench

# ── Run Python benchmarks ──
uv run pytest tests/benchmarks/ -m benchmark --benchmark-json=results.json

# ── Lint everything ──
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv run ruff check python/ tests/

# ── Build a release wheel ──
uv run maturin build --release
```

---

## 3. Phase 1 — Rust Core: Projection Math

**Goal:** Implement forward and inverse coordinate transformations for the most common CRSes, matching PROJ's output to sub-millimetre accuracy.

**Duration estimate:** 2-3 weeks

### 3.1 Deliverables

| File | Contents |
|---|---|
| `src/proj/ellipsoid.rs` | Ellipsoid constants (WGS84, GRS80, Bessel, Clarke 1866, International 1924) |
| `src/proj/common.rs` | Meridional arc length, authalic/conformal latitude helpers |
| `src/proj/transverse_mercator.rs` | Krüger n-series forward/inverse, 6th order |
| `src/proj/mercator.rs` | Normal Mercator and Web Mercator (EPSG:3857) |
| `src/proj/lambert_conformal.rs` | Lambert Conformal Conic 1SP and 2SP |
| `src/proj/albers_equal_area.rs` | Albers Equal Area Conic |
| `src/proj/stereographic.rs` | Polar Stereographic (EPSG:3031/3413) and oblique |
| `src/proj/sinusoidal.rs` | Sinusoidal (MODIS native grid) |
| `src/proj/equirectangular.rs` | Plate Carrée (EPSG:4326 treated as projection) |
| `src/proj/pipeline.rs` | Chain inverse_A → forward_B, batch transform N points |
| `src/proj/approx.rs` | 3-point linear approximation per scanline |
| `src/affine.rs` | 6-parameter affine transform, inversion, composition |

### 3.2 Implementation Notes for the Agent

**Ellipsoid struct:**
```rust
#[derive(Clone, Copy, Debug)]
pub struct Ellipsoid {
    pub a: f64,          // Semi-major axis (metres)
    pub f: f64,          // Flattening  (dimensionless)
    // Derived (computed once):
    pub b: f64,          // Semi-minor axis: a * (1 - f)
    pub e: f64,          // First eccentricity: sqrt(2f - f²)
    pub e2: f64,         // e squared
    pub ep2: f64,        // Second eccentricity squared: e² / (1 - e²)
    pub n: f64,          // Third flattening: f / (2 - f)
}

pub const WGS84: Ellipsoid = Ellipsoid::new(6_378_137.0, 1.0 / 298.257223563);
pub const GRS80: Ellipsoid = Ellipsoid::new(6_378_137.0, 1.0 / 298.257222101);
```

**Projection trait:**
```rust
pub trait Projection: Send + Sync {
    /// Forward: (lon_rad, lat_rad) → (easting, northing)
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError>;
    
    /// Inverse: (easting, northing) → (lon_rad, lat_rad)
    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError>;
    
    /// Batch forward transform (default: loop, override for SIMD)
    fn forward_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        for c in coords.iter_mut() {
            *c = self.forward(c.0, c.1)?;
        }
        Ok(())
    }
    
    /// Batch inverse transform
    fn inverse_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        for c in coords.iter_mut() {
            *c = self.inverse(c.0, c.1)?;
        }
        Ok(())
    }
    
    fn ellipsoid(&self) -> &Ellipsoid;
}
```

**Transverse Mercator — the hardest one.** Use the Krüger n-series (not the older λ-series). The coefficients for 6th-order in `n`:

```rust
// α coefficients for forward transform (n-series, Karney 2011)
fn alpha_coefficients(n: f64) -> [f64; 6] {
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;
    let n5 = n4 * n;
    let n6 = n5 * n;
    [
        n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0
            - 127.0 * n5 / 288.0 + 7891.0 * n6 / 37800.0,
        13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0
            + 281.0 * n5 / 630.0 - 1983433.0 * n6 / 1935360.0,
        61.0 * n3 / 240.0 - 103.0 * n4 / 140.0 + 15061.0 * n5 / 26880.0
            + 167603.0 * n6 / 181440.0,
        49561.0 * n4 / 161280.0 - 179.0 * n5 / 168.0 + 6601661.0 * n6 / 7257600.0,
        34729.0 * n5 / 80640.0 - 3418889.0 * n6 / 1995840.0,
        212378941.0 * n6 / 319334400.0,
    ]
}
```

**Linear approximation (critical for performance):**
```rust
pub struct LinearApprox {
    // For each scanline, store 3 transformed points and interpolate
    // Error tolerance: 0.125 pixels (GDAL default)
    tolerance_px: f64,
}

impl LinearApprox {
    /// For a scanline of `width` pixels, compute exact transforms at
    /// start, middle, end, then linearly interpolate between them.
    /// If interpolation error exceeds tolerance, recursively subdivide.
    pub fn transform_scanline(
        &self,
        pipeline: &ProjectionPipeline,
        dst_affine: &Affine,
        src_affine_inv: &Affine,
        y: usize,
        width: usize,
        out_src_x: &mut [f64],
        out_src_y: &mut [f64],
    ) -> Result<(), ProjError>;
}
```

### 3.3 Acceptance Criteria

- All projections pass point-accuracy tests against pyproj/PROJ output.
- Maximum error < 1mm at the equator for Transverse Mercator at ±3° from central meridian.
- Maximum error < 0.1mm for Web Mercator (simple formulae).
- Batch transform of 1M points in < 20ms on Apple Silicon (M-series).
- `cargo test` passes with zero warnings under `clippy`.

### 3.4 Testing Strategy for This Phase

```rust
// In each projection module, include tests like:
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_tm_forward_inverse_roundtrip() {
        let tm = TransverseMercator::utm_zone(33, true); // UTM zone 33N
        let (lon, lat) = (15.0_f64.to_radians(), 52.0_f64.to_radians());
        let (e, n) = tm.forward(lon, lat).unwrap();
        let (lon2, lat2) = tm.inverse(e, n).unwrap();
        assert_relative_eq!(lon, lon2, epsilon = 1e-12);
        assert_relative_eq!(lat, lat2, epsilon = 1e-12);
    }
    
    #[test]
    fn test_tm_matches_proj() {
        // Pre-computed values from: echo "15 52" | cs2cs EPSG:4326 EPSG:32633
        let tm = TransverseMercator::utm_zone(33, true);
        let (e, n) = tm.forward(15.0_f64.to_radians(), 52.0_f64.to_radians()).unwrap();
        assert_relative_eq!(e, 500000.0, epsilon = 0.001); // Within 1mm
        assert_relative_eq!(n, 5760838.263, epsilon = 0.001);
    }
}
```

And Python-side tests comparing against pyproj:

```python
# tests/test_projections.py
import numpy as np
import pyproj
from rust-warp._rust import transform_points

def test_utm_matches_pyproj():
    """Verify Rust projection matches pyproj for UTM zone 33N."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    
    # Generate 10000 random points in the valid range
    rng = np.random.default_rng(42)
    lons = rng.uniform(9.0, 21.0, 10000)
    lats = rng.uniform(46.0, 58.0, 10000)
    
    # pyproj reference
    ex_pyproj, ny_pyproj = transformer.transform(lons, lats)
    
    # rust-warp Rust implementation
    ex_rust, ny_rust = transform_points(
        lons, lats,
        src_crs="EPSG:4326",
        dst_crs="EPSG:32633",
    )
    
    np.testing.assert_allclose(ex_rust, ex_pyproj, atol=0.001)  # 1mm
    np.testing.assert_allclose(ny_rust, ny_pyproj, atol=0.001)
```

---

## 4. Phase 2 — Rust Core: Warp Engine and Resampling Kernels

**Goal:** Implement the inverse-mapping warp loop and resampling kernels. Given source pixel data and a coordinate transformation, produce reprojected output data.

**Duration estimate:** 2-3 weeks

### 4.1 Deliverables

| File | Contents |
|---|---|
| `src/resample/nearest.rs` | Nearest neighbour (fastest path) |
| `src/resample/bilinear.rs` | 2×2 bilinear interpolation |
| `src/resample/cubic.rs` | 4×4 cubic convolution (Keys, 1981) |
| `src/resample/lanczos.rs` | 6×6 Lanczos (sinc windowed) |
| `src/resample/average.rs` | Area-weighted average for downsampling |
| `src/warp/engine.rs` | The main warp loop |
| `src/warp/nodata.rs` | Nodata/NaN handling during resampling |
| `src/error.rs` | Error enum for the whole crate |

### 4.2 The Warp Loop (Most Critical Code in the Project)

```rust
// src/warp/engine.rs — pseudocode for the core algorithm

/// Reproject a 2D source array into a destination array.
///
/// Algorithm (inverse mapping):
/// 1. For each output pixel (dst_y, dst_x):
///    a. Convert pixel coords to projected coords via dst_affine
///    b. Transform projected coords to source CRS via pipeline.inverse()
///       (using linear approximation for speed)
///    c. Convert source CRS coords to source pixel coords via src_affine_inv
///    d. Sample the source array at (src_px, src_py) using the resampling kernel
///    e. Write the result to the output array
pub fn warp_2d<T: Copy + NumCast + Default>(
    src: &ArrayView2<T>,
    src_affine: &Affine,
    dst_shape: (usize, usize),
    dst_affine: &Affine,
    pipeline: &ProjectionPipeline,
    resampling: ResamplingMethod,
    nodata: Option<T>,
) -> Result<Array2<T>, WarpError> {
    let (dst_h, dst_w) = dst_shape;
    let mut dst = Array2::from_elem(dst_shape, nodata.unwrap_or_default());
    
    let src_affine_inv = src_affine.inverse()?;
    let approx = LinearApprox::new(0.125); // 0.125 pixel tolerance
    
    let (src_h, src_w) = src.dim();
    
    // Pre-allocate scanline buffers for source pixel coordinates
    let mut src_x_buf = vec![0.0f64; dst_w];
    let mut src_y_buf = vec![0.0f64; dst_w];
    
    for dst_y in 0..dst_h {
        // ── Step 1: Compute source pixel coords for this scanline ──
        // Uses linear approximation: only 3 exact projections per scanline,
        // the rest are linearly interpolated
        approx.transform_scanline(
            pipeline, dst_affine, &src_affine_inv,
            dst_y, dst_w,
            &mut src_x_buf, &mut src_y_buf,
        )?;
        
        // ── Step 2: Sample source array for each output pixel ──
        for dst_x in 0..dst_w {
            let sx = src_x_buf[dst_x];
            let sy = src_y_buf[dst_x];
            
            // Bounds check (with kernel radius padding)
            let radius = resampling.kernel_radius();
            if sx < -radius || sy < -radius
                || sx >= (src_w as f64) + radius
                || sy >= (src_h as f64) + radius
            {
                // Outside source extent → nodata
                continue;
            }
            
            // Apply resampling kernel
            let value = match resampling {
                ResamplingMethod::Nearest => nearest::sample(src, sx, sy, nodata),
                ResamplingMethod::Bilinear => bilinear::sample(src, sx, sy, nodata),
                ResamplingMethod::Cubic => cubic::sample(src, sx, sy, nodata),
                ResamplingMethod::Lanczos => lanczos::sample(src, sx, sy, nodata),
            };
            
            if let Some(v) = value {
                dst[(dst_y, dst_x)] = v;
            }
        }
    }
    
    Ok(dst)
}
```

### 4.3 Resampling Kernel Implementations

**Bilinear (the most important after nearest):**

```rust
// src/resample/bilinear.rs

pub fn sample<T: Copy + NumCast>(
    src: &ArrayView2<T>,
    x: f64, y: f64,
    nodata: Option<T>,
) -> Option<T> {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x0 as f64;   // fractional x [0, 1)
    let fy = y - y0 as f64;   // fractional y [0, 1)
    
    let (h, w) = src.dim();
    
    // Gather 4 neighbours (with bounds checking)
    let get = |iy: isize, ix: isize| -> Option<f64> {
        if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
            return None;
        }
        let v = src[(iy as usize, ix as usize)];
        let fv: f64 = NumCast::from(v)?;
        // Check nodata
        if let Some(nd) = nodata {
            let fnd: f64 = NumCast::from(nd)?;
            if (fv - fnd).abs() < f64::EPSILON || fv.is_nan() {
                return None;
            }
        }
        Some(fv)
    };
    
    let v00 = get(y0, x0)?;
    let v10 = get(y0, x0 + 1)?;
    let v01 = get(y0 + 1, x0)?;
    let v11 = get(y0 + 1, x0 + 1)?;
    
    // Bilinear interpolation:
    // v = (1-fy) * [(1-fx)*v00 + fx*v10] + fy * [(1-fx)*v01 + fx*v11]
    let result = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10)
               + fy * ((1.0 - fx) * v01 + fx * v11);
    
    NumCast::from(result)
}
```

**Cubic convolution (Keys, 1981):**

The kernel weight function for parameter `a = -0.5`:
```
W(t) = (a+2)|t|³ - (a+3)|t|² + 1       for 0 ≤ |t| ≤ 1
W(t) = a|t|³ - 5a|t|² + 8a|t| - 4a     for 1 < |t| ≤ 2
W(t) = 0                                 for |t| > 2
```

Sample a 4×4 neighbourhood, weight each pixel by `W(dx) * W(dy)`.

### 4.4 Data Type Support

The warp engine must be generic over element type. Support at minimum:

| Rust type | numpy dtype | Notes |
|---|---|---|
| `u8` | uint8 | Satellite imagery (Landsat, Sentinel-2 L1C) |
| `u16` | uint16 | Sentinel-2 L2A, elevation models |
| `i16` | int16 | Temperature anomalies, signed elevation |
| `f32` | float32 | Most scientific data, SST, NDVI |
| `f64` | float64 | High-precision applications |

Use Rust generics with trait bounds: `T: Copy + NumCast + PartialOrd + Default + Send`.

### 4.5 Acceptance Criteria

- Nearest-neighbour warp of 1024×1024 float32 (UTM→4326) in < 5ms.
- Bilinear warp of same in < 15ms.
- Output matches GDAL `gdalwarp` pixel-for-pixel for nearest, and within 1 ULP for bilinear/cubic (allowing for floating-point ordering differences).
- NaN propagation works correctly: any NaN in the kernel neighbourhood produces NaN in the output.
- Nodata values are not interpolated (bilinear between nodata and valid should produce nodata, matching GDAL's behaviour).

---

## 5. Phase 3 — Python Bindings via PyO3

**Goal:** Expose the Rust warp engine to Python as numpy-array-in, numpy-array-out functions. No xarray or dask yet — just raw arrays.

**Duration estimate:** 1-2 weeks

### 5.1 Deliverables

| File | Contents |
|---|---|
| `src/lib.rs` | PyO3 module registration |
| `src/py/reproject.rs` | `reproject_array()` binding |
| `src/py/plan.rs` | `plan_reproject()` binding |
| `src/py/types.rs` | Python-visible enums and structs |
| `python/rust-warp/__init__.py` | Package init, imports from `_rust` |
| `python/rust-warp/_rust.pyi` | Type stubs |

### 5.2 Primary Python-Callable Function

```rust
// src/py/reproject.rs

use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;

/// Reproject a 2D numpy array between coordinate reference systems.
///
/// Args:
///     src: Source array (2D, float32 or float64)
///     src_crs: Source CRS as PROJ string or "EPSG:XXXX"
///     src_transform: Affine geotransform [a, b, c, d, e, f]
///         where: x_geo = a*col + b*row + c
///                y_geo = d*col + e*row + f
///     dst_crs: Destination CRS
///     dst_transform: Destination affine geotransform
///     dst_shape: Output array shape (height, width)
///     resampling: One of "nearest", "bilinear", "cubic", "lanczos"
///     nodata: Optional nodata value
///
/// Returns:
///     Reprojected 2D numpy array of same dtype as input
#[pyfunction]
#[pyo3(signature = (src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling="bilinear", nodata=None))]
fn reproject_array<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, f64>,
    src_crs: &str,
    src_transform: [f64; 6],
    dst_crs: &str,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
    resampling: &str,
    nodata: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Release the GIL for the entire computation!
    let src_array = src.as_array().to_owned();
    let result = py.allow_threads(|| {
        let src_view = src_array.view();
        let pipeline = ProjectionPipeline::new(src_crs, dst_crs)?;
        let src_aff = Affine::from_gdal(&src_transform);
        let dst_aff = Affine::from_gdal(&dst_transform);
        let method = ResamplingMethod::from_str(resampling)?;
        
        warp_2d(&src_view, &src_aff, dst_shape, &dst_aff, &pipeline, method, nodata)
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(result.into_pyarray(py))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reproject_array, m)?)?;
    m.add_function(wrap_pyfunction!(plan_reproject, m)?)?;
    m.add_function(wrap_pyfunction!(transform_points, m)?)?;
    Ok(())
}
```

### 5.3 The Type Stub File

```python
# python/rust-warp/_rust.pyi
import numpy as np
import numpy.typing as npt

def reproject_array(
    src: npt.NDArray[np.float64],
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    resampling: str = "bilinear",
    nodata: float | None = None,
) -> npt.NDArray[np.float64]: ...

def plan_reproject(
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    src_shape: tuple[int, int],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    src_chunks: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    dst_chunks: tuple[int, int] | None = None,
    resampling: str = "bilinear",
) -> list[dict]: ...

def transform_points(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    src_crs: str,
    dst_crs: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
```

### 5.4 Acceptance Criteria

- `from rust-warp._rust import reproject_array` works.
- Passing a numpy array in produces a numpy array out with correct shape/dtype.
- The GIL is released during computation (verify with threading test).
- Invalid CRS strings produce clear Python `ValueError`.
- Memory usage is bounded: no more than `2 × (src_size + dst_size)`.

---

## 6. Phase 4 — Chunk Planner and Dask Integration

**Goal:** Enable lazy, chunk-aware reprojection with dask. The user should be able to reproject a dask-backed xarray DataArray without materialising the full dataset.

**Duration estimate:** 2-3 weeks

### 6.1 Deliverables

| File | Contents |
|---|---|
| `src/chunk/planner.rs` | Compute source windows for each dest tile (Rust) |
| `src/chunk/halo.rs` | Kernel-radius padding calculation |
| `src/py/plan.rs` | `plan_reproject()` PyO3 binding |
| `python/rust-warp/geobox.py` | GeoBox class |
| `python/rust-warp/dask_graph.py` | Dask graph construction |
| `python/rust-warp/reproject.py` | High-level `reproject()` that dispatches numpy/dask |

### 6.2 The Chunk Planning Algorithm

For each destination tile (a rectangular region of the output grid):

1. **Walk the tile boundary** (all 4 edges, with `pts_per_side` sample points).
2. **Inverse-project each boundary point** to source CRS coordinates.
3. **Compute the bounding box** of these source-CRS points.
4. **Convert to source pixel coordinates** via inverse affine.
5. **Add halo padding** for the resampling kernel radius.
6. **Clip to source array bounds**.
7. **Record the mapping**: `(dst_slice, src_slice, local_dst_transform, local_src_transform)`.

```rust
// src/chunk/planner.rs

pub struct TilePlan {
    /// Slice in the destination array: (y_start, y_end, x_start, x_end)
    pub dst_slice: (usize, usize, usize, usize),
    /// Slice in the source array (with halo): (y_start, y_end, x_start, x_end)
    pub src_slice: (usize, usize, usize, usize),
    /// Affine transform for the source slice (adjusted for the offset)
    pub src_transform: [f64; 6],
    /// Affine transform for the destination tile (adjusted for the offset)
    pub dst_transform: [f64; 6],
    /// Shape of the destination tile
    pub dst_tile_shape: (usize, usize),
    /// Whether this tile has any valid source coverage
    pub has_data: bool,
}

pub fn plan_tiles(
    src_crs: &str,
    src_transform: &Affine,
    src_shape: (usize, usize),
    dst_crs: &str,
    dst_transform: &Affine,
    dst_shape: (usize, usize),
    dst_tile_size: (usize, usize),   // e.g. (512, 512)
    kernel_radius: usize,            // e.g. 1 for bilinear, 2 for cubic
    pts_per_edge: usize,             // e.g. 21
) -> Result<Vec<TilePlan>, PlanError> {
    // ... implementation
}
```

### 6.3 Dask Graph Builder (Python Side)

```python
# python/rust-warp/dask_graph.py

import dask.array as da
import numpy as np
from rust-warp._rust import reproject_array, plan_reproject

def _reproject_tile(src_block, src_crs, src_transform, dst_crs, dst_transform,
                    dst_shape, resampling, nodata):
    """Worker function called per dask task."""
    return reproject_array(
        np.ascontiguousarray(src_block),
        src_crs, tuple(src_transform),
        dst_crs, tuple(dst_transform),
        dst_shape, resampling, nodata,
    )

def reproject_dask(src_data, src_geobox, dst_geobox, resampling='bilinear',
                   dst_chunks=(512, 512), nodata=None):
    """Build a dask graph for lazy reprojection.
    
    Parameters
    ----------
    src_data : dask.array.Array
        2D or 3D dask array (bands, y, x) or (y, x)
    src_geobox : GeoBox
        Source grid definition
    dst_geobox : GeoBox
        Destination grid definition
    resampling : str
        Resampling method
    dst_chunks : tuple
        Chunk size for the output array (y_chunk, x_chunk)
    nodata : float, optional
        Nodata value
    
    Returns
    -------
    dask.array.Array
        Lazy reprojected array
    """
    tiles = plan_reproject(
        src_crs=str(src_geobox.crs),
        src_transform=tuple(src_geobox.affine),
        src_shape=src_geobox.shape,
        dst_crs=str(dst_geobox.crs),
        dst_transform=tuple(dst_geobox.affine),
        dst_shape=dst_geobox.shape,
        dst_chunks=dst_chunks,
        resampling=resampling,
    )
    
    # Build output array from delayed tiles
    tile_rows = []
    y_pos = 0
    
    while y_pos < dst_geobox.shape[0]:
        tile_cols = []
        x_pos = 0
        tile_h = min(dst_chunks[0], dst_geobox.shape[0] - y_pos)
        
        while x_pos < dst_geobox.shape[1]:
            tile_w = min(dst_chunks[1], dst_geobox.shape[1] - x_pos)
            
            # Find the matching tile plan
            tile = _find_tile(tiles, y_pos, x_pos)
            
            if tile is None or not tile['has_data']:
                # No source data covers this tile
                block = da.full((tile_h, tile_w), nodata or np.nan,
                              dtype=src_data.dtype, chunks=(tile_h, tile_w))
            else:
                # Extract the source region this tile needs
                sy0, sy1, sx0, sx1 = tile['src_slice']
                src_block = src_data[sy0:sy1, sx0:sx1]
                
                block = da.from_delayed(
                    dask.delayed(_reproject_tile)(
                        src_block,
                        str(src_geobox.crs),
                        tile['src_transform'],
                        str(dst_geobox.crs),
                        tile['dst_transform'],
                        (tile_h, tile_w),
                        resampling,
                        nodata,
                    ),
                    shape=(tile_h, tile_w),
                    dtype=src_data.dtype,
                )
            
            tile_cols.append(block)
            x_pos += tile_w
        
        tile_rows.append(da.concatenate(tile_cols, axis=1))
        y_pos += tile_h
    
    return da.concatenate(tile_rows, axis=0)
```

### 6.4 Acceptance Criteria

- Reprojecting a dask array does NOT materialise the source array.
- Each dask task reads only the source region it needs (verified with dask graph inspection).
- Output of chunked reprojection is pixel-identical to non-chunked reprojection.
- Works with any combination of source/destination chunk sizes.
- Memory usage per task is bounded by `src_tile_size + dst_tile_size + overhead`.

---

## 7. Phase 5 — Xarray Accessor and High-Level API

**Goal:** Provide a clean, discoverable API for xarray users.

**Duration estimate:** 1-2 weeks

### 7.1 Deliverables

| File | Contents |
|---|---|
| `python/rust-warp/xarray_accessor.py` | `.rust-warp` accessor on DataArray and Dataset |
| `python/rust-warp/geobox.py` | GeoBox with `from_xarray()` and `from_bbox()` |
| `python/rust-warp/__init__.py` | Clean public API |

### 7.2 User-Facing API

```python
import xarray as xr
import rust-warp

# ── Option 1: Accessor ──
ds = xr.open_dataset("sentinel2.zarr", engine="zarr", chunks={})
reprojected = ds.rust-warp.reproject("EPSG:4326", resolution=0.0001)

# ── Option 2: Functional ──
from rust-warp import reproject, GeoBox

dst_geobox = GeoBox.from_bbox(
    bbox=(-10, 50, 2, 60),
    crs="EPSG:4326",
    resolution=0.001,
)
reprojected = reproject(ds, dst_geobox, resampling="bilinear")

# ── Option 3: Match another dataset's grid ──
reprojected = ds.rust-warp.reproject_match(reference_ds)
```

### 7.3 GeoBox Class

```python
# python/rust-warp/geobox.py

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GeoBox:
    """A geo-registered pixel grid: CRS + affine transform + shape."""
    
    crs: str                        # "EPSG:4326" or PROJ string
    shape: tuple[int, int]          # (height, width)
    affine: tuple[float, ...]       # 6-element GDAL geotransform
    
    @classmethod
    def from_bbox(cls, bbox, crs, resolution, shape=None):
        """Create from bounding box and resolution."""
        ...
    
    @classmethod
    def from_xarray(cls, da):
        """Extract from xarray DataArray with CRS metadata."""
        # Supports rioxarray-style and odc-geo-style metadata
        ...
    
    @property 
    def bounds(self):
        """(left, bottom, right, top) in CRS units."""
        ...
    
    def xr_coords(self):
        """Generate xarray-compatible x/y coordinate arrays."""
        ...
```

### 7.4 Accessor Implementation

```python
# python/rust-warp/xarray_accessor.py

import xarray as xr
from .geobox import GeoBox
from .reproject import reproject

@xr.register_dataarray_accessor("rust-warp")
class rust-warpAccessor:
    def __init__(self, da):
        self._da = da
    
    @property
    def geobox(self):
        return GeoBox.from_xarray(self._da)
    
    def reproject(self, dst_crs, resolution=None, shape=None,
                  resampling='bilinear', nodata=None, dst_chunks=None):
        """Reproject this DataArray to a new CRS."""
        src_geobox = self.geobox
        dst_geobox = self._compute_dst_geobox(dst_crs, resolution, shape)
        return reproject(self._da, dst_geobox, resampling=resampling,
                        nodata=nodata, dst_chunks=dst_chunks)
    
    def reproject_match(self, other, resampling='bilinear', nodata=None):
        """Reproject to match another DataArray's grid."""
        dst_geobox = GeoBox.from_xarray(other)
        return reproject(self._da, dst_geobox, resampling=resampling,
                        nodata=nodata)
```

---

## 8. Phase 6 — SIMD Optimisation and Parallelism

**Goal:** Squeeze maximum performance from the hot paths.

**Duration estimate:** 2-3 weeks

### 8.1 SIMD Strategy

Target two architectures:

| Arch | ISA | Registers | Floats/register |
|---|---|---|---|
| Apple Silicon | NEON | 32 × 128-bit | 4 × f32 or 2 × f64 |
| x86_64 (CI/servers) | AVX2 | 16 × 256-bit | 8 × f32 or 4 × f64 |

**What to SIMD-ise:**

1. **Bilinear kernel** — process 4 (NEON) or 8 (AVX2) output pixels simultaneously. Each pixel needs 4 source lookups and a weighted sum — this vectorises perfectly.
2. **Projection math** — batch `sin`/`cos`/`atan` for multiple points. Use polynomial approximations (Chebyshev or minimax) that SIMD-ise well.
3. **Linear interpolation of source coordinates** — the approximation step is pure linear interp of scanline coordinates, a textbook SIMD workload.

**Implementation approach:** Use Rust's portable SIMD (`std::simd`, stabilised in nightly, or use `packed_simd2` / `wide` crate for stable Rust). Alternatively, write architecture-specific code behind `#[cfg]` gates.

### 8.2 Rayon Parallelism

For single large chunks, use Rayon to parallelise across scanlines:

```rust
use rayon::prelude::*;

// Parallel warp: each scanline is independent
(0..dst_h).into_par_iter().for_each(|dst_y| {
    let mut src_x = vec![0.0; dst_w];
    let mut src_y = vec![0.0; dst_w];
    approx.transform_scanline(pipeline, dst_affine, &src_affine_inv,
                               dst_y, dst_w, &mut src_x, &mut src_y);
    // ... sample and write to output row
});
```

This gives intra-chunk parallelism complementary to dask's inter-chunk parallelism.

---

## 9. Phase 7 — Production Hardening

**Goal:** Handle edge cases, improve CRS coverage, prepare for release.

**Duration estimate:** 3-4 weeks

### 9.1 Edge Cases to Handle

| Edge Case | Strategy |
|---|---|
| Antimeridian crossing (±180°) | Detect via boundary check, split tile into two halves |
| Polar singularity | Clamp latitude to ±89.999° for Mercator-family |
| Source completely outside dest extent | `has_data: false` in TilePlan, skip tile |
| NaN in source affecting kernel | Propagate NaN if any kernel input is NaN (configurable) |
| Integer overflow on large rasters | Use `usize` everywhere, test with 100k×100k arrays |
| Non-north-up (rotated) affines | Full 6-parameter affine support, not just axis-aligned |
| 3D arrays (band, y, x) | Loop over leading dimensions in the Python layer |
| Dataset (multi-variable) | Loop over data variables in the accessor |

### 9.2 Extended CRS Support

- Parse `EPSG:XXXX` codes by maintaining a lookup table of projection parameters for the ~200 most common EPSGs.
- Fall back to `pyproj` for CRS parsing when the Rust side doesn't recognise the code (call Python from Rust via PyO3's `Python::import`).
- Support PROJ strings directly (`+proj=utm +zone=33 +datum=WGS84`).

### 9.3 CI / Release Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  rust-tests:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, macos-14]  # macos-14 = ARM
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all
      - run: cargo clippy -- -D warnings

  python-tests:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras
      - run: uv run pytest tests/ -x -v

  wheels:
    if: startsWith(github.ref, 'refs/tags/')
    uses: PyO3/maturin-action@v1
    with:
      command: build
      args: --release --out dist
```

---

## 10. Testing Plan

### 10.1 Test Hierarchy

```
Level 1: Unit Tests (Rust — cargo test)
├── Projection forward/inverse roundtrip for each projection
├── Projection accuracy against precomputed PROJ reference values
├── Affine transform operations
├── Each resampling kernel against known synthetic inputs
├── Linear approximation accuracy vs exact
└── Chunk planner slice computation

Level 2: Integration Tests (Python — pytest)
├── reproject_array() with known inputs → expected outputs
├── Data type round-trips (uint8, uint16, float32, float64)
├── Nodata handling (NaN, specific value, boundary pixels)
├── CRS parsing (EPSG codes, PROJ strings)
└── GIL release verification (concurrent threads)

Level 3: Correctness Tests (Python — pytest, comparison)
├── Pixel-by-pixel comparison vs GDAL (gdalwarp) for:
│   ├── Nearest neighbour: exact match expected
│   ├── Bilinear: max 1 ULP difference
│   ├── Cubic: max 2 ULP difference
│   └── Lanczos: max 2 ULP difference
├── Comparison vs odc-geo xr_reproject for real Sentinel-2 tiles
├── Comparison vs rioxarray rio.reproject for real Landsat tiles
└── Comparison vs pyresample for swath-like data

Level 4: Dask/Chunked Tests (Python — pytest)
├── Chunked output == non-chunked output (bit-identical)
├── Different chunk sizes produce same output
├── Memory usage stays bounded (tracemalloc)
├── Source array not materialised (mock/spy on .compute())
├── Works with dask distributed scheduler
└── Works with dask threaded scheduler (GIL-free verification)

Level 5: Performance Tests (Python — pytest-benchmark)
├── Projection throughput (points/second)
├── Per-chunk warp latency (various sizes and methods)
├── End-to-end reprojection (various dataset sizes)
├── Multi-thread scaling (1, 2, 4, 8 threads)
└── Memory high-water-mark
```

### 10.2 Test Data

```python
# scripts/generate_test_data.py

"""Generate synthetic test datasets for the rust-warp test suite.

Creates:
  tests/data/synthetic_utm33n_256.npy     - 256×256 float32, UTM zone 33N
  tests/data/synthetic_utm33n_1024.npy    - 1024×1024 float32, UTM zone 33N
  tests/data/synthetic_utm33n_4096.npy    - 4096×4096 float32, UTM zone 33N
  tests/data/synthetic_4326_256.npy       - 256×256 float32, EPSG:4326
  tests/data/synthetic_webmerc_1024.npy   - 1024×1024 float32, EPSG:3857
  tests/data/synthetic_checkerboard.npy   - Checkerboard pattern for visual QA
  tests/data/synthetic_with_nodata.npy    - Array with nodata holes
  tests/data/sentinel2_tile_*.zarr        - Real S2 subsets (downloaded once)
  tests/data/reference_*.npy              - GDAL-produced reference outputs
"""
```

### 10.3 Correctness Comparison Fixture

```python
# tests/conftest.py

import numpy as np
import pytest
from osgeo import gdal
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject as rio_reproject, Resampling

@pytest.fixture
def gdal_reference():
    """Produce a GDAL-reprojected reference for comparison."""
    def _reproject(src_array, src_crs, src_transform, dst_crs, dst_transform,
                   dst_shape, resampling='bilinear'):
        resampling_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'lanczos': Resampling.lanczos,
        }
        dst = np.empty(dst_shape, dtype=src_array.dtype)
        rio_reproject(
            src_array, dst,
            src_crs=src_crs, src_transform=rasterio.transform.Affine(*src_transform[:6]),
            dst_crs=dst_crs, dst_transform=rasterio.transform.Affine(*dst_transform[:6]),
            resampling=resampling_map[resampling],
        )
        return dst
    return _reproject

@pytest.fixture
def odc_reference():
    """Produce an odc-geo-reprojected reference for comparison."""
    def _reproject(xr_da, dst_crs, resolution):
        from odc.geo.xr import xr_reproject
        from odc.geo.geobox import GeoBox
        gbox = GeoBox.from_bbox(xr_da.odc.geobox.boundingbox, dst_crs, resolution)
        return xr_reproject(xr_da, gbox)
    return _reproject
```

### 10.4 Example Correctness Test

```python
# tests/test_warp.py

import numpy as np
import pytest
from rust-warp._rust import reproject_array

class TestWarpCorrectness:
    """Compare rust-warp output pixel-by-pixel against GDAL."""
    
    @pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic"])
    @pytest.mark.parametrize("src_crs,dst_crs", [
        ("EPSG:32633", "EPSG:4326"),   # UTM 33N → WGS84
        ("EPSG:4326", "EPSG:3857"),    # WGS84 → Web Mercator
        ("EPSG:32633", "EPSG:3857"),   # UTM → Web Mercator
        ("EPSG:3857", "EPSG:32633"),   # Web Mercator → UTM
    ])
    def test_matches_gdal(self, gdal_reference, resampling, src_crs, dst_crs):
        rng = np.random.default_rng(42)
        src = rng.standard_normal((256, 256)).astype(np.float64)
        
        # Define transforms (these would come from real metadata)
        src_transform = _make_transform(src_crs, src.shape)
        dst_shape, dst_transform = _compute_dst_grid(
            src.shape, src_transform, src_crs, dst_crs
        )
        
        # GDAL reference
        expected = gdal_reference(
            src, src_crs, src_transform, dst_crs, dst_transform,
            dst_shape, resampling
        )
        
        # rust-warp
        actual = reproject_array(
            src, src_crs, src_transform, dst_crs, dst_transform,
            dst_shape, resampling
        )
        
        # Compare
        if resampling == 'nearest':
            # Nearest should be exact (same pixel selection)
            valid = ~np.isnan(expected) & ~np.isnan(actual)
            np.testing.assert_array_equal(actual[valid], expected[valid])
        else:
            # Interpolating methods: allow tiny floating-point differences
            valid = ~np.isnan(expected) & ~np.isnan(actual)
            np.testing.assert_allclose(
                actual[valid], expected[valid],
                rtol=1e-6, atol=1e-10,
            )
```

---

## 11. Benchmark Plan

### 11.1 Benchmark Matrix

We measure **three libraries** across **four axes**:

| Axis | Values |
|---|---|
| **Dataset size** | 256², 1024², 4096², 10000² |
| **Resampling method** | nearest, bilinear, cubic |
| **CRS pair** | UTM→4326, 4326→3857, UTM→3857 |
| **Chunking** | non-chunked, 256² chunks, 512² chunks, 1024² chunks |

Libraries under test:

| Library | How invoked |
|---|---|
| **rust-warp** (this project) | `reproject_array()` and `reproject()` with dask |
| **GDAL direct** (via rasterio) | `rasterio.warp.reproject()` |
| **odc-geo** | `xr_reproject()` with dask |
| **rioxarray** | `da.rio.reproject()` (non-dask only) |

### 11.2 Benchmark Runner

```python
# tests/benchmarks/bench_e2e.py

import numpy as np
import pytest
import time

SIZES = [256, 1024, 4096]
RESAMPLINGS = ['nearest', 'bilinear', 'cubic']
CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),
    ("EPSG:4326", "EPSG:3857"),
]

@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("resampling", RESAMPLINGS)
@pytest.mark.parametrize("src_crs,dst_crs", CRS_PAIRS)
class TestBenchmarkEndToEnd:
    
    def test_rust-warp(self, benchmark, size, resampling, src_crs, dst_crs):
        from rust-warp._rust import reproject_array
        src, src_tf, dst_tf, dst_shape = _setup(size, src_crs, dst_crs)
        
        benchmark(reproject_array,
                  src, src_crs, src_tf, dst_crs, dst_tf, dst_shape, resampling)
    
    def test_rasterio(self, benchmark, size, resampling, src_crs, dst_crs):
        import rasterio
        from rasterio.warp import reproject as rio_reproject, Resampling
        src, src_tf, dst_tf, dst_shape = _setup(size, src_crs, dst_crs)
        dst = np.empty(dst_shape, dtype=np.float64)
        resamp = getattr(Resampling, resampling)
        
        def run():
            rio_reproject(src, dst, src_crs=src_crs,
                         src_transform=rasterio.transform.Affine(*src_tf),
                         dst_crs=dst_crs,
                         dst_transform=rasterio.transform.Affine(*dst_tf),
                         resampling=resamp)
        benchmark(run)
    
    def test_odc_geo(self, benchmark, size, resampling, src_crs, dst_crs):
        from odc.geo.xr import xr_reproject
        da = _make_xarray_da(size, src_crs)
        dst_geobox = _make_odc_geobox(size, dst_crs)
        
        benchmark(xr_reproject, da, dst_geobox, resampling=resampling)
```

### 11.3 Threading Benchmark (Critical for GIL-Free Claim)

```python
# tests/benchmarks/bench_parallel.py

import concurrent.futures
import time
import numpy as np

def test_parallel_scaling():
    """Verify that rust-warp scales with threads (unlike GDAL)."""
    from rust-warp._rust import reproject_array
    
    src = np.random.randn(1024, 1024).astype(np.float64)
    src_crs, dst_crs = "EPSG:32633", "EPSG:4326"
    src_tf, dst_tf, dst_shape = _setup_transforms(1024, src_crs, dst_crs)
    
    def single_reproject(_):
        return reproject_array(src, src_crs, src_tf, dst_crs, dst_tf,
                              dst_shape, 'bilinear')
    
    results = {}
    for n_threads in [1, 2, 4, 8]:
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(single_reproject, range(n_threads * 4)))
        elapsed = time.perf_counter() - start
        results[n_threads] = elapsed
        print(f"  {n_threads} threads: {elapsed:.3f}s")
    
    # Verify near-linear scaling (at least 2x speedup with 4 threads)
    assert results[4] < results[1] * 0.6, (
        f"Insufficient parallel scaling: 1 thread={results[1]:.2f}s, "
        f"4 threads={results[4]:.2f}s"
    )
```

### 11.4 Memory Benchmark

```python
# tests/benchmarks/bench_memory.py

import tracemalloc
import numpy as np

def test_memory_bounded():
    """Verify memory usage doesn't exceed 2x (src + dst)."""
    from rust-warp._rust import reproject_array
    
    size = 4096
    src = np.random.randn(size, size).astype(np.float32)
    expected_bytes = src.nbytes * 3  # src + dst + overhead
    
    tracemalloc.start()
    result = reproject_array(src, "EPSG:32633", ..., "EPSG:4326", ..., 
                            (size, size), 'bilinear')
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < expected_bytes, (
        f"Peak memory {peak/1e6:.1f} MB exceeds limit {expected_bytes/1e6:.1f} MB"
    )
```

---

## Appendix A — Key Mathematical References

| Resource | What it provides |
|---|---|
| Snyder, J.P. (1987) "Map Projections: A Working Manual" USGS PP 1395 | The definitive reference for all forward/inverse formulae. Free PDF from USGS. |
| Karney, C.F.F. (2011) "Transverse Mercator with an accuracy of a few nanometers" | The modern Krüger n-series to 8th order. Use this for TM implementation. |
| IOGP Guidance Note 7-2 (2019) | EPSG projection method formulae. Canonical parameterisation for all EPSG codes. |
| GDAL source: `alg/gdalwarpkernel.cpp` | Reference implementation of resampling kernels. Check edge cases here. |
| PROJ source: `src/projections/` | Reference projection implementations. Gold standard for accuracy testing. |
| Keys, R.G. (1981) "Cubic convolution interpolation" IEEE Trans. ASSP | The cubic convolution kernel definition. |

---

## Appendix B — File-by-File Manifest

This is the complete list of files to create, in the order they should be created:

### Phase 1 Files
```
src/error.rs
src/affine.rs
src/proj/mod.rs
src/proj/ellipsoid.rs
src/proj/common.rs
src/proj/transverse_mercator.rs
src/proj/mercator.rs
src/proj/lambert_conformal.rs
src/proj/albers_equal_area.rs
src/proj/stereographic.rs
src/proj/sinusoidal.rs
src/proj/equirectangular.rs
src/proj/pipeline.rs
src/proj/approx.rs
tests/test_projections.py
```

### Phase 2 Files
```
src/resample/mod.rs
src/resample/nearest.rs
src/resample/bilinear.rs
src/resample/cubic.rs
src/resample/lanczos.rs
src/resample/average.rs
src/warp/mod.rs
src/warp/engine.rs
src/warp/nodata.rs
benches/warp_bench.rs
tests/test_resampling.py
tests/test_warp.py
```

### Phase 3 Files
```
src/lib.rs
src/py/mod.rs
src/py/reproject.rs
src/py/plan.rs
src/py/types.rs
python/rust-warp/__init__.py
python/rust-warp/_rust.pyi
```

### Phase 4 Files
```
src/chunk/mod.rs
src/chunk/planner.rs
src/chunk/halo.rs
python/rust-warp/geobox.py
python/rust-warp/dask_graph.py
python/rust-warp/reproject.py
tests/test_dask.py
```

### Phase 5 Files
```
python/rust-warp/xarray_accessor.py
tests/test_xarray.py
```

### Phase 6 Files
```
src/warp/simd.rs
(modifications to existing warp/engine.rs and resample/*.rs)
tests/benchmarks/bench_projections.py
tests/benchmarks/bench_warp.py
tests/benchmarks/bench_e2e.py
tests/benchmarks/bench_parallel.py
tests/benchmarks/bench_memory.py
```

### Phase 7 Files
```
.github/workflows/ci.yml
.github/workflows/release.yml
scripts/generate_test_data.py
scripts/run_benchmarks.py
(modifications across all modules for edge cases)
```

---

## Quick-Start for an Agent

If you are an AI agent receiving this plan to implement a specific phase:

1. **Read the phase section thoroughly** — each has its own deliverables, implementation notes, and acceptance criteria.
2. **Create the project skeleton first** (Section 2) if it doesn't exist yet.
3. **Run `uv sync --all-extras`** to ensure the environment works before writing code.
4. **Write Rust tests alongside the Rust code** — every `.rs` file should have a `#[cfg(test)] mod tests` block.
5. **Run `cargo test` after every file** — do not proceed to the next file if tests fail.
6. **Run `cargo clippy -- -D warnings`** — zero warnings policy.
7. **After completing the Rust changes, run `uv run maturin develop`** to rebuild the Python extension.
8. **Then run the Python tests** with `uv run pytest tests/ -x -v`.
9. **Do not skip the comparison tests** — the whole point is matching GDAL's output.
