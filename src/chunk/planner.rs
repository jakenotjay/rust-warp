//! Chunk planner: maps destination tiles to source ROIs for chunked reprojection.

use rayon::prelude::*;

use crate::affine::Affine;
use crate::error::PlanError;
use crate::proj::pipeline::Pipeline;

/// A plan for reprojecting a single destination tile.
#[derive(Clone, Debug)]
pub struct TilePlan {
    /// Destination tile bounds (row_start, row_end, col_start, col_end), end-exclusive.
    pub dst_slice: (usize, usize, usize, usize),
    /// Source ROI with halo, clipped to source bounds (row_start, row_end, col_start, col_end).
    pub src_slice: (usize, usize, usize, usize),
    /// Affine transform with origin shifted to src_slice start.
    pub src_transform: Affine,
    /// Affine transform with origin shifted to dst_slice start.
    pub dst_transform: Affine,
    /// Shape of this destination tile (rows, cols).
    pub dst_tile_shape: (usize, usize),
    /// Whether the source ROI has valid coverage.
    pub has_data: bool,
}

/// Generate sample points along the boundary of a rectangular tile.
///
/// Returns (col, row) pairs at pixel centers along all 4 edges.
fn tile_boundary_points(
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
    pts_per_edge: usize,
) -> Vec<(f64, f64)> {
    let pts = pts_per_edge.max(2);
    let mut points = Vec::with_capacity(pts * 4);

    let r0 = row0 as f64 + 0.5;
    let r1 = (row1 as f64) - 0.5;
    let c0 = col0 as f64 + 0.5;
    let c1 = (col1 as f64) - 0.5;

    // Handle degenerate tiles (1 pixel wide/tall)
    let row_step = if pts > 1 && r1 > r0 {
        (r1 - r0) / (pts - 1) as f64
    } else {
        0.0
    };
    let col_step = if pts > 1 && c1 > c0 {
        (c1 - c0) / (pts - 1) as f64
    } else {
        0.0
    };

    // Top edge: row=r0, col varies
    for i in 0..pts {
        let c = c0 + col_step * i as f64;
        points.push((c.min(c1), r0));
    }
    // Bottom edge: row=r1, col varies
    for i in 0..pts {
        let c = c0 + col_step * i as f64;
        points.push((c.min(c1), r1.max(r0)));
    }
    // Left edge: col=c0, row varies (skip corners already covered)
    for i in 1..pts.saturating_sub(1) {
        let r = r0 + row_step * i as f64;
        points.push((c0, r));
    }
    // Right edge: col=c1, row varies (skip corners already covered)
    for i in 1..pts.saturating_sub(1) {
        let r = r0 + row_step * i as f64;
        points.push((c1.max(c0), r));
    }

    points
}

/// Project the source image boundary into dst pixel space.
///
/// Used as a cheap pre-filter: any destination tile whose pixel bbox lies entirely
/// outside this AABB can be immediately marked `has_data = false` without firing
/// per-tile projection calls.
///
/// Returns `Some((min_row, max_row, min_col, max_col))` in dst pixel coordinates,
/// or `None` if the source has no coverage within the destination image bounds
/// (including the case where the projection fails for all boundary points).
fn src_footprint_in_dst_pixels(
    src_crs: &str,
    src_transform: &Affine,
    src_shape: (usize, usize),
    dst_crs: &str,
    dst_transform: &Affine,
    dst_shape: (usize, usize),
    pts_per_edge: usize,
) -> Option<(f64, f64, f64, f64)> {
    // Forward pipeline: src CRS → dst CRS.
    // Swapping args so transform_inv(src_x, src_y) → (dst_x, dst_y).
    let fwd = Pipeline::new(dst_crs, src_crs).ok()?;
    let dst_inv = dst_transform.inverse().ok()?;

    let boundary = tile_boundary_points(0, src_shape.0, 0, src_shape.1, pts_per_edge);

    let mut min_col = f64::INFINITY;
    let mut max_col = f64::NEG_INFINITY;
    let mut min_row = f64::INFINITY;
    let mut max_row = f64::NEG_INFINITY;
    let mut count = 0usize;

    for &(sc, sr) in &boundary {
        // src pixel → src CRS
        let (sx, sy) = src_transform.forward(sc, sr);
        // src CRS → dst CRS
        if let Ok((dx, dy)) = fwd.transform_inv(sx, sy) {
            // dst CRS → dst pixel
            let (dc, dr) = dst_inv.forward(dx, dy);
            if dc.is_finite() && dr.is_finite() {
                min_col = min_col.min(dc);
                max_col = max_col.max(dc);
                min_row = min_row.min(dr);
                max_row = max_row.max(dr);
                count += 1;
            }
        }
    }

    if count == 0 {
        return None;
    }

    // Return None if the AABB lies entirely outside the destination image bounds.
    // This avoids a spurious Some when the source projects to valid dst CRS coords
    // but those coords are far outside the actual destination image.
    let (dst_rows, dst_cols) = dst_shape;
    if max_row < 0.0
        || min_row > dst_rows as f64
        || max_col < 0.0
        || min_col > dst_cols as f64
    {
        return None;
    }

    Some((min_row, max_row, min_col, max_col))
}

/// Compute the plan for a single destination tile.
///
/// Extracted from the `plan_tiles` loop so it can be called from `par_iter`.
/// All arguments are either `Copy` or immutable references to `Send + Sync` types.
#[allow(clippy::too_many_arguments)]
fn plan_single_tile(
    row0: usize,
    col0: usize,
    tile_h: usize,
    tile_w: usize,
    dst_rows: usize,
    dst_cols: usize,
    pipeline: &Pipeline,
    src_transform: &Affine,
    src_inv: &Affine,
    dst_transform: &Affine,
    src_shape: (usize, usize),
    kernel_radius: usize,
    pts_per_edge: usize,
    footprint: Option<(f64, f64, f64, f64)>,
) -> Result<TilePlan, PlanError> {
    let (src_rows, src_cols) = src_shape;
    let row1 = (row0 + tile_h).min(dst_rows);
    let col1 = (col0 + tile_w).min(dst_cols);

    // Shifted dst affine for this tile (origin at tile top-left pixel).
    let (dst_ox, dst_oy) = dst_transform.forward(col0 as f64, row0 as f64);
    let dst_tile_transform = Affine::new(
        dst_transform.a,
        dst_transform.b,
        dst_ox,
        dst_transform.d,
        dst_transform.e,
        dst_oy,
    );
    let dst_tile_shape = (row1 - row0, col1 - col0);

    // Fast rejection: tile bbox is entirely outside the source footprint AABB.
    // Expand the footprint by the kernel halo so we never reject a tile that
    // is only needed for halo pixels of an adjacent in-bounds tile.
    let halo = kernel_radius as f64;
    if let Some((fp_rmin, fp_rmax, fp_cmin, fp_cmax)) = footprint {
        if (row1 as f64) <= fp_rmin - halo
            || (row0 as f64) >= fp_rmax + halo
            || (col1 as f64) <= fp_cmin - halo
            || (col0 as f64) >= fp_cmax + halo
        {
            let src_tile_transform = Affine::new(
                src_transform.a,
                src_transform.b,
                src_transform.c,
                src_transform.d,
                src_transform.e,
                src_transform.f,
            );
            return Ok(TilePlan {
                dst_slice: (row0, row1, col0, col1),
                src_slice: (0, 0, 0, 0),
                src_transform: src_tile_transform,
                dst_transform: dst_tile_transform,
                dst_tile_shape,
                has_data: false,
            });
        }
    }

    // Sample boundary points of this destination tile and project dst → src.
    let boundary = tile_boundary_points(row0, row1, col0, col1, pts_per_edge);

    let mut min_src_col = f64::INFINITY;
    let mut max_src_col = f64::NEG_INFINITY;
    let mut min_src_row = f64::INFINITY;
    let mut max_src_row = f64::NEG_INFINITY;
    let mut valid_count = 0usize;

    for &(col_px, row_px) in &boundary {
        // Pixel coords → dst CRS coords
        let (dx, dy) = dst_transform.forward(col_px, row_px);

        // dst CRS → src CRS
        if let Ok((sx, sy)) = pipeline.transform_inv(dx, dy) {
            // src CRS → src pixel coords
            let (sc, sr) = src_inv.forward(sx, sy);

            if sc.is_finite() && sr.is_finite() {
                min_src_col = min_src_col.min(sc);
                max_src_col = max_src_col.max(sc);
                min_src_row = min_src_row.min(sr);
                max_src_row = max_src_row.max(sr);
                valid_count += 1;
            }
        }
    }

    let has_data;
    let src_slice;

    if valid_count == 0 {
        // No valid projections — tile is fully outside source extent.
        has_data = false;
        src_slice = (0, 0, 0, 0);
    } else {
        // Expand by kernel radius (halo) and clip to source bounds.
        let sr0 = (min_src_row - halo).floor().max(0.0) as usize;
        let sr1 = ((max_src_row + halo).ceil() as usize + 1).min(src_rows);
        let sc0 = (min_src_col - halo).floor().max(0.0) as usize;
        let sc1 = ((max_src_col + halo).ceil() as usize + 1).min(src_cols);

        has_data = sr1 > sr0 && sc1 > sc0;
        src_slice = (sr0, sr1, sc0, sc1);
    }

    let (src_sr0, _, src_sc0, _) = src_slice;
    let (src_ox, src_oy) = src_transform.forward(src_sc0 as f64, src_sr0 as f64);
    let src_tile_transform = Affine::new(
        src_transform.a,
        src_transform.b,
        src_ox,
        src_transform.d,
        src_transform.e,
        src_oy,
    );

    Ok(TilePlan {
        dst_slice: (row0, row1, col0, col1),
        src_slice,
        src_transform: src_tile_transform,
        dst_transform: dst_tile_transform,
        dst_tile_shape,
        has_data,
    })
}

/// Sequential reference implementation of plan_tiles — no Rayon, no footprint pre-filter.
///
/// Used only in benchmarks to establish a before/after baseline against `plan_tiles`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn plan_tiles_sequential(
    src_crs: &str,
    src_transform: &Affine,
    src_shape: (usize, usize),
    dst_crs: &str,
    dst_transform: &Affine,
    dst_shape: (usize, usize),
    dst_tile_size: (usize, usize),
    kernel_radius: usize,
    pts_per_edge: usize,
) -> Result<Vec<TilePlan>, PlanError> {
    let (dst_rows, dst_cols) = dst_shape;
    let (tile_h, tile_w) = dst_tile_size;

    if tile_h == 0 || tile_w == 0 {
        return Err(PlanError::General("Tile size must be > 0".into()));
    }

    let pipeline = Pipeline::new(src_crs, dst_crs)?;
    let src_inv = src_transform
        .inverse()
        .map_err(|e| PlanError::General(e.to_string()))?;

    let mut plans = Vec::new();
    let mut row0 = 0;
    while row0 < dst_rows {
        let mut col0 = 0;
        while col0 < dst_cols {
            let plan = plan_single_tile(
                row0,
                col0,
                tile_h,
                tile_w,
                dst_rows,
                dst_cols,
                &pipeline,
                src_transform,
                &src_inv,
                dst_transform,
                src_shape,
                kernel_radius,
                pts_per_edge,
                None, // no footprint pre-filter
            )?;
            plans.push(plan);
            col0 += tile_w;
        }
        row0 += tile_h;
    }

    Ok(plans)
}

/// Plan tile-level reprojection from source to destination grids.
///
/// Divides the destination grid into tiles of `dst_tile_size` and for each tile,
/// computes the corresponding source ROI (with halo padding for the resampling kernel).
/// Tiles are planned in parallel using Rayon; output order is row-major (same as
/// sequential).
#[allow(clippy::too_many_arguments)]
pub fn plan_tiles(
    src_crs: &str,
    src_transform: &Affine,
    src_shape: (usize, usize),
    dst_crs: &str,
    dst_transform: &Affine,
    dst_shape: (usize, usize),
    dst_tile_size: (usize, usize),
    kernel_radius: usize,
    pts_per_edge: usize,
) -> Result<Vec<TilePlan>, PlanError> {
    let (dst_rows, dst_cols) = dst_shape;
    let (tile_h, tile_w) = dst_tile_size;

    if tile_h == 0 || tile_w == 0 {
        return Err(PlanError::General("Tile size must be > 0".into()));
    }

    // Build pipeline: src_crs → dst_crs. transform_inv goes dst→src.
    let pipeline = Pipeline::new(src_crs, dst_crs)?;

    let src_inv = src_transform
        .inverse()
        .map_err(|e| PlanError::General(e.to_string()))?;

    // Compute source footprint in dst pixel space once up front.
    // Returns None when the source has zero coverage in the destination image,
    // which lets us skip all per-tile projection work entirely.
    let footprint = src_footprint_in_dst_pixels(
        src_crs,
        src_transform,
        src_shape,
        dst_crs,
        dst_transform,
        dst_shape,
        pts_per_edge,
    );

    // Pre-generate tile (row0, col0) pairs in row-major order.
    // Vec::into_par_iter() is an IndexedParallelIterator, so collect() below
    // preserves this ordering despite parallel execution.
    let row_starts: Vec<usize> = (0..)
        .map(|i| i * tile_h)
        .take_while(|&r| r < dst_rows)
        .collect();
    let col_starts: Vec<usize> = (0..)
        .map(|i| i * tile_w)
        .take_while(|&c| c < dst_cols)
        .collect();

    let tile_coords: Vec<(usize, usize)> = row_starts
        .iter()
        .flat_map(|&r| col_starts.iter().map(move |&c| (r, c)))
        .collect();

    // Fast path: source has no coverage in the destination image.
    // Generate all-nodata plans without any projection calls.
    if footprint.is_none() {
        return Ok(tile_coords
            .iter()
            .map(|&(r0, c0)| {
                let r1 = (r0 + tile_h).min(dst_rows);
                let c1 = (c0 + tile_w).min(dst_cols);
                let (dst_ox, dst_oy) = dst_transform.forward(c0 as f64, r0 as f64);
                let dst_tile_transform = Affine::new(
                    dst_transform.a,
                    dst_transform.b,
                    dst_ox,
                    dst_transform.d,
                    dst_transform.e,
                    dst_oy,
                );
                TilePlan {
                    dst_slice: (r0, r1, c0, c1),
                    src_slice: (0, 0, 0, 0),
                    src_transform: *src_transform,
                    dst_transform: dst_tile_transform,
                    dst_tile_shape: (r1 - r0, c1 - c0),
                    has_data: false,
                }
            })
            .collect());
    }

    // All captured values are either Copy or &T where T: Sync — closure is Send.
    tile_coords
        .into_par_iter()
        .map(|(row0, col0)| {
            plan_single_tile(
                row0,
                col0,
                tile_h,
                tile_w,
                dst_rows,
                dst_cols,
                &pipeline,
                src_transform,
                &src_inv,
                dst_transform,
                src_shape,
                kernel_radius,
                pts_per_edge,
                footprint,
            )
        })
        .collect::<Result<Vec<_>, _>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn utm33_affine() -> Affine {
        Affine::new(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
    }

    #[test]
    fn test_same_crs_4_tiles_cover_full_extent() {
        let transform = utm33_affine();
        let shape = (64, 64);
        let plans = plan_tiles(
            "EPSG:32633",
            &transform,
            shape,
            "EPSG:32633",
            &transform,
            shape,
            (32, 32),
            1,
            8,
        )
        .unwrap();

        assert_eq!(plans.len(), 4);

        // Verify tiles cover the full extent without gaps
        let mut covered = vec![vec![false; 64]; 64];
        for plan in &plans {
            let (r0, r1, c0, c1) = plan.dst_slice;
            for row in &mut covered[r0..r1] {
                for cell in &mut row[c0..c1] {
                    assert!(!*cell, "Overlapping tiles");
                    *cell = true;
                }
            }
            assert!(plan.has_data);
        }
        for row in &covered {
            for &cell in row {
                assert!(cell, "Gap in coverage");
            }
        }
    }

    #[test]
    fn test_cross_crs_valid_slices() {
        let src_transform = utm33_affine();
        let src_shape = (64, 64);
        let dst_transform = Affine::new(0.001, 0.0, 14.0, 0.0, -0.001, 60.0);
        let dst_shape = (64, 64);

        let plans = plan_tiles(
            "EPSG:32633",
            &src_transform,
            src_shape,
            "EPSG:4326",
            &dst_transform,
            dst_shape,
            (32, 32),
            1,
            8,
        )
        .unwrap();

        assert!(!plans.is_empty());

        for plan in &plans {
            if plan.has_data {
                let (sr0, sr1, sc0, sc1) = plan.src_slice;
                assert!(sr1 <= src_shape.0, "src row end {sr1} > {}", src_shape.0);
                assert!(sc1 <= src_shape.1, "src col end {sc1} > {}", src_shape.1);
                assert!(sr1 > sr0, "Empty src rows");
                assert!(sc1 > sc0, "Empty src cols");
            }
        }
    }

    #[test]
    fn test_halo_padding() {
        // Same CRS, single tile covering the whole image
        let transform = utm33_affine();
        let shape = (64, 64);

        // With kernel_radius=0 (nearest)
        let plans_no_halo = plan_tiles(
            "EPSG:32633",
            &transform,
            shape,
            "EPSG:32633",
            &transform,
            shape,
            (32, 32),
            0,
            8,
        )
        .unwrap();

        // With kernel_radius=3 (lanczos)
        let plans_halo = plan_tiles(
            "EPSG:32633",
            &transform,
            shape,
            "EPSG:32633",
            &transform,
            shape,
            (32, 32),
            3,
            8,
        )
        .unwrap();

        // Interior tile (not touching source boundary) should have wider src_slice with halo
        // Look at the second tile (row 0-32, col 32-64) — its src should start before col 32
        let tile_no_halo = &plans_no_halo[1]; // (0,32, 32,64)
        let tile_halo = &plans_halo[1];

        assert!(
            tile_halo.src_slice.2 < tile_no_halo.src_slice.2 || tile_halo.src_slice.2 == 0,
            "Halo should widen source slice on the left: {:?} vs {:?}",
            tile_halo.src_slice,
            tile_no_halo.src_slice,
        );
    }

    #[test]
    fn test_edge_tile_clipping() {
        // Uneven: 100x100 with 64x64 tiles → last tile is partial
        let transform = utm33_affine();
        let shape = (100, 100);

        let plans = plan_tiles(
            "EPSG:32633",
            &transform,
            shape,
            "EPSG:32633",
            &transform,
            shape,
            (64, 64),
            1,
            8,
        )
        .unwrap();

        // Should have 2x2 = 4 tiles
        assert_eq!(plans.len(), 4);

        // Last tile should be 36x36
        let last = &plans[3];
        assert_eq!(last.dst_slice, (64, 100, 64, 100));
        assert_eq!(last.dst_tile_shape, (36, 36));

        // Source slices should be within bounds
        for plan in &plans {
            if plan.has_data {
                assert!(plan.src_slice.1 <= shape.0);
                assert!(plan.src_slice.3 <= shape.1);
            }
        }
    }

    #[test]
    fn test_has_data_false_for_out_of_bounds() {
        // Source is tiny UTM, dest is far away in geographic coords
        let src_transform = utm33_affine();
        let src_shape = (4, 4);
        // Destination is in the southern hemisphere, far from UTM 33N
        let dst_transform = Affine::new(1.0, 0.0, -180.0, 0.0, -1.0, -60.0);
        let dst_shape = (4, 4);

        let plans = plan_tiles(
            "EPSG:32633",
            &src_transform,
            src_shape,
            "EPSG:4326",
            &dst_transform,
            dst_shape,
            (4, 4),
            1,
            8,
        )
        .unwrap();

        // The tile should have no data since source is far from destination
        assert_eq!(plans.len(), 1);
        assert!(!plans[0].has_data);
    }

    #[test]
    fn test_affine_adjustment() {
        let transform = utm33_affine();
        let shape = (64, 64);

        let plans = plan_tiles(
            "EPSG:32633",
            &transform,
            shape,
            "EPSG:32633",
            &transform,
            shape,
            (32, 32),
            1,
            8,
        )
        .unwrap();

        // First tile starts at (0,0), so dst_transform origin should match original
        let first = &plans[0];
        assert_relative_eq!(first.dst_transform.c, transform.c, epsilon = 1e-10);
        assert_relative_eq!(first.dst_transform.f, transform.f, epsilon = 1e-10);
        assert_relative_eq!(first.dst_transform.a, transform.a, epsilon = 1e-10);
        assert_relative_eq!(first.dst_transform.e, transform.e, epsilon = 1e-10);

        // Second tile starts at col=32, so dst origin x should be shifted
        let second = &plans[1]; // (0, 32, 32, 64)
        let expected_x = transform.c + 32.0 * transform.a;
        assert_relative_eq!(second.dst_transform.c, expected_x, epsilon = 1e-10);
        assert_relative_eq!(second.dst_transform.f, transform.f, epsilon = 1e-10);
    }

    #[test]
    fn test_no_chunks_single_tile() {
        let transform = utm33_affine();
        let shape = (64, 64);

        let plans = plan_tiles(
            "EPSG:32633",
            &transform,
            shape,
            "EPSG:32633",
            &transform,
            shape,
            shape,
            1,
            8,
        )
        .unwrap();

        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].dst_slice, (0, 64, 0, 64));
        assert_eq!(plans[0].dst_tile_shape, (64, 64));
    }

    #[test]
    fn test_tile_boundary_points_basic() {
        let pts = tile_boundary_points(0, 4, 0, 4, 4);
        // Should have 4*4 - 4 corners counted twice on edges = 4+4+2+2 = 12
        assert_eq!(pts.len(), 12);

        // All points should be within tile bounds (at pixel centers)
        for (c, r) in &pts {
            assert!(*c >= 0.5 && *c <= 3.5, "col {c} out of range");
            assert!(*r >= 0.5 && *r <= 3.5, "row {r} out of range");
        }
    }

    #[test]
    fn test_zero_tile_size_error() {
        let transform = utm33_affine();
        let result = plan_tiles(
            "EPSG:32633",
            &transform,
            (64, 64),
            "EPSG:32633",
            &transform,
            (64, 64),
            (0, 32),
            1,
            8,
        );
        assert!(result.is_err());
    }

    // --- New tests ---

    #[test]
    fn test_src_footprint_helper_same_crs() {
        // Same CRS: source footprint in dst pixel space should approximately cover
        // the source image extent [0, src_rows] x [0, src_cols].
        let transform = utm33_affine();
        let src_shape = (64, 64);

        let fp = src_footprint_in_dst_pixels(
            "EPSG:32633",
            &transform,
            src_shape,
            "EPSG:32633",
            &transform,
            src_shape,
            8,
        );

        assert!(fp.is_some(), "Same-CRS footprint must be Some");
        let (min_row, max_row, min_col, max_col) = fp.unwrap();
        // Same transform → footprint should be roughly the source pixel bounds
        assert!(min_row <= 1.0, "min_row={min_row}");
        assert!(max_row >= 63.0, "max_row={max_row}");
        assert!(min_col <= 1.0, "min_col={min_col}");
        assert!(max_col >= 63.0, "max_col={max_col}");
    }

    #[test]
    fn test_src_footprint_helper_no_overlap() {
        // Source in UTM 33N (Northern Europe), destination in southern hemisphere.
        // The source projects to valid geographic coords (~15°E, ~59°N) but those
        // coords map to dst pixel (~195, ~-119) which is outside the 4x4 destination.
        let src_transform = utm33_affine();
        let src_shape = (4, 4);
        let dst_transform = Affine::new(1.0, 0.0, -180.0, 0.0, -1.0, -60.0);
        let dst_shape = (4, 4);

        let fp = src_footprint_in_dst_pixels(
            "EPSG:32633",
            &src_transform,
            src_shape,
            "EPSG:4326",
            &dst_transform,
            dst_shape,
            8,
        );

        assert!(fp.is_none(), "Non-overlapping footprint must be None");
    }

    #[test]
    fn test_prefilter_skips_clearly_outside_tiles() {
        // Small source (64x64 pixels) in UTM 33N.
        // Large destination grid in geographic coords that extends far beyond the source.
        // The destination spans 0°–40°E, 40°–70°N in 1° tiles.
        let src_transform = utm33_affine();
        let src_shape = (64, 64);
        // Destination: 30 rows x 40 cols at 1-degree resolution, covering 40°–70°N, 0°–40°E
        let dst_transform = Affine::new(1.0, 0.0, 0.0, 0.0, -1.0, 70.0);
        let dst_shape = (30, 40);

        let plans = plan_tiles(
            "EPSG:32633",
            &src_transform,
            src_shape,
            "EPSG:4326",
            &dst_transform,
            dst_shape,
            (5, 5),
            1,
            8,
        )
        .unwrap();

        let total = plans.len();
        let with_data = plans.iter().filter(|p| p.has_data).count();
        let without_data = plans.iter().filter(|p| !p.has_data).count();

        assert_eq!(total, with_data + without_data);
        // The source (UTM 33N, ~500km east of central meridian, ~59-60°N) covers
        // only a small portion of the destination — most tiles should have no data.
        assert!(
            without_data > with_data,
            "Expected most tiles to be filtered out (without={without_data}, with={with_data})"
        );
        // All has_data=true tiles must have valid, non-empty src_slice within bounds
        for plan in plans.iter().filter(|p| p.has_data) {
            let (sr0, sr1, sc0, sc1) = plan.src_slice;
            assert!(sr1 > sr0);
            assert!(sc1 > sc0);
            assert!(sr1 <= src_shape.0);
            assert!(sc1 <= src_shape.1);
        }
    }

    #[test]
    fn test_plan_tiles_parallel_deterministic() {
        // Calling plan_tiles twice with the same args must produce identical results.
        // This verifies that Rayon's indexed collect preserves row-major order.
        let src_transform = utm33_affine();
        let src_shape = (128, 128);
        let dst_transform = Affine::new(0.001, 0.0, 14.0, 0.0, -0.001, 60.0);
        let dst_shape = (64, 64);

        let plans_a = plan_tiles(
            "EPSG:32633",
            &src_transform,
            src_shape,
            "EPSG:4326",
            &dst_transform,
            dst_shape,
            (16, 16),
            1,
            8,
        )
        .unwrap();

        let plans_b = plan_tiles(
            "EPSG:32633",
            &src_transform,
            src_shape,
            "EPSG:4326",
            &dst_transform,
            dst_shape,
            (16, 16),
            1,
            8,
        )
        .unwrap();

        assert_eq!(plans_a.len(), plans_b.len());
        for (a, b) in plans_a.iter().zip(plans_b.iter()) {
            assert_eq!(a.dst_slice, b.dst_slice);
            assert_eq!(a.src_slice, b.src_slice);
            assert_eq!(a.has_data, b.has_data);
            assert_eq!(a.dst_tile_shape, b.dst_tile_shape);
        }
    }

    #[test]
    fn test_sequential_and_parallel_agree() {
        // Verify plan_tiles_sequential (benchmark baseline) and plan_tiles
        // produce identical tile plans for the same arguments.
        let transform = utm33_affine();
        let shape = (64, 64);

        let seq = plan_tiles_sequential(
            "EPSG:32633", &transform, shape,
            "EPSG:32633", &transform, shape,
            (16, 16), 1, 8,
        ).unwrap();

        let par = plan_tiles(
            "EPSG:32633", &transform, shape,
            "EPSG:32633", &transform, shape,
            (16, 16), 1, 8,
        ).unwrap();

        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.dst_slice, p.dst_slice);
            assert_eq!(s.src_slice, p.src_slice);
            assert_eq!(s.has_data,  p.has_data);
            assert_eq!(s.dst_tile_shape, p.dst_tile_shape);
            assert_eq!(s.src_transform, p.src_transform);
        }
    }

    #[test]
    fn test_sequential_and_parallel_agree_cross_crs() {
        // Cross-CRS case: UTM 33N source, WGS84 destination spanning a larger area.
        // Most tiles are outside the source footprint AABB, so plan_tiles (parallel)
        // fast-rejects them.  For tiles where parallel reports has_data=true, sequential
        // must agree completely — both traversed the same projection code path.
        // Tiles where parallel reports has_data=false may differ on src_slice vs sequential
        // because the AABB is sampled and therefore approximate; that is accepted behaviour.
        let src_transform = utm33_affine();
        let src_shape = (64, 64);
        // Destination: 30 rows x 40 cols at 1-degree resolution, 0°–40°E, 40°–70°N.
        // Source (~15°E, ~59.5°N) falls within this extent but covers only a few tiles.
        let dst_transform = Affine::new(1.0, 0.0, 0.0, 0.0, -1.0, 70.0);
        let dst_shape = (30, 40);

        let seq = plan_tiles_sequential(
            "EPSG:32633", &src_transform, src_shape,
            "EPSG:4326",  &dst_transform, dst_shape,
            (5, 5), 1, 8,
        ).unwrap();

        let par = plan_tiles(
            "EPSG:32633", &src_transform, src_shape,
            "EPSG:4326",  &dst_transform, dst_shape,
            (5, 5), 1, 8,
        ).unwrap();

        assert_eq!(seq.len(), par.len());

        // Confirm the footprint filter is doing work.
        let n_without_data = par.iter().filter(|p| !p.has_data).count();
        assert!(
            n_without_data > par.len() / 2,
            "Expected most tiles to be filtered out (got {n_without_data}/{} without data)",
            par.len(),
        );

        for (s, p) in seq.iter().zip(par.iter()) {
            // dst geometry is index-based; both must always agree.
            assert_eq!(s.dst_slice, p.dst_slice);
            assert_eq!(s.dst_tile_shape, p.dst_tile_shape);

            // For tiles where the parallel planner found data, sequential must agree fully.
            // (The AABB prefilter is conservative on the has_data=true side: if parallel
            // says has_data=true the tile was not AABB-rejected and both paths used the
            // same projection, so results must be identical.)
            if p.has_data {
                assert!(s.has_data, "sequential missed a tile parallel found: {:?}", p.dst_slice);
                assert_eq!(s.src_slice, p.src_slice);
                assert_eq!(s.src_transform, p.src_transform);
            }
        }
    }
}
