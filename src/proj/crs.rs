use crate::error::ProjError;
use proj4rs::Proj;

/// Thin wrapper around proj4rs that handles radians/degrees conversion transparently.
///
/// proj4rs uses radians for geographic CRS, but our affine transforms produce
/// degrees for EPSG:4326-like CRS. This wrapper auto-converts.
pub struct CrsTransform {
    src: Proj,
    dst: Proj,
    src_is_geo: bool,
    dst_is_geo: bool,
}

impl CrsTransform {
    /// Create a new CRS transform from source and destination CRS strings.
    ///
    /// Accepts EPSG codes ("EPSG:4326") or PROJ strings ("+proj=utm +zone=33 ...").
    pub fn new(src_crs: &str, dst_crs: &str) -> Result<Self, ProjError> {
        let src = Proj::from_user_string(src_crs)
            .map_err(|e| ProjError::UnknownCrs(format!("{src_crs}: {e}")))?;
        let dst = Proj::from_user_string(dst_crs)
            .map_err(|e| ProjError::UnknownCrs(format!("{dst_crs}: {e}")))?;
        let src_is_geo = src.is_latlong();
        let dst_is_geo = dst.is_latlong();
        Ok(Self {
            src,
            dst,
            src_is_geo,
            dst_is_geo,
        })
    }

    /// Transform a single point from destination CRS to source CRS.
    ///
    /// This is the direction the warp engine needs: given an output pixel
    /// coordinate in dst CRS, find the corresponding source CRS coordinate.
    ///
    /// Input/output coordinates are in CRS native units (degrees for geographic,
    /// metres for projected). The radians conversion is handled internally.
    pub fn transform_inv(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let mut point = if self.dst_is_geo {
            (x.to_radians(), y.to_radians())
        } else {
            (x, y)
        };

        proj4rs::transform::transform(&self.dst, &self.src, &mut point)
            .map_err(|e| ProjError::TransformFailed(e.to_string()))?;

        if self.src_is_geo {
            Ok((point.0.to_degrees(), point.1.to_degrees()))
        } else {
            Ok(point)
        }
    }

    /// Batch transform from destination CRS to source CRS.
    ///
    /// Coordinates are modified in-place. All coordinates use CRS native units.
    pub fn transform_inv_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        if self.dst_is_geo {
            for c in coords.iter_mut() {
                c.0 = c.0.to_radians();
                c.1 = c.1.to_radians();
            }
        }

        proj4rs::transform::transform(&self.dst, &self.src, coords)
            .map_err(|e| ProjError::TransformFailed(e.to_string()))?;

        if self.src_is_geo {
            for c in coords.iter_mut() {
                c.0 = c.0.to_degrees();
                c.1 = c.1.to_degrees();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_roundtrip_4326_to_32633() {
        // Oslo, Norway: ~10.75°E, ~59.91°N
        let fwd = CrsTransform::new("EPSG:4326", "EPSG:32633").unwrap();
        let inv = CrsTransform::new("EPSG:32633", "EPSG:4326").unwrap();

        let lon = 10.75;
        let lat = 59.91;

        // 4326 -> 32633 (forward direction = inv of the wrapper)
        let (e, n) = inv.transform_inv(lon, lat).unwrap();
        // UTM zone 33 easting should be near 500000 + offset, northing near 6.6M
        assert!(e > 200_000.0 && e < 800_000.0, "easting out of range: {e}");
        assert!(
            n > 6_000_000.0 && n < 7_000_000.0,
            "northing out of range: {n}"
        );

        // 32633 -> 4326 (roundtrip)
        let (lon2, lat2) = fwd.transform_inv(e, n).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-8);
        assert_relative_eq!(lat2, lat, epsilon = 1e-8);
    }

    #[test]
    fn test_invalid_crs() {
        let result = CrsTransform::new("EPSG:99999", "EPSG:4326");
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_transform() {
        // dst=EPSG:32633, src=EPSG:4326 → transform_inv goes UTM→4326
        let ct = CrsTransform::new("EPSG:4326", "EPSG:32633").unwrap();

        let mut coords = vec![(500000.0, 6600000.0), (510000.0, 6610000.0)];
        ct.transform_inv_batch(&mut coords).unwrap();

        // Results should be in degrees (geographic source)
        for (lon, lat) in &coords {
            assert!(*lon > 5.0 && *lon < 20.0, "lon out of range: {lon}");
            assert!(*lat > 55.0 && *lat < 65.0, "lat out of range: {lat}");
        }
    }

    #[test]
    fn test_projected_to_projected() {
        // UTM 33N to Web Mercator (both projected, no degree conversion)
        let ct = CrsTransform::new("EPSG:3857", "EPSG:32633").unwrap();
        let (x, y) = ct.transform_inv(500000.0, 6600000.0).unwrap();
        // Web Mercator coords should be reasonable
        assert!(x.abs() < 20_000_000.0, "x out of range: {x}");
        assert!(y.abs() < 20_000_000.0, "y out of range: {y}");
    }
}
