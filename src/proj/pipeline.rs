//! Pipeline — CRS-to-CRS transform chain that dispatches between
//! native pure-Rust projections and proj4rs fallback.

use crate::error::ProjError;
use crate::proj::albers_equal_area::AlbersEqualArea;
use crate::proj::crs::CrsTransform;
use crate::proj::ellipsoid::GRS80;
use crate::proj::lambert_conformal::LambertConformalConic;
use crate::proj::mercator::WebMercator;
use crate::proj::stereographic::PolarStereographic;
use crate::proj::transverse_mercator::TransverseMercator;
use crate::proj::Projection;

/// Describes a CRS endpoint in the pipeline.
enum CrsEndpoint {
    /// Geographic CRS — coordinates are in degrees externally, radians internally.
    /// No forward/inverse needed — identity pass-through on (lon_rad, lat_rad).
    Geographic,
    /// Projected CRS — coordinates are in meters.
    Projected(Box<dyn Projection>),
}

/// A CRS-to-CRS transform pipeline.
///
/// For supported EPSG codes, uses native pure-Rust projection math.
/// Falls back to proj4rs for anything else.
pub enum Pipeline {
    /// Both src and dst are recognized native projections.
    #[allow(private_interfaces)]
    Native { src: CrsEndpoint, dst: CrsEndpoint },
    /// Fallback to proj4rs for unsupported CRSes.
    Proj4rs(Box<CrsTransform>),
}

impl Pipeline {
    /// Create a Pipeline from source and destination CRS strings.
    ///
    /// Tries to parse both as EPSG codes and create native projections.
    /// Falls back to proj4rs if either is unrecognized.
    pub fn new(src_crs: &str, dst_crs: &str) -> Result<Self, ProjError> {
        let src = parse_epsg(src_crs);
        let dst = parse_epsg(dst_crs);

        if let (Some(src_ep), Some(dst_ep)) = (src, dst) {
            Ok(Pipeline::Native {
                src: src_ep,
                dst: dst_ep,
            })
        } else {
            let ct = CrsTransform::new(src_crs, dst_crs)?;
            Ok(Pipeline::Proj4rs(Box::new(ct)))
        }
    }

    /// Transform a single point from destination CRS to source CRS.
    ///
    /// This is the direction the warp engine needs: given an output pixel
    /// coordinate in dst CRS, find the corresponding source CRS coordinate.
    ///
    /// Input/output coordinates are in CRS native units (degrees for geographic,
    /// metres for projected).
    pub fn transform_inv(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        match self {
            Pipeline::Native { src, dst } => {
                // Step 1: dst coords → (lon_rad, lat_rad)
                let (lon, lat) = match dst {
                    CrsEndpoint::Geographic => (x.to_radians(), y.to_radians()),
                    CrsEndpoint::Projected(proj) => proj.inverse(x, y)?,
                };

                // Step 2: (lon_rad, lat_rad) → src coords
                match src {
                    CrsEndpoint::Geographic => Ok((lon.to_degrees(), lat.to_degrees())),
                    CrsEndpoint::Projected(proj) => proj.forward(lon, lat),
                }
            }
            Pipeline::Proj4rs(ct) => ct.transform_inv(x, y),
        }
    }

    /// Batch transform from destination CRS to source CRS.
    ///
    /// Coordinates are modified in-place. All coordinates use CRS native units.
    pub fn transform_inv_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        match self {
            Pipeline::Native { src, dst } => {
                for c in coords.iter_mut() {
                    let (lon, lat) = match dst {
                        CrsEndpoint::Geographic => (c.0.to_radians(), c.1.to_radians()),
                        CrsEndpoint::Projected(proj) => proj.inverse(c.0, c.1)?,
                    };

                    *c = match src {
                        CrsEndpoint::Geographic => (lon.to_degrees(), lat.to_degrees()),
                        CrsEndpoint::Projected(proj) => proj.forward(lon, lat)?,
                    };
                }
                Ok(())
            }
            Pipeline::Proj4rs(ct) => ct.transform_inv_batch(coords),
        }
    }
}

/// Try to parse an EPSG code and return a `CrsEndpoint`.
fn parse_epsg(crs: &str) -> Option<CrsEndpoint> {
    let code = crs
        .strip_prefix("EPSG:")
        .or_else(|| crs.strip_prefix("epsg:"))?
        .parse::<u32>()
        .ok()?;

    match code {
        // Geographic CRS — identity pass-through (degrees ↔ radians handled by Pipeline)
        4326 => Some(CrsEndpoint::Geographic),

        // Web Mercator
        3857 => Some(CrsEndpoint::Projected(Box::new(WebMercator::new()))),

        // UTM North: EPSG:326XX (zone 1–60)
        32601..=32660 => {
            let zone = (code - 32600) as u8;
            Some(CrsEndpoint::Projected(Box::new(
                TransverseMercator::utm_zone(zone, true),
            )))
        }

        // UTM South: EPSG:327XX (zone 1–60)
        32701..=32760 => {
            let zone = (code - 32700) as u8;
            Some(CrsEndpoint::Projected(Box::new(
                TransverseMercator::utm_zone(zone, false),
            )))
        }

        // Lambert-93 (France): EPSG:2154
        2154 => Some(CrsEndpoint::Projected(Box::new(
            LambertConformalConic::new_2sp(
                GRS80,
                3.0_f64.to_radians(),  // lon0
                46.5_f64.to_radians(), // lat0
                44.0_f64.to_radians(), // lat1
                49.0_f64.to_radians(), // lat2
                700_000.0,             // false easting
                6_600_000.0,           // false northing
            ),
        ))),

        // CONUS Albers Equal Area: EPSG:5070
        5070 => Some(CrsEndpoint::Projected(Box::new(AlbersEqualArea::new(
            GRS80,
            (-96.0_f64).to_radians(), // lon0
            23.0_f64.to_radians(),    // lat0
            29.5_f64.to_radians(),    // lat1
            45.5_f64.to_radians(),    // lat2
            0.0,                      // false easting
            0.0,                      // false northing
        )))),

        // Antarctic Polar Stereographic: EPSG:3031
        3031 => Some(CrsEndpoint::Projected(Box::new(
            PolarStereographic::antarctic(),
        ))),

        // Arctic NSIDC Polar Stereographic: EPSG:3413
        3413 => Some(CrsEndpoint::Projected(Box::new(
            PolarStereographic::arctic(),
        ))),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_native_utm33_to_4326() {
        let pipe = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        // Input: UTM 33N coords → output: lon/lat in degrees
        let (lon, lat) = pipe.transform_inv(500000.0, 5760000.0).unwrap();
        assert!(lon > 14.0 && lon < 16.0, "lon = {lon}");
        assert!(lat > 51.0 && lat < 53.0, "lat = {lat}");
    }

    #[test]
    fn test_native_4326_to_utm33() {
        let pipe = Pipeline::new("EPSG:32633", "EPSG:4326").unwrap();
        // Input: lon/lat in degrees → output: UTM 33N coords
        let (e, n) = pipe.transform_inv(15.0, 52.0).unwrap();
        assert_relative_eq!(e, 500_000.0, epsilon = 1.0);
        assert!(n > 5_760_000.0 && n < 5_762_000.0, "northing = {n}");
    }

    #[test]
    fn test_native_matches_proj4rs() {
        // Verify the native pipeline produces the same results as proj4rs
        let native = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        let fallback = {
            let ct = CrsTransform::new("EPSG:4326", "EPSG:32633").unwrap();
            Pipeline::Proj4rs(Box::new(ct))
        };

        let test_points: &[(f64, f64)] = &[
            (500000.0, 5760000.0),
            (400000.0, 6000000.0),
            (600000.0, 5500000.0),
        ];

        for &(x, y) in test_points {
            let (lon_n, lat_n) = native.transform_inv(x, y).unwrap();
            let (lon_p, lat_p) = fallback.transform_inv(x, y).unwrap();
            // Should match within 1mm (< 0.001 degrees ~ 110m, so use tighter bound)
            assert_relative_eq!(lon_n, lon_p, epsilon = 1e-6);
            assert_relative_eq!(lat_n, lat_p, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_native_web_mercator() {
        let pipe = Pipeline::new("EPSG:4326", "EPSG:3857").unwrap();
        // (0, 0) in Web Mercator → (0°, 0°)
        let (lon, lat) = pipe.transform_inv(0.0, 0.0).unwrap();
        assert_relative_eq!(lon, 0.0, epsilon = 1e-10);
        assert_relative_eq!(lat, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_native_utm_to_webmerc() {
        let pipe = Pipeline::new("EPSG:3857", "EPSG:32633").unwrap();
        let (x, y) = pipe.transform_inv(500000.0, 5760000.0).unwrap();
        // Should produce valid Web Mercator coordinates
        assert!(x.abs() < 20_037_509.0, "x = {x}");
        assert!(y.abs() < 20_037_509.0, "y = {y}");
    }

    #[test]
    fn test_fallback_to_proj4rs() {
        // Use a CRS that's not natively supported → should fallback to proj4rs
        let pipe = Pipeline::new("EPSG:4326", "EPSG:32661").unwrap();
        assert!(matches!(pipe, Pipeline::Proj4rs(_)));
    }

    #[test]
    fn test_batch_transform() {
        let pipe = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        let mut coords = vec![(500000.0, 5760000.0), (510000.0, 5770000.0)];
        pipe.transform_inv_batch(&mut coords).unwrap();
        for (lon, lat) in &coords {
            assert!(*lon > 14.0 && *lon < 16.0, "lon = {lon}");
            assert!(*lat > 51.0 && *lat < 53.0, "lat = {lat}");
        }
    }

    #[test]
    fn test_identity_same_crs() {
        let pipe = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();
        let (x, y) = pipe.transform_inv(500000.0, 5760000.0).unwrap();
        assert_relative_eq!(x, 500000.0, epsilon = 0.01);
        assert_relative_eq!(y, 5760000.0, epsilon = 0.01);
    }

    #[test]
    fn test_native_vs_proj4rs_100_random_points() {
        // Cross-validate native projections against proj4rs at 100 random-ish points
        let crs_pairs = [
            ("EPSG:4326", "EPSG:32633"), // UTM 33N
            ("EPSG:4326", "EPSG:32617"), // UTM 17N
            ("EPSG:4326", "EPSG:3857"),  // Web Mercator
        ];

        for &(src, dst) in &crs_pairs {
            let native = Pipeline::new(src, dst).unwrap();
            let fallback = {
                let ct = CrsTransform::new(src, dst).unwrap();
                Pipeline::Proj4rs(Box::new(ct))
            };

            // Generate test points in the destination CRS valid range
            let base_points: Vec<(f64, f64)> = match dst {
                "EPSG:32633" => (0..100)
                    .map(|i| {
                        let x = 300000.0 + (i as f64 / 99.0) * 400000.0;
                        let y = 5000000.0 + (i as f64 / 99.0) * 2000000.0;
                        (x, y)
                    })
                    .collect(),
                "EPSG:32617" => (0..100)
                    .map(|i| {
                        let x = 300000.0 + (i as f64 / 99.0) * 400000.0;
                        let y = 3000000.0 + (i as f64 / 99.0) * 2000000.0;
                        (x, y)
                    })
                    .collect(),
                "EPSG:3857" => (0..100)
                    .map(|i| {
                        let x = -10000000.0 + (i as f64 / 99.0) * 20000000.0;
                        let y = -8000000.0 + (i as f64 / 99.0) * 16000000.0;
                        (x, y)
                    })
                    .collect(),
                _ => unreachable!(),
            };

            let mut max_err = 0.0_f64;
            for &(x, y) in &base_points {
                let (nx, ny) = native.transform_inv(x, y).unwrap();
                let (px, py) = fallback.transform_inv(x, y).unwrap();
                let err = ((nx - px).powi(2) + (ny - py).powi(2)).sqrt();
                max_err = max_err.max(err);
            }

            // < 1mm error between native and proj4rs
            assert!(
                max_err < 0.001,
                "CRS {src}->{dst}: max error = {max_err:.6} m (want < 0.001 m)"
            );
        }
    }

    #[test]
    fn test_roundtrip_utm_zones() {
        // UTM → 4326 → UTM should return to original coordinates.
        // transform_inv goes dst→src, so:
        //   Pipeline::new("EPSG:4326", utm) : transform_inv(utm_x, utm_y) → (lon, lat)
        //   Pipeline::new(utm, "EPSG:4326") : transform_inv(lon, lat) → (utm_x, utm_y)
        for zone in [17_u32, 33, 45, 1, 60] {
            let crs = format!("EPSG:326{zone:02}");
            let to_4326 = Pipeline::new("EPSG:4326", &crs).unwrap();
            let from_4326 = Pipeline::new(&crs, "EPSG:4326").unwrap();

            let orig = (500000.0, 5500000.0);
            let (lon, lat) = to_4326.transform_inv(orig.0, orig.1).unwrap();
            let (x, y) = from_4326.transform_inv(lon, lat).unwrap();

            assert_relative_eq!(x, orig.0, epsilon = 0.01);
            assert_relative_eq!(y, orig.1, epsilon = 0.01);
        }
    }

    #[test]
    fn test_native_lambert93() {
        // EPSG:2154 should route to native Lambert
        let pipe = Pipeline::new("EPSG:4326", "EPSG:2154").unwrap();
        assert!(matches!(pipe, Pipeline::Native { .. }));
        // Paris (lon=2.35, lat=48.86) → Lambert-93 coords
        let (lon, lat) = pipe.transform_inv(652_000.0, 6_862_000.0).unwrap();
        assert!(lon > 1.0 && lon < 4.0, "lon = {lon}");
        assert!(lat > 47.0 && lat < 50.0, "lat = {lat}");
    }

    #[test]
    fn test_native_lambert93_matches_proj4rs() {
        let native = Pipeline::new("EPSG:4326", "EPSG:2154").unwrap();
        let fallback = {
            let ct = CrsTransform::new("EPSG:4326", "EPSG:2154").unwrap();
            Pipeline::Proj4rs(Box::new(ct))
        };
        let test_points: &[(f64, f64)] = &[
            (652_000.0, 6_862_000.0), // Paris area
            (400_000.0, 6_400_000.0), // SW France
            (900_000.0, 7_000_000.0), // NE France
        ];
        for &(x, y) in test_points {
            let (lon_n, lat_n) = native.transform_inv(x, y).unwrap();
            let (lon_p, lat_p) = fallback.transform_inv(x, y).unwrap();
            assert_relative_eq!(lon_n, lon_p, epsilon = 1e-6);
            assert_relative_eq!(lat_n, lat_p, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_native_albers_conus() {
        // EPSG:5070 should route to native Albers
        let pipe = Pipeline::new("EPSG:4326", "EPSG:5070").unwrap();
        assert!(matches!(pipe, Pipeline::Native { .. }));
        // Central US → lon ~ -96, lat ~ 39
        let (lon, lat) = pipe.transform_inv(0.0, 1_700_000.0).unwrap();
        assert!(lon > -100.0 && lon < -90.0, "lon = {lon}");
        assert!(lat > 35.0 && lat < 45.0, "lat = {lat}");
    }

    #[test]
    fn test_native_albers_matches_proj4rs() {
        let native = Pipeline::new("EPSG:4326", "EPSG:5070").unwrap();
        let fallback = {
            let ct = CrsTransform::new("EPSG:4326", "EPSG:5070").unwrap();
            Pipeline::Proj4rs(Box::new(ct))
        };
        let test_points: &[(f64, f64)] = &[
            (0.0, 1_700_000.0),         // Central US
            (-2_000_000.0, 500_000.0),  // West coast
            (2_000_000.0, 1_500_000.0), // East coast
        ];
        for &(x, y) in test_points {
            let (lon_n, lat_n) = native.transform_inv(x, y).unwrap();
            let (lon_p, lat_p) = fallback.transform_inv(x, y).unwrap();
            assert_relative_eq!(lon_n, lon_p, epsilon = 1e-5);
            assert_relative_eq!(lat_n, lat_p, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_native_polar_antarctic() {
        // EPSG:3031 should route to native Polar Stereographic
        let pipe = Pipeline::new("EPSG:4326", "EPSG:3031").unwrap();
        assert!(matches!(pipe, Pipeline::Native { .. }));
        // Origin (0, 0) in EPSG:3031 is the South Pole
        let (_lon, lat) = pipe.transform_inv(0.0, 0.0).unwrap();
        assert!(lat < -85.0, "lat = {lat}");
    }

    #[test]
    fn test_native_polar_arctic() {
        // EPSG:3413 should route to native Polar Stereographic
        let pipe = Pipeline::new("EPSG:4326", "EPSG:3413").unwrap();
        assert!(matches!(pipe, Pipeline::Native { .. }));
        // Origin (0, 0) in EPSG:3413 is the North Pole
        let (_lon, lat) = pipe.transform_inv(0.0, 0.0).unwrap();
        assert!(lat > 85.0, "lat = {lat}");
    }

    #[test]
    fn test_native_polar_matches_proj4rs() {
        for (epsg, test_pts) in [
            (
                "EPSG:3031",
                vec![
                    (100_000.0, -100_000.0),
                    (500_000.0, 500_000.0),
                    (-300_000.0, -200_000.0),
                ],
            ),
            (
                "EPSG:3413",
                vec![
                    (100_000.0, -100_000.0),
                    (300_000.0, 500_000.0),
                    (-200_000.0, -300_000.0),
                ],
            ),
        ] {
            let native = Pipeline::new("EPSG:4326", epsg).unwrap();
            let fallback = {
                let ct = CrsTransform::new("EPSG:4326", epsg).unwrap();
                Pipeline::Proj4rs(Box::new(ct))
            };
            for (x, y) in test_pts {
                let (lon_n, lat_n) = native.transform_inv(x, y).unwrap();
                let (lon_p, lat_p) = fallback.transform_inv(x, y).unwrap();
                assert_relative_eq!(lon_n, lon_p, epsilon = 1e-6);
                assert_relative_eq!(lat_n, lat_p, epsilon = 1e-6);
            }
        }
    }
}
