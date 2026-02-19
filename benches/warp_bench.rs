use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;

use _rust::affine::Affine;
use _rust::proj::approx::LinearApprox;
use _rust::proj::crs::CrsTransform;
use _rust::proj::pipeline::Pipeline;
use _rust::resample::ResamplingMethod;
use _rust::warp::engine;

fn make_test_data(size: usize) -> (Array2<f64>, Affine, Affine, Pipeline) {
    let mut src = Array2::zeros((size, size));
    for row in 0..size {
        for col in 0..size {
            src[(row, col)] = (row * size + col) as f64;
        }
    }

    let pixel_size = 100.0;
    let src_affine = Affine::new(pixel_size, 0.0, 500000.0, 0.0, -pixel_size, 6600000.0);

    // Approximate 4326 destination covering the same area
    let dst_affine = Affine::new(0.0014, 0.0, 15.0, 0.0, -0.0009, 59.52);

    let pipeline = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();

    (src, src_affine, dst_affine, pipeline)
}

fn make_downscale_data(
    src_size: usize,
    dst_size: usize,
) -> (Array2<f64>, Affine, Affine, Pipeline) {
    let mut src = Array2::zeros((src_size, src_size));
    for row in 0..src_size {
        for col in 0..src_size {
            src[(row, col)] = (row * src_size + col) as f64;
        }
    }

    let src_pixel_size = 100.0;
    let dst_pixel_size = src_pixel_size * (src_size as f64 / dst_size as f64);
    let src_affine = Affine::new(
        src_pixel_size,
        0.0,
        500000.0,
        0.0,
        -src_pixel_size,
        6600000.0,
    );
    let dst_affine = Affine::new(
        dst_pixel_size,
        0.0,
        500000.0,
        0.0,
        -dst_pixel_size,
        6600000.0,
    );

    let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

    (src, src_affine, dst_affine, pipeline)
}

fn bench_warp_nearest(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    for &size in &sizes {
        let (src, src_affine, dst_affine, pipeline) = make_test_data(size);
        let dst_shape = (size, size);

        c.bench_function(&format!("warp_nearest_{size}x{size}"), |b| {
            b.iter(|| {
                engine::warp(
                    &src.view(),
                    &src_affine,
                    &dst_affine,
                    dst_shape,
                    &pipeline,
                    ResamplingMethod::Nearest,
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_warp_bilinear(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    for &size in &sizes {
        let (src, src_affine, dst_affine, pipeline) = make_test_data(size);
        let dst_shape = (size, size);

        c.bench_function(&format!("warp_bilinear_{size}x{size}"), |b| {
            b.iter(|| {
                engine::warp(
                    &src.view(),
                    &src_affine,
                    &dst_affine,
                    dst_shape,
                    &pipeline,
                    ResamplingMethod::Bilinear,
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_warp_cubic(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    for &size in &sizes {
        let (src, src_affine, dst_affine, pipeline) = make_test_data(size);
        let dst_shape = (size, size);

        c.bench_function(&format!("warp_cubic_{size}x{size}"), |b| {
            b.iter(|| {
                engine::warp(
                    &src.view(),
                    &src_affine,
                    &dst_affine,
                    dst_shape,
                    &pipeline,
                    ResamplingMethod::Cubic,
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_warp_lanczos(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    for &size in &sizes {
        let (src, src_affine, dst_affine, pipeline) = make_test_data(size);
        let dst_shape = (size, size);

        c.bench_function(&format!("warp_lanczos_{size}x{size}"), |b| {
            b.iter(|| {
                engine::warp(
                    &src.view(),
                    &src_affine,
                    &dst_affine,
                    dst_shape,
                    &pipeline,
                    ResamplingMethod::Lanczos,
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_warp_average(c: &mut Criterion) {
    let sizes = [(256, 64), (512, 128), (1024, 256), (2048, 512)];
    for &(src_size, dst_size) in &sizes {
        let (src, src_affine, dst_affine, pipeline) = make_downscale_data(src_size, dst_size);
        let dst_shape = (dst_size, dst_size);

        c.bench_function(&format!("warp_average_{src_size}to{dst_size}"), |b| {
            b.iter(|| {
                engine::warp(
                    &src.view(),
                    &src_affine,
                    &dst_affine,
                    dst_shape,
                    &pipeline,
                    ResamplingMethod::Average,
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_warp_scaling(c: &mut Criterion) {
    // Bilinear at increasing sizes to show scaling curve
    let sizes = [256, 512, 1024, 2048, 4096];
    for &size in &sizes {
        let (src, src_affine, dst_affine, pipeline) = make_test_data(size);
        let dst_shape = (size, size);

        c.bench_function(&format!("warp_scaling_bilinear_{size}x{size}"), |b| {
            b.iter(|| {
                engine::warp(
                    &src.view(),
                    &src_affine,
                    &dst_affine,
                    dst_shape,
                    &pipeline,
                    ResamplingMethod::Bilinear,
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_warp_thread_scaling(c: &mut Criterion) {
    // Bilinear 1024×1024 with different thread counts
    let (src, src_affine, dst_affine, pipeline) = make_test_data(1024);
    let dst_shape = (1024, 1024);

    for &threads in &[1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        c.bench_function(&format!("warp_threads_{threads}_bilinear_1024"), |b| {
            b.iter(|| {
                pool.install(|| {
                    engine::warp(
                        &src.view(),
                        &src_affine,
                        &dst_affine,
                        dst_shape,
                        &pipeline,
                        ResamplingMethod::Bilinear,
                        None,
                    )
                    .unwrap()
                })
            });
        });
    }
}

fn bench_projection_throughput(c: &mut Criterion) {
    // Points/sec for native projections
    let n = 1_000_000_usize;

    // UTM 33N
    let pipe_utm = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
    let mut coords_utm: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let x = 300000.0 + (i as f64 / n as f64) * 400000.0;
            let y = 5000000.0 + (i as f64 / n as f64) * 2000000.0;
            (x, y)
        })
        .collect();

    c.bench_function("proj_utm33_1M", |b| {
        b.iter(|| {
            for (i, c) in coords_utm.iter_mut().enumerate() {
                *c = (
                    300000.0 + (i as f64 / n as f64) * 400000.0,
                    5000000.0 + (i as f64 / n as f64) * 2000000.0,
                );
            }
            pipe_utm.transform_inv_batch(&mut coords_utm).unwrap();
        });
    });

    // Web Mercator
    let pipe_wm = Pipeline::new("EPSG:4326", "EPSG:3857").unwrap();
    let mut coords_wm: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let x = -10_000_000.0 + (i as f64 / n as f64) * 20_000_000.0;
            let y = -5_000_000.0 + (i as f64 / n as f64) * 10_000_000.0;
            (x, y)
        })
        .collect();

    c.bench_function("proj_webmerc_1M", |b| {
        b.iter(|| {
            for (i, c) in coords_wm.iter_mut().enumerate() {
                *c = (
                    -10_000_000.0 + (i as f64 / n as f64) * 20_000_000.0,
                    -5_000_000.0 + (i as f64 / n as f64) * 10_000_000.0,
                );
            }
            pipe_wm.transform_inv_batch(&mut coords_wm).unwrap();
        });
    });
}

fn bench_linear_approx_scanline(c: &mut Criterion) {
    let pipeline = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
    let dst_affine = Affine::new(100.0, 0.0, 400000.0, 0.0, -100.0, 6000000.0);
    let src_affine = Affine::new(0.001, 0.0, 13.0, 0.0, -0.001, 55.0);
    let src_inv = src_affine.inverse().unwrap();
    let approx = LinearApprox::default();

    for &width in &[256, 1024, 4096] {
        let mut out_col = vec![0.0; width];
        let mut out_row = vec![0.0; width];

        c.bench_function(&format!("linear_approx_scanline_{width}"), |b| {
            b.iter(|| {
                approx
                    .transform_scanline(
                        &pipeline,
                        &dst_affine,
                        &src_inv,
                        100,
                        width,
                        &mut out_col,
                        &mut out_row,
                    )
                    .unwrap();
            });
        });
    }
}

fn bench_crs_transform(c: &mut Criterion) {
    let ct = CrsTransform::new("EPSG:4326", "EPSG:32633").unwrap();
    let n = 1_000_000;
    let mut coords: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let lon = 10.0 + (i as f64 / n as f64) * 5.0;
            let lat = 55.0 + (i as f64 / n as f64) * 10.0;
            (lon, lat)
        })
        .collect();

    c.bench_function("crs_transform_1M_points", |b| {
        b.iter(|| {
            // Reset coords each iteration
            for (i, c) in coords.iter_mut().enumerate() {
                let lon = 10.0 + (i as f64 / n as f64) * 5.0;
                let lat = 55.0 + (i as f64 / n as f64) * 10.0;
                *c = (lon, lat);
            }
            ct.transform_inv_batch(&mut coords).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_warp_nearest,
    bench_warp_bilinear,
    bench_warp_cubic,
    bench_warp_lanczos,
    bench_warp_average,
    bench_warp_scaling,
    bench_warp_thread_scaling,
    bench_projection_throughput,
    bench_linear_approx_scanline,
    bench_crs_transform
);
criterion_main!(benches);
