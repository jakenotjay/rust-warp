use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;

use _rust::affine::Affine;
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
    let sizes = [256, 512, 1024];
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
    let sizes = [256, 512, 1024];
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
    let sizes = [256, 512, 1024];
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
    let sizes = [256, 512, 1024];
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
    // Average is most meaningful for downsampling: 512→128
    let sizes = [(256, 64), (512, 128), (1024, 256)];
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
    bench_crs_transform
);
criterion_main!(benches);
