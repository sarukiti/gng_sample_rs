use nalgebra::Point2;
use image::{self, DynamicImage, GenericImageView};

use plotters::prelude::*;
use plotters_bitmap::BitMapBackend;

use crate::MAX_N;

// 画像ファイルからデータ点を抽出する
pub fn extract_data_from_image(img: &DynamicImage) -> Result<Vec<Point2<f64>>, image::ImageError> {
    let mut non_zero_point = Vec::new();
    for (x, y, pixel) in img.pixels() {
        if pixel.0.iter().any(|&channel_value| channel_value < 240) {
            non_zero_point.push(Point2::new(x as f64, y as f64));
        }
    }
    Ok(non_zero_point)
}

// --- GNG 状態をプロットする関数 ---
pub fn plot_gng_state(
    filename: &str,
    iteration: usize,
    data_points: &Vec<Point2<f64>>, // 元のデータ点
    neurons: &Vec<Point2<f64>>,
    neurons_exist: &Vec<bool>,
    connectivity: &Vec<Vec<bool>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?; // 背景を白に

    // 描画範囲を決定 (データ点とニューロンを含むように)
    let (min_x, max_x, min_y, max_y) = find_bounds(data_points, neurons, neurons_exist);
    let x_range = (min_x - 10.0)..(max_x + 10.0); // 少しマージンを追加
    let y_range = (min_y - 10.0)..(max_y + 10.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("GNG State - Iteration {}", iteration),
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // 1. 元のデータ点をプロット (薄い色で)
    chart.draw_series(
        data_points
            .iter()
            .map(|p| Circle::new((p.x, p.y), 2, ShapeStyle::from(&BLACK.mix(0.1)).filled())),
    )?;

    // 2. アクティブなニューロンをプロット (赤色)
    let active_neurons: Vec<(f64, f64)> = (0..MAX_N)
        .filter(|&i| neurons_exist[i])
        .map(|i| (neurons[i].x, neurons[i].y))
        .collect();
    chart.draw_series(
        active_neurons
            .iter()
            .map(|p| Circle::new(*p, 4, ShapeStyle::from(&RED).filled())),
    )?;

    // 3. エッジ (接続) をプロット (青線)
    let mut edges = Vec::new();
    for r in 0..MAX_N {
        if !neurons_exist[r] {
            continue;
        }
        for c in (r + 1)..MAX_N {
            // 重複を避ける
            if neurons_exist[c] && connectivity[r][c] {
                edges.push(((neurons[r].x, neurons[r].y), (neurons[c].x, neurons[c].y)));
            }
        }
    }
    chart.draw_series(
        edges
            .into_iter()
            .map(|(p1, p2)| PathElement::new(vec![p1, p2], &BLUE)),
    )?;

    root.present()?; // 描画を確定
    Ok(())
}

// --- 描画範囲を計算するヘルパー関数 ---
fn find_bounds(
    data_points: &Vec<Point2<f64>>,
    neurons: &Vec<Point2<f64>>,
    neurons_exist: &Vec<bool>,
) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    // データ点の範囲
    for p in data_points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    // アクティブなニューロンの範囲
    for i in 0..MAX_N {
        if neurons_exist[i] {
            min_x = min_x.min(neurons[i].x);
            max_x = max_x.max(neurons[i].x);
            min_y = min_y.min(neurons[i].y);
            max_y = max_y.max(neurons[i].y);
        }
    }

    // デフォルト値 (データもニューロンもない場合)
    if min_x == f64::INFINITY {
        min_x = 0.0;
    }
    if max_x == f64::NEG_INFINITY {
        max_x = 500.0;
    } // main の範囲に合わせる
    if min_y == f64::INFINITY {
        min_y = 0.0;
    }
    if max_y == f64::NEG_INFINITY {
        max_y = 500.0;
    }

    (min_x, max_x, min_y, max_y)
}
