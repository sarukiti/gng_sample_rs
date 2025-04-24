use nalgebra::Point2;
use image::{self, DynamicImage, GenericImageView};
use std::collections::HashMap; // HashMap をインポート

use plotters::prelude::*;
use plotters_bitmap::BitMapBackend;

use crate::MAX_N;

// 画像ファイルからデータ点を抽出する (変更なし)
pub fn extract_data_from_image(img: &DynamicImage) -> Result<Vec<Point2<f64>>, image::ImageError> {
    let mut data_points = Vec::new();
    for (x, y, pixel) in img.pixels() {
        // 例: 黒いピクセルをデータ点とする (閾値は適宜調整)
        if pixel[0] < 128 && pixel[1] < 128 && pixel[2] < 128 {
            // 画像座標系からプロット座標系へ（必要なら調整）
            data_points.push(Point2::new(x as f64, (img.height() - 1 - y) as f64));
        }
    }
    Ok(data_points)
}

// --- GNG 状態をプロットする関数 (修正) ---
pub fn plot_gng_state(
    filename: &str,
    iteration: usize,
    data_points: &Vec<Point2<f64>>, // 元のデータ点
    neurons: &Vec<Point2<f64>>,
    neurons_exist: &Vec<bool>,
    edges: &Vec<HashMap<usize, f64>>, // connectivity の代わりに edges を受け取る
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?; // 背景を白に

    // 描画範囲を決定 (データ点とニューロンを含むように) (変更なし)
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

    // 1. 元のデータ点をプロット (薄い色で) (変更なし)
    chart.draw_series(
        data_points
            .iter()
            .map(|p| Circle::new((p.x, p.y), 2, ShapeStyle::from(&BLACK.mix(0.1)).filled())),
    )?;

    // 2. アクティブなニューロンをプロット (赤色) (変更なし)
    let active_neurons: Vec<(f64, f64)> = (0..MAX_N)
        .filter(|&i| neurons_exist[i])
        .map(|i| (neurons[i].x, neurons[i].y))
        .collect();
    chart.draw_series(
        active_neurons
            .iter()
            .map(|p| Circle::new(*p, 4, ShapeStyle::from(&RED).filled())),
    )?;

    // 3. エッジ (接続) をプロット (青線) (edges を使用するように修正)
    let mut edge_lines = Vec::new();
    for r in 0..MAX_N {
        if !neurons_exist[r] {
            continue;
        }
        // edges[r] から接続情報を取得
        for (&c, _) in &edges[r] {
            // 重複描画を防ぐため r < c のペアのみ描画
            // また、接続先の c も存在するか確認 (通常は存在するはずだが念のため)
            if r < c && neurons_exist[c] {
                edge_lines.push(((neurons[r].x, neurons[r].y), (neurons[c].x, neurons[c].y)));
            }
        }
    }
    chart.draw_series(
        edge_lines
            .into_iter()
            .map(|(p1, p2)| PathElement::new(vec![p1, p2], &BLUE)),
    )?;

    root.present()?; // 描画を確定
    Ok(())
}

// --- 描画範囲を計算するヘルパー関数 (変更なし) ---
fn find_bounds(
    data_points: &Vec<Point2<f64>>,
    neurons: &Vec<Point2<f64>>,
    neurons_exist: &Vec<bool>,
) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let update_bounds = |p: &Point2<f64>, min_x: &mut f64, max_x: &mut f64, min_y: &mut f64, max_y: &mut f64| {
        *min_x = min_x.min(p.x);
        *max_x = max_x.max(p.x);
        *min_y = min_y.min(p.y);
        *max_y = max_y.max(p.y);
    };

    for p in data_points {
        update_bounds(p, &mut min_x, &mut max_x, &mut min_y, &mut max_y);
    }
    for i in 0..MAX_N {
        if neurons_exist[i] {
            update_bounds(&neurons[i], &mut min_x, &mut max_x, &mut min_y, &mut max_y);
        }
    }

    // データやニューロンがない場合のデフォルト値
    if min_x == f64::INFINITY { min_x = 0.0; }
    if max_x == f64::NEG_INFINITY { max_x = 1.0; }
    if min_y == f64::INFINITY { min_y = 0.0; }
    if max_y == f64::NEG_INFINITY { max_y = 1.0; }

    (min_x, max_x, min_y, max_y)
}
