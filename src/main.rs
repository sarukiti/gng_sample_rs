mod utils;

use nalgebra::{Point2, Vector2};
use rand::Rng;
use std::cmp::Ordering;
use std::vec::Vec; // Ordering をインポート

// plotters のインポートを追加
use plotters::prelude::*;
use plotters_bitmap::BitMapBackend; // Bitmap バックエンドを使用

use utils::extract_data_from_image;

// --- 定数 ---
const LR_S1: f64 = 0.5; // 勝者ニューロンの学習率
const LR_S2: f64 = 0.02; // 隣接ニューロンの学習率
const BETA: f64 = 0.02; // 全ニューロンのエラー減衰率
const ALPHA: f64 = 0.5; // 新規ノード挿入時のエラー減衰率
const MAX_N: usize = 500; // 最大ニューロン数
const TH_AGE: f64 = 2.0; // エッジの最大年齢
const LAMBDA_VALUE: usize = 4; // 新規ノード挿入の間隔 (ステップ数)
const MAX_ITERATIONS: usize = 500; // 最大イテレーション数を定義

// --- GNG アルゴリズム実装 ---
fn run_gng(cluster_pos: Vec<Point2<f64>>) {
    if cluster_pos.is_empty() {
        eprintln!("Error: Input data (cluster_pos) is empty.");
        return;
    }
    if MAX_N < 2 {
        eprintln!("Error: MAX_N must be at least 2.");
        return;
    }

    // プロット結果を保存するディレクトリを作成
    let plot_dir = "gng_plots";
    std::fs::create_dir_all(plot_dir).expect("Failed to create plot directory"); // "プロットディレクトリの作成に失敗しました"

    let mut rng = rand::thread_rng();
    let data_len = cluster_pos.len();

    // --- 初期化 (ステップ 0) ---
    let mut neurons: Vec<Point2<f64>> = vec![Point2::origin(); MAX_N];
    let mut neurons_error: Vec<f64> = vec![0.0; MAX_N];
    let mut neurons_exist: Vec<bool> = vec![false; MAX_N];
    let mut connectivity: Vec<Vec<bool>> = vec![vec![false; MAX_N]; MAX_N];
    let mut edge_age: Vec<Vec<f64>> = vec![vec![0.0; MAX_N]; MAX_N];

    neurons[0] = Point2::new(rng.gen_range(0.0..500.0), rng.gen_range(0.0..500.0));
    neurons[1] = Point2::new(rng.gen_range(0.0..500.0), rng.gen_range(0.0..500.0));
    neurons_exist[0] = true;
    neurons_exist[1] = true;
    connectivity[0][1] = true;
    connectivity[1][0] = true;

    // --- メインループ ---
    for i in 1..MAX_ITERATIONS {
        // ステップ 1: ランダムな入力ベクトル v を取得
        let v_index = rng.gen_range(0..data_len);
        let v = cluster_pos[v_index]; // v は Point2<f64>

        // ステップ 2: 勝者 s1 と 2 番目の勝者 s2 を見つける
        let active_indices: Vec<usize> = (0..MAX_N).filter(|&idx| neurons_exist[idx]).collect();

        if active_indices.len() < 2 {
            println!(
                "Warning: Less than 2 active neurons at iteration {}. Skipping.",
                i
            ); // "警告: イテレーション {} でアクティブなニューロンが 2 未満です。スキップします。"
            continue;
        }

        // v からすべてのアクティブなニューロンまでの二乗距離を計算
        let mut distances: Vec<(usize, f64)> = active_indices
            .iter()
            .map(|&idx| {
                let dist_sq = nalgebra::distance_squared(&neurons[idx], &v);
                (idx, dist_sq)
            })
            .collect();

        // 距離でソート (昇順)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let s1_index = distances[0].0;
        let s2_index = distances[1].0;

        // ステップ 3: 勝者 s1 の累積エラーに二乗誤差を加算
        neurons_error[s1_index] += distances[0].1; // distances[0].1 は dist_sq(s1, v)

        // ステップ 4: s1 とそのトポロジカルな隣接ニューロンの参照ベクトルを更新
        // s1 を更新
        let s1_update_vector: Vector2<f64> = (v - neurons[s1_index]) * LR_S1;
        neurons[s1_index] += s1_update_vector;

        // s1 の直接の隣接ニューロンを更新 (標準的な GNG の解釈)
        for k in 0..MAX_N {
            if connectivity[s1_index][k] {
                // k が s1 の隣接ニューロンの場合
                let neighbor_update_vector: Vector2<f64> = (v - neurons[k]) * LR_S2;
                neurons[k] += neighbor_update_vector;
            }
        }

        // ステップ 5: エッジの年齢をリセットするか、s1 と s2 の間にエッジを作成
        if !connectivity[s1_index][s2_index] {
            connectivity[s1_index][s2_index] = true;
            connectivity[s2_index][s1_index] = true;
        }
        edge_age[s1_index][s2_index] = 0.0;
        edge_age[s2_index][s1_index] = 0.0;

        // ステップ 6: s1 に接続されているすべてのエッジの年齢をインクリメント
        for k in 0..MAX_N {
            if k != s1_index && connectivity[s1_index][k] {
                edge_age[s1_index][k] += 1.0;
                edge_age[k][s1_index] += 1.0;
            }
        }

        // ステップ 7: th_age より古いエッジと、その結果として孤立したノードを削除
        let mut edges_to_remove = Vec::new();
        for r in 0..MAX_N {
            for c in (r + 1)..MAX_N {
                // 上三角をチェック
                if connectivity[r][c] && edge_age[r][c] > TH_AGE {
                    edges_to_remove.push((r, c));
                }
            }
        }

        for (r, c) in edges_to_remove {
            connectivity[r][c] = false;
            connectivity[c][r] = false;
        }

        // 古いエッジを削除した後に孤立したノードを確認
        let mut nodes_to_remove = Vec::new();
        for k in 0..MAX_N {
            if neurons_exist[k] {
                // 行 k に true の値 (接続) があるか確認
                let has_connection = connectivity[k].iter().any(|&connected| connected);
                if !has_connection {
                    nodes_to_remove.push(k);
                }
            }
        }
        for k in nodes_to_remove {
            if neurons_exist[k] {
                // 再度確認
                // println!("Removing isolated neuron {}", k); // デバッグ: "孤立したニューロン {} を削除中"
                neurons_exist[k] = false;
                neurons_error[k] = 0.0; // エラーをリセット
                // 削除されたノードの接続をクリア (重要!)
                for j in 0..MAX_N {
                    connectivity[k][j] = false;
                    connectivity[j][k] = false;
                    edge_age[k][j] = 0.0; // 年齢もリセット
                    edge_age[j][k] = 0.0;
                }
            }
        }

        // ステップ 8: lambda ステップごとに新しいノードを挿入
        let current_neuron_count = neurons_exist.iter().filter(|&&e| e).count();
        if i % LAMBDA_VALUE == 0 && current_neuron_count < MAX_N {
            // (i) 既存のニューロンの中で最大のエラーを持つノード q を見つける
            let q_index_option =
                (0..MAX_N)
                    .filter(|&idx| neurons_exist[idx])
                    .max_by(|&idx_a, &idx_b| {
                        neurons_error[idx_a]
                            .partial_cmp(&neurons_error[idx_b])
                            .unwrap_or(Ordering::Equal)
                    });

            if let Some(q_index) = q_index_option {
                // (ii) q の隣接ニューロン f の中で最も遠いものを見つける (Python コードに従う)
                let q_pos = neurons[q_index];
                let mut f_index_option: Option<usize> = None;
                let mut max_dist_sq = -1.0;

                for k in 0..MAX_N {
                    if k != q_index && connectivity[q_index][k] {
                        // k が隣接ニューロンの場合
                        let neighbor_pos = neurons[k];
                        let dist_sq = nalgebra::distance_squared(&neighbor_pos, &q_pos);
                        if dist_sq > max_dist_sq {
                            max_dist_sq = dist_sq;
                            f_index_option = Some(k);
                        }
                    }
                }

                if let Some(f_index) = f_index_option {
                    // 新しいノード r のインデックスを見つける (最初の存在しないニューロン)
                    let r_index_option = neurons_exist.iter().position(|&exists| !exists);

                    if let Some(r_index) = r_index_option {
                        let f_pos = neurons[f_index];

                        // 新しいノード r を q と f の中間点に挿入
                        neurons_exist[r_index] = true;
                        // nalgebra::center またはベクトル演算を使用して中間点を計算
                        neurons[r_index] = nalgebra::center(&q_pos, &f_pos);
                        // 代替: neurons[r_index] = q_pos + (f_pos - q_pos) * 0.5;

                        // (iii) エッジ q-f を削除し、エッジ q-r と r-f を追加
                        connectivity[q_index][f_index] = false;
                        connectivity[f_index][q_index] = false;
                        edge_age[q_index][f_index] = 0.0; // Python に従って年齢をリセット
                        edge_age[f_index][q_index] = 0.0;

                        connectivity[q_index][r_index] = true;
                        connectivity[r_index][q_index] = true;
                        edge_age[q_index][r_index] = 0.0;
                        edge_age[r_index][q_index] = 0.0;

                        connectivity[r_index][f_index] = true;
                        connectivity[f_index][r_index] = true;
                        edge_age[r_index][f_index] = 0.0;
                        edge_age[f_index][r_index] = 0.0;

                        // (iv) q と f のエラーを係数 alpha で減少させる (Python に従う)
                        neurons_error[q_index] *= 1.0 - ALPHA;
                        neurons_error[f_index] *= 1.0 - ALPHA;

                        // (v) r のエラーを設定する (更新されたエラーを使用して Python の式を使用)
                        neurons_error[r_index] =
                            neurons_error[q_index] + neurons_error[f_index] * 0.5;
                    } else {
                        eprintln!("Error: No free slot found for new neuron r."); // "エラー: 新しいニューロン r のための空きスロットが見つかりません。"
                    }
                } // else: ノード q には隣接ニューロンがなかった
            } // else: 既存のニューロンが見つからなかった
        } // else if i % LAMBDA_VALUE == 0: 最大容量に達した

        // ステップ 9: 既存のすべてのノードのエラーを係数 beta で減少させる
        for k in 0..MAX_N {
            if neurons_exist[k] {
                neurons_error[k] *= 1.0 - BETA;
            }
        }

        // ステップ 10: 終了条件を確認 (最大イテレーション数)
        // プロットは省略。

        // --- プロット処理を追加 ---
        // 定期的に (例: 100 イテレーションごと) または最後にプロット
        if cfg!(debug_assertions) {
            if i % 100 == 0 || i == MAX_ITERATIONS - 1 {
                let plot_filename = format!("{}/gng_state_iter_{:05}.png", plot_dir, i);
                if let Err(e) = plot_gng_state(
                    &plot_filename,
                    i,
                    &cluster_pos, // 元のデータ点もプロットする場合
                    &neurons,
                    &neurons_exist,
                    &connectivity,
                ) {
                    eprintln!("Failed to plot GNG state at iteration {}: {}", i, e); // "イテレーション {} での GNG 状態のプロットに失敗しました: {}"
                }

                let active_count = neurons_exist.iter().filter(|&&e| e).count();
                println!("Iteration {}: Active neurons = {}", i, active_count); // "イテレーション {}: アクティブなニューロン = {}"
            }
        }
        // --- プロット処理の終わり ---
    } // メインループ終了

    println!("Simulation finished after {} iterations.", MAX_ITERATIONS); // "シミュレーションは {} イテレーション後に終了しました。"
}

// --- GNG 状態をプロットする関数 ---
fn plot_gng_state(
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. データを生成
    let img = image::open("sample_image_2.png")?;
    let cluster_pos = extract_data_from_image(&img)?;

    // 2. GNG アルゴリズムを実行
    run_gng(cluster_pos);
    Ok(())
}
