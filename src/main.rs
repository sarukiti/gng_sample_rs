mod utils;

use nalgebra::{Point2, Vector2};
use rand::Rng;
use rand::rngs::ThreadRng;
use std::cmp::Ordering;
use std::vec::Vec;

use utils::{extract_data_from_image, plot_gng_state};

// --- 定数 ---
const LR_S1: f64 = 0.5; // 勝者ニューロンの学習率
const LR_S2: f64 = 0.002; // 隣接ニューロンの学習率
const BETA: f64 = 0.002; // 全ニューロンのエラー減衰率
const ALPHA: f64 = 0.5; // 新規ノード挿入時のエラー減衰率
const MAX_N: usize = 500; // 最大ニューロン数
const TH_AGE: f64 = 2.0; // エッジの最大年齢
const LAMBDA_VALUE: usize = 8; // 新規ノード挿入の間隔 (ステップ数)
const MAX_ITERATIONS: usize = 1000; // 最大イテレーション数を定義

// --- GNG 状態構造体 ---
struct GngState {
    neurons: Vec<Point2<f64>>,
    neurons_error: Vec<f64>,
    neurons_exist: Vec<bool>,
    connectivity: Vec<Vec<bool>>,
    edge_age: Vec<Vec<f64>>,
}

impl GngState {
    fn new(max_n: usize) -> Self {
        GngState {
            neurons: vec![Point2::origin(); max_n],
            neurons_error: vec![0.0; max_n],
            neurons_exist: vec![false; max_n],
            connectivity: vec![vec![false; max_n]; max_n],
            edge_age: vec![vec![0.0; max_n]; max_n],
        }
    }

    fn get_active_neuron_indices(&self) -> Vec<usize> {
        (0..MAX_N).filter(|&idx| self.neurons_exist[idx]).collect()
    }

    fn get_active_neuron_count(&self) -> usize {
        self.neurons_exist.iter().filter(|&&e| e).count()
    }

    fn find_first_inactive_neuron(&self) -> Option<usize> {
        self.neurons_exist.iter().position(|&exists| !exists)
    }
}

// --- GNG アルゴリズムの各ステップ ---

// ステップ 0: 初期化
fn initialize_gng(state: &mut GngState, rng: &mut ThreadRng) {
    state.neurons[0] = Point2::new(rng.gen_range(0.0..500.0), rng.gen_range(0.0..500.0));
    state.neurons[1] = Point2::new(rng.gen_range(0.0..500.0), rng.gen_range(0.0..500.0));
    state.neurons_exist[0] = true;
    state.neurons_exist[1] = true;
    state.connectivity[0][1] = true;
    state.connectivity[1][0] = true;
    // edge_age はデフォルトで 0.0 なので初期化不要
}

// ステップ 1: ランダムな入力ベクトル v を取得 (呼び出し側で実行)

// ステップ 2: 勝者 s1 と 2 番目の勝者 s2 を見つける
fn find_bmus(state: &GngState, v: &Point2<f64>) -> Option<(usize, usize, f64)> {
    let active_indices = state.get_active_neuron_indices();
    if active_indices.len() < 2 {
        return None; // アクティブなニューロンが2未満の場合は勝者を見つけられない
    }

    let mut distances: Vec<(usize, f64)> = active_indices
        .iter()
        .map(|&idx| {
            let dist_sq = nalgebra::distance_squared(&state.neurons[idx], v);
            (idx, dist_sq)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let s1_index = distances[0].0;
    let s2_index = distances[1].0;
    let s1_dist_sq = distances[0].1;

    Some((s1_index, s2_index, s1_dist_sq))
}

// ステップ 3 & 4: エラー加算とニューロン更新
fn update_error_and_neurons(
    state: &mut GngState,
    s1_index: usize,
    s1_dist_sq: f64,
    v: &Point2<f64>,
) {
    // ステップ 3: 勝者 s1 の累積エラーに二乗誤差を加算
    state.neurons_error[s1_index] += s1_dist_sq;

    // ステップ 4: s1 とそのトポロジカルな隣接ニューロンの参照ベクトルを更新
    // s1 を更新
    let s1_update_vector: Vector2<f64> = (v - state.neurons[s1_index]) * LR_S1;
    state.neurons[s1_index] += s1_update_vector;

    // s1 の直接の隣接ニューロンを更新
    for k in 0..MAX_N {
        if state.connectivity[s1_index][k] {
            let neighbor_update_vector: Vector2<f64> = (v - state.neurons[k]) * LR_S2;
            state.neurons[k] += neighbor_update_vector;
        }
    }
}

// ステップ 5 & 6: エッジの作成/リセットと年齢インクリメント
fn update_edges(state: &mut GngState, s1_index: usize, s2_index: usize) {
    // ステップ 5: エッジの年齢をリセットするか、s1 と s2 の間にエッジを作成
    if !state.connectivity[s1_index][s2_index] {
        state.connectivity[s1_index][s2_index] = true;
        state.connectivity[s2_index][s1_index] = true;
    }
    state.edge_age[s1_index][s2_index] = 0.0;
    state.edge_age[s2_index][s1_index] = 0.0;

    // ステップ 6: s1 に接続されているすべてのエッジの年齢をインクリメント
    for k in 0..MAX_N {
        if k != s1_index && state.connectivity[s1_index][k] {
            state.edge_age[s1_index][k] += 1.0;
            state.edge_age[k][s1_index] += 1.0;
        }
    }
}

// ステップ 7: 古いエッジと孤立ノードを削除 (最適化案1適用)
fn remove_old_edges_and_nodes(state: &mut GngState) {
    let active_indices = state.get_active_neuron_indices(); // アクティブなノードのインデックスを取得
    let num_active = active_indices.len();

    // --- 古いエッジの削除 ---
    let mut edges_to_remove = Vec::new();
    // アクティブなノードのペアのみをチェック
    for i in 0..num_active {
        let r = active_indices[i];
        // i+1 から始めることで重複チェックと自己ループを避ける
        for j in (i + 1)..num_active {
            let c = active_indices[j];
            // 接続が存在し、かつ年齢が閾値を超えているかチェック
            // connectivity は対称なので片方だけチェックすれば良い
            if state.connectivity[r][c] && state.edge_age[r][c] > TH_AGE {
                edges_to_remove.push((r, c));
            }
        }
    }

    // 見つかった古いエッジを削除
    for (r, c) in &edges_to_remove { // イミュータブル参照で十分
        state.connectivity[*r][*c] = false;
        state.connectivity[*c][*r] = false;
        state.edge_age[*r][*c] = 0.0;
        state.edge_age[*c][*r] = 0.0;
    }

    // --- 孤立ノードの削除 ---
    // エッジ削除後に孤立した可能性のあるノードをチェック
    let mut nodes_to_remove = Vec::new();
    // 再度アクティブなノードリストを使う
    for &k in &active_indices { // k はアクティブなノードのインデックス
        // 削除されたエッジの影響を考慮するため、現在の接続状態を確認
        // k に接続しているアクティブなノードが存在するかチェック
        // connectivity[k] を直接チェックする方が効率的
        let is_still_connected = (0..MAX_N).any(|j| state.connectivity[k][j]);

        // 接続がなければ削除候補リストに追加
        // ただし、そのノードがまだ存在しているか確認 (他の処理で削除されていないか)
        if !is_still_connected && state.neurons_exist[k] {
            nodes_to_remove.push(k);
        }
    }

    // 見つかった孤立ノードを削除
    for k in nodes_to_remove {
        // ノードが存在することを確認 (他の処理で削除されていないか)
        if state.neurons_exist[k] {
            state.neurons_exist[k] = false;
            state.neurons_error[k] = 0.0;
            // 関連する接続情報はエッジ削除処理でクリアされているか、
            // もしくは接続がないためクリア不要なはず。
            // 念のためクリアする場合 (通常は不要):
            // for j in 0..MAX_N {
            //     state.connectivity[k][j] = false;
            //     state.connectivity[j][k] = false;
            //     state.edge_age[k][j] = 0.0;
            //     state.edge_age[j][k] = 0.0;
            // }
        }
    }
}

// ステップ 8: 新しいノードを挿入
fn insert_node(state: &mut GngState) {
    // (i) 最大エラーを持つノード q を見つける
    let q_index_option = (0..MAX_N)
        .filter(|&idx| state.neurons_exist[idx])
        .max_by(|&idx_a, &idx_b| {
            state.neurons_error[idx_a]
                .partial_cmp(&state.neurons_error[idx_b])
                .unwrap_or(Ordering::Equal)
        });

    if let Some(q_index) = q_index_option {
        // (ii) q の隣接ニューロン f の中で最もエラーを持つものを見つける
        let mut f_index_option: Option<usize> = None;
        let mut max_neighbor_error = -1.0; // 負の値で初期化

        for k in 0..MAX_N {
            if k != q_index && state.connectivity[q_index][k] {
                let neighbor_error = state.neurons_error[k]; // エラー値で比較
                if neighbor_error > max_neighbor_error {
                    max_neighbor_error = neighbor_error;
                    f_index_option = Some(k);
                }
            }
        }


        if let Some(f_index) = f_index_option {
            // 新しいノード r のインデックスを見つける
            if let Some(r_index) = state.find_first_inactive_neuron() {
                let q_pos = state.neurons[q_index]; // q_pos は後で使うのでここで取得
                let f_pos = state.neurons[f_index];

                // 新しいノード r を q と f の中間点に挿入
                state.neurons_exist[r_index] = true;
                state.neurons[r_index] = nalgebra::center(&q_pos, &f_pos);

                // (iii) エッジ q-f を削除し、エッジ q-r と r-f を追加
                state.connectivity[q_index][f_index] = false;
                state.connectivity[f_index][q_index] = false;
                state.edge_age[q_index][f_index] = 0.0;
                state.edge_age[f_index][q_index] = 0.0;

                state.connectivity[q_index][r_index] = true;
                state.connectivity[r_index][q_index] = true;
                state.edge_age[q_index][r_index] = 0.0;
                state.edge_age[r_index][q_index] = 0.0;

                state.connectivity[r_index][f_index] = true;
                state.connectivity[f_index][r_index] = true;
                state.edge_age[r_index][f_index] = 0.0;
                state.edge_age[f_index][r_index] = 0.0;

                // (iv) q と f のエラーを係数 alpha で減少させる
                state.neurons_error[q_index] *= 1.0 - ALPHA;
                state.neurons_error[f_index] *= 1.0 - ALPHA;

                // (v) r のエラーを設定する
                state.neurons_error[r_index] =
                    (state.neurons_error[q_index] + state.neurons_error[f_index]) * 0.5;
            } else {
                eprintln!("Error: No free slot found for new neuron r.");
            }
        }
    }
}

// ステップ 9: 全ノードのエラーを減衰
fn decay_errors(state: &mut GngState) {
    for k in 0..MAX_N {
        if state.neurons_exist[k] {
            state.neurons_error[k] *= 1.0 - BETA;
        }
    }
}

// プロット処理
fn plot_state(state: &GngState, iteration: usize, cluster_pos: &Vec<Point2<f64>>, plot_dir: &str) {
    if cfg!(debug_assertions) {
        if iteration % 100 == 0 || iteration == MAX_ITERATIONS - 1 {
            let plot_filename = format!("{}/gng_state_iter_{:05}.png", plot_dir, iteration);
            if let Err(e) = plot_gng_state(
                &plot_filename,
                iteration,
                cluster_pos,
                &state.neurons,
                &state.neurons_exist,
                &state.connectivity,
            ) {
                eprintln!("Failed to plot GNG state at iteration {}: {}", iteration, e);
            }

            let active_count = state.get_active_neuron_count();
            println!("Iteration {}: Active neurons = {}", iteration, active_count);
        }
    }
}

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
    std::fs::create_dir_all(plot_dir).expect("Failed to create plot directory");

    let mut rng = rand::thread_rng();
    let data_len = cluster_pos.len();

    // --- 初期化 (ステップ 0) ---
    let mut state = GngState::new(MAX_N);
    initialize_gng(&mut state, &mut rng);

    // --- メインループ ---
    for i in 1..MAX_ITERATIONS {
        // ステップ 1: ランダムな入力ベクトル v を取得
        let v_index = rng.gen_range(0..data_len);
        let v = cluster_pos[v_index];
        // ステップ 2: 勝者 s1 と 2 番目の勝者 s2 を見つける
        match find_bmus(&state, &v) {
            Some((s1_index, s2_index, s1_dist_sq)) => {
                // ステップ 3 & 4: エラー加算とニューロン更新
                update_error_and_neurons(&mut state, s1_index, s1_dist_sq, &v);
                // ステップ 5 & 6: エッジの作成/リセットと年齢インクリメント
                update_edges(&mut state, s1_index, s2_index);
                // ステップ 7: 古いエッジと孤立ノードを削除
                remove_old_edges_and_nodes(&mut state);
                // ステップ 8: lambda ステップごとに新しいノードを挿入
                if i % LAMBDA_VALUE == 0 && state.get_active_neuron_count() < MAX_N {
                    insert_node(&mut state);
                }
                // ステップ 9: 全ノードのエラーを減衰
                decay_errors(&mut state);
                // ステップ 10: プロット (条件付き)
                // plot_state(&state, i, &cluster_pos, plot_dir);
            }
            None => {
                println!(
                    "Warning: Less than 2 active neurons at iteration {}. Skipping.",
                    i
                );
                continue;
            }
        }
    }

    println!("Simulation finished after {} iterations.", MAX_ITERATIONS);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. データを生成
    let img = image::open("sample_image.png")?;
    let cluster_pos = extract_data_from_image(&img)?;

    // 2. GNG アルゴリズムを実行
    run_gng(cluster_pos);
    Ok(())
}
