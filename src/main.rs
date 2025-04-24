mod utils;

use nalgebra::{Point2, Vector2};
use rand::Rng;
use rand::rngs::ThreadRng;
use std::cmp::Ordering;
use std::collections::HashMap; // HashMap をインポート
use std::vec::Vec;

use utils::{extract_data_from_image, plot_gng_state}; // plot_gng_state は別途修正が必要

// --- 定数 ---
const LR_S1: f64 = 0.5; // 勝者ニューロンの学習率
const LR_S2: f64 = 0.002; // 隣接ニューロンの学習率
const BETA: f64 = 0.002; // 全ニューロンのエラー減衰率
const ALPHA: f64 = 0.5; // 新規ノード挿入時のエラー減衰率
const MAX_N: usize = 500; // 最大ニューロン数
const TH_AGE: f64 = 2.0; // エッジの最大年齢
const LAMBDA_VALUE: usize = 8; // 新規ノード挿入の間隔 (ステップ数)
const MAX_ITERATIONS: usize = 1000; // 最大イテレーション数を定義

// --- GNG 状態構造体 (変更) ---
struct GngState {
    neurons: Vec<Point2<f64>>,
    neurons_error: Vec<f64>,
    neurons_exist: Vec<bool>,
    // connectivity と edge_age を統合
    edges: Vec<HashMap<usize, f64>>, // キー: 隣接ノードインデックス, 値: エッジ年齢
    active_indices: Vec<usize>, // アクティブなインデックスのリストを追加
}

impl GngState {
    // new を修正
    fn new(max_n: usize) -> Self {
        GngState {
            neurons: vec![Point2::origin(); max_n],
            neurons_error: vec![0.0; max_n],
            neurons_exist: vec![false; max_n],
            // edges を初期化
            edges: vec![HashMap::new(); max_n],
            active_indices: Vec::new(), // 空で初期化
        }
    }

    // 修正: リストへのスライスを返す
    fn get_active_neuron_indices(&self) -> &[usize] {
        &self.active_indices
    }

    // 修正: リストの長さを使う
    fn get_active_neuron_count(&self) -> usize {
        self.active_indices.len()
    }

    fn find_first_inactive_neuron(&self) -> Option<usize> {
        self.neurons_exist.iter().position(|&exists| !exists)
    }
}

// --- GNG アルゴリズムの各ステップ ---

// ステップ 0: 初期化 (修正)
fn initialize_gng(state: &mut GngState, rng: &mut ThreadRng) {
    state.neurons[0] = Point2::new(rng.gen_range(0.0..500.0), rng.gen_range(0.0..500.0));
    state.neurons[1] = Point2::new(rng.gen_range(0.0..500.0), rng.gen_range(0.0..500.0));
    state.neurons_exist[0] = true;
    state.neurons_exist[1] = true;
    state.active_indices.push(0); // アクティブリストに追加
    state.active_indices.push(1); // アクティブリストに追加
    // edges を使って初期接続を追加 (年齢は 0.0)
    state.edges[0].insert(1, 0.0);
    state.edges[1].insert(0, 0.0);
}

// ステップ 1: ランダムな入力ベクトル v を取得 (呼び出し側で実行)

// ステップ 2: 勝者 s1 と 2 番目の勝者 s2 を見つける (アロケーション削減版)
fn find_bmus(state: &GngState, v: &Point2<f64>) -> Option<(usize, usize, f64)> {
    let active_indices = state.get_active_neuron_indices(); // スライス (&[usize]) を取得
    if active_indices.len() < 2 {
        return None;
    }

    let mut s1_index = usize::MAX;
    let mut s2_index = usize::MAX;
    let mut min_dist_sq = f64::INFINITY;
    let mut second_min_dist_sq = f64::INFINITY;

    for &idx in active_indices { // スライスを直接イテレート
        let dist_sq = nalgebra::distance_squared(&state.neurons[idx], v);

        if dist_sq < min_dist_sq {
            // 現在の s1 を s2 に降格
            second_min_dist_sq = min_dist_sq;
            s2_index = s1_index;
            // 新しい s1 を設定
            min_dist_sq = dist_sq;
            s1_index = idx;
        } else if dist_sq < second_min_dist_sq {
            // 新しい s2 を設定
            second_min_dist_sq = dist_sq;
            s2_index = idx;
        }
    }

    // usize::MAX は初期値なので、有効なインデックスが設定されているか確認
    if s1_index != usize::MAX && s2_index != usize::MAX {
        Some((s1_index, s2_index, min_dist_sq))
    } else {
        // アクティブノードが2つ未満だった場合など
        None // またはエラー処理
    }
}

// ステップ 3 & 4: エラー加算とニューロン更新 (修正)
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

    // s1 の直接の隣接ニューロンを更新 (edges を使用)
    // 隣接ノードのインデックスリストを先に取得（借用チェッカ対策）
    let neighbor_indices: Vec<usize> = state.edges[s1_index].keys().cloned().collect();
    for k in neighbor_indices {
        // state.neurons[k] を変更するため、可変参照が必要
        let neighbor_update_vector: Vector2<f64> = (v - state.neurons[k]) * LR_S2;
        state.neurons[k] += neighbor_update_vector;
    }
}

// ステップ 5 & 6: エッジの作成/リセットと年齢インクリメント (修正)
fn update_edges(state: &mut GngState, s1_index: usize, s2_index: usize) {
    // ステップ 5: エッジの年齢をリセットするか、s1 と s2 の間にエッジを作成 (edges を使用)
    // 存在確認と挿入/更新を同時に行う
    state.edges[s1_index].insert(s2_index, 0.0);
    state.edges[s2_index].insert(s1_index, 0.0);

    // ステップ 6: s1 に接続されているすべてのエッジの年齢をインクリメント (edges を使用)
    // s1 の隣接ノードリストを取得
    let neighbors_of_s1: Vec<usize> = state.edges[s1_index].keys().cloned().collect();
    for &k in &neighbors_of_s1 {
        // s1 -> k のエッジ年齢をインクリメント
        if let Some(age) = state.edges[s1_index].get_mut(&k) {
            *age += 1.0;
        }
        // k -> s1 のエッジ年齢をインクリメント
        if let Some(age) = state.edges[k].get_mut(&s1_index) {
            *age += 1.0;
        }
    }
}

// ステップ 7: 古いエッジと孤立ノードを削除 (修正)
fn remove_old_edges_and_nodes(state: &mut GngState) {
    // --- 古いエッジの削除 ---
    let mut edges_to_remove = Vec::new();
    // get_active_neuron_indices() は &[] を返すので、コピーが必要な場合がある
    // ここではイミュータブルな参照でループするので問題ない
    let current_active_indices_for_edge_check = state.get_active_neuron_indices().to_vec(); // 削除中にリストが変わる可能性があるのでコピー
    for &r in &current_active_indices_for_edge_check {
        // state.edges[r] をイテレート
        for (&c, &age) in &state.edges[r] {
            // 重複削除を防ぐため、r < c のペアのみをリストに追加
            if r < c && age > TH_AGE {
                edges_to_remove.push((r, c));
            }
        }
    }

    // 見つかった古いエッジを削除 (edges から削除)
    for (r, c) in &edges_to_remove {
        state.edges[*r].remove(c);
        state.edges[*c].remove(r);
    }

    // --- 孤立ノードの削除 ---
    let mut nodes_to_remove = Vec::new();
    // 再度アクティブなノードリストを使う (削除中にリストが変わる可能性があるのでコピー)
    let current_active_indices_for_node_check = state.get_active_neuron_indices().to_vec();
    for &k in &current_active_indices_for_node_check {
        // ノード k が存在し、エッジが空かチェック
        if state.neurons_exist[k] && state.edges[k].is_empty() {
            nodes_to_remove.push(k);
        }
    }

    // 見つかった孤立ノードを削除
    for k in nodes_to_remove {
        // ノードが存在することを確認
        if state.neurons_exist[k] {
            state.neurons_exist[k] = false;
            state.neurons_error[k] = 0.0;
            // edges[k] は既に空のはずなのでクリア不要

            // アクティブリストから削除 (swap_remove を使用)
            if let Some(pos) = state.active_indices.iter().position(|&idx| idx == k) {
                state.active_indices.swap_remove(pos);
            }
        }
    }
}

// ステップ 8: 新しいノードを挿入 (修正)
fn insert_node(state: &mut GngState) {
    // (i) 最大エラーを持つノード q を見つける
    // get_active_neuron_indices() は &[] を返すので、コピーが必要な場合がある
    // ここではイミュータブルな参照でループするので問題ない
    let q_index_option = state.get_active_neuron_indices() // スライスを使用
        .iter() // イテレータを取得
        .copied() // &usize を usize に変換
        .max_by(|&idx_a, &idx_b| {
            state.neurons_error[idx_a]
                .partial_cmp(&state.neurons_error[idx_b])
                .unwrap_or(Ordering::Equal)
        });

    if let Some(q_index) = q_index_option {
        // (ii) q の隣接ニューロン f の中で最もエラーを持つものを見つける (edges を使用)
        let mut f_index_option: Option<usize> = None;
        let mut max_neighbor_error = -1.0;

        // state.edges[q_index] をイテレート
        for (&k, _) in &state.edges[q_index] {
            // k は q の隣接ノードのインデックス
            let neighbor_error = state.neurons_error[k];
            if neighbor_error > max_neighbor_error {
                max_neighbor_error = neighbor_error;
                f_index_option = Some(k);
            }
        }

        if let Some(f_index) = f_index_option {
            // 新しいノード r のインデックスを見つける (変更なし)
            if let Some(r_index) = state.find_first_inactive_neuron() {
                let q_pos = state.neurons[q_index];
                let f_pos = state.neurons[f_index];

                // 新しいノード r を q と f の中間点に挿入 (変更なし)
                state.neurons_exist[r_index] = true;
                state.active_indices.push(r_index); // アクティブリストに追加
                state.neurons[r_index] = nalgebra::center(&q_pos, &f_pos);

                // (iii) エッジ q-f を削除し、エッジ q-r と r-f を追加 (edges を使用)
                state.edges[q_index].remove(&f_index);
                state.edges[f_index].remove(&q_index);

                state.edges[q_index].insert(r_index, 0.0);
                state.edges[r_index].insert(q_index, 0.0);

                state.edges[r_index].insert(f_index, 0.0);
                state.edges[f_index].insert(r_index, 0.0);

                // (iv) q と f のエラーを係数 alpha で減少させる (変更なし)
                state.neurons_error[q_index] *= 1.0 - ALPHA;
                state.neurons_error[f_index] *= 1.0 - ALPHA;

                // (v) r のエラーを設定する (変更なし)
                state.neurons_error[r_index] =
                    (state.neurons_error[q_index] + state.neurons_error[f_index]) * 0.5;
            } else {
                eprintln!("Error: No free slot found for new neuron r.");
            }
        }
    }
}

// ステップ 9: 全ノードのエラーを減衰 (変更なし)
fn decay_errors(state: &mut GngState) {
    for k in 0..MAX_N {
        if state.neurons_exist[k] {
            state.neurons_error[k] *= 1.0 - BETA;
        }
    }
}

// プロット処理 (utils::plot_gng_state の修正が必要だった箇所を修正)
fn plot_state(state: &GngState, iteration: usize, cluster_pos: &Vec<Point2<f64>>, plot_dir: &str) {
    if cfg!(debug_assertions) {
        if iteration % 100 == 0 || iteration == MAX_ITERATIONS - 1 {
            let plot_filename = format!("{}/gng_state_iter_{:05}.png", plot_dir, iteration);
            // plot_gng_state に state.edges を渡すように変更
            if let Err(e) = plot_gng_state(
                &plot_filename,
                iteration,
                cluster_pos,
                &state.neurons,
                &state.neurons_exist,
                &state.edges, // connectivity の代わりに edges を渡す
            ) {
                eprintln!("Failed to plot GNG state at iteration {}: {}", iteration, e);
            }
            // eprintln! 警告を削除

            let active_count = state.get_active_neuron_count(); // 修正された関数を使用
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
                if i % LAMBDA_VALUE == 0 && state.get_active_neuron_count() < MAX_N { // 修正された関数を使用
                    insert_node(&mut state);
                }
                // ステップ 9: 全ノードのエラーを減衰
                decay_errors(&mut state);
                // ステップ 10: プロット (条件付き) - コメントアウトを解除
                plot_state(&state, i, &cluster_pos, plot_dir); // plot_state を呼び出すように修正
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

// main 関数 (変更なし)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. データを生成
    let img = image::open("sample_image.png")?;
    let cluster_pos = extract_data_from_image(&img)?;

    // 2. GNG アルゴリズムを実行
    run_gng(cluster_pos);
    Ok(())
}
