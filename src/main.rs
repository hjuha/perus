use std::{io, f32};
use rand::Rng;

type Matrix = Vec<Vec<f32>>;

#[derive(Debug)]
struct Instance {
    n: usize,
    epsilon: f32,
    delta: f32,
    time_limit: f32,
    matrix: Matrix,
}

fn is_zero(x: f32) -> bool {
    x.abs() < 0.00000001
}

fn log_sum_exp(x: f32, y: f32) -> f32 {
    if !x.is_finite() {
        y
    } else if !y.is_finite() {
        x
    } else {
        let z = f32::max(x, y);
        z + ((x - z).exp() + (y - z).exp()).ln()
    }
}

fn panic_input_format() {
    let msg = "Give the input in the following format:

    n epsilon delta time_limit
    a_11 a_12 ... a_1n
    a_21 a_22 ... a_2n
    ...
    a_n1 a_n2 ... a_nn
    ";
    panic!("{}", msg);
}

fn read_input() -> Instance {
    let mut line = String::new();
    io::stdin().read_line(&mut line).expect("Failed to read line.");
    let args: Vec<&str> = line.split_whitespace().collect();
    if args.len() != 4 {
        panic_input_format();
    }
    let n: usize = args[0].parse().unwrap();
    let epsilon: f32 = args[1].parse().unwrap();
    let delta: f32 = args[2].parse().unwrap();
    let time_limit: f32 = args[3].parse().unwrap();

    let mut matrix: Matrix = Vec::with_capacity(n);
    for _ in 0..n {
        let mut line = String::new();
        io::stdin().read_line(&mut line).expect("Failed to read line.");
        let line: Vec<f32> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        if line.len() != n {
            panic_input_format();
        }
        matrix.push(line);
    }
    Instance { n, epsilon, delta, time_limit, matrix }
}

fn preprocess(instance: &Instance) -> Matrix {
    let mut row_bound: Vec<f32> = vec![0.0, f32::consts::E];
    for i in 2..=instance.n {
        let previous = row_bound[i - 1];
        row_bound.push(previous + 1. + 0.5 / previous + 0.6 / previous / previous);
    }
    let row_bound: Vec<f32> = row_bound.iter().map(|x| x / f32::consts::E).collect();
    let mut weights: Matrix = Vec::with_capacity(instance.n);
    for i in 0..instance.n {
        let mut weight_row: Vec<f32> = Vec::with_capacity(instance.n);
        for j in 0..instance.n {
            let mut line = instance.matrix[i][j..].to_vec();
            line.sort_by(|x, y| y.partial_cmp(x).unwrap());
            let divisor = (j + 1..=instance.n).map(|k| line[k - j - 1] * (row_bound[k] - row_bound[k - 1])).sum::<f32>();
            
            let weight = if is_zero(instance.matrix[i][j]) {
                0.0
            } else if is_zero(divisor) {
                f32::INFINITY
            } else {
                instance.matrix[i][j] / divisor
            };
            weight_row.push(weight);
        }
        weights.push(weight_row);
    }
    weights
}

fn draw_sample(instance: &Instance, weights: &Matrix) -> f32 {
    let mut importance_weight: f32 = 0.0;
    let mut not_used = vec![1.0; instance.n];
    let mut rng = rand::thread_rng();
    for j in 0..instance.n {
        let norm: f32 = (0..instance.n).map(|i| weights[i][j] * not_used[i]).sum();
        if is_zero(norm) {
            return f32::NEG_INFINITY;
        }
        let choice: f32 = rng.gen::<f32>() * norm;
        let mut cumulative: f32 = 0.0;
        if let Some(i) = (0..instance.n).find(|&i| {
            cumulative += weights[i][j] * not_used[i];
            choice <= cumulative
        }) {
            not_used[i] = 0.0;
            if norm.is_finite() {
                importance_weight += instance.matrix[i][j].ln() - (weights[i][j] / norm).ln();
            } else {
                importance_weight += instance.matrix[i][j].ln();
            }
        }
    }
    importance_weight
}

fn main() {
    let instance = read_input();
    let weights = preprocess(&instance);

    let mut count: f32 = 0.0;
    let mut cumulative: f32 = f32::NEG_INFINITY;
    for _ in 0..1000 {
        count += 1.0;
        let sample = draw_sample(&instance, &weights);
        cumulative = log_sum_exp(cumulative, sample);
    }

    println!("{}", cumulative - count.ln());
}