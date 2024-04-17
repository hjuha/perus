use std::io;
use std::f32;
use rand;
use rand::Rng;

#[derive(Debug)]
struct Matrix {
    data: Vec<Vec<f32>>,
}

impl Matrix {
    fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i][j]
    }

    pub fn get_row(&self, i: usize, offset: usize) -> &[f32] {
        &self.data[i].as_slice()[offset..]
    }
}

#[derive(Debug)]
struct Instance {
    n: usize,
    epsilon: f32,
    delta: f32,
    time_limit: f32,
    matrix: Matrix,
}

fn close(x: f32, y: f32) -> bool {
    (x - y).abs() < 0.00000001
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

    let mut matrix: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n {
        let mut line = String::new();
        io::stdin().read_line(&mut line).expect("Failed to read line.");
        let split = line.split_whitespace();
        let line: Vec<f32> = split.map(|x| x.parse().unwrap()).collect();
        if line.len() != n {
            panic_input_format();
        }
        matrix.push(line);
    }
    let matrix = Matrix { data: matrix };
    Instance { n: n, epsilon: epsilon, delta: delta, time_limit: time_limit, matrix: matrix }
}

fn preprocess(instance: &Instance) -> Matrix {
    let mut row_bound: Vec<f32> = vec![0.0, f32::consts::E];
    for i in 2..=instance.n {
        let previous = row_bound[i - 1];
        row_bound.push(previous + 1. + 0.5 / previous + 0.6 / previous / previous);
    }
    let row_bound: Vec<f32> = row_bound.iter().map(|x| x / f32::consts::E).collect();
    let mut weights: Vec<Vec<f32>> = Vec::new();
    for i in 0..instance.n {
        let mut weight_row: Vec<f32> = Vec::new();
        for j in 0..instance.n {
            let mut line = Vec::from(instance.matrix.get_row(i,j));
            line.sort_by(|x, y| y.partial_cmp(x).unwrap());
            let divisor = (j + 1..=instance.n).into_iter().map(|k| line[k - j - 1] * (row_bound[k] - row_bound[k - 1])).sum::<f32>();
            
            let weight = if close(instance.matrix.get(i, j), 0.0) {
                0.0
            } else if close(divisor, 0.0) {
                10000000.0
            } else {
                instance.matrix.get(i, j) / divisor
            };
            weight_row.push(weight);
        }
        weights.push(weight_row);
    }
    Matrix { data: weights }
}

fn draw_sample(instance: &Instance, weights: &Matrix) -> f32 {
    let mut importance_weight: f32 = 0.0;
    let mut not_used: Vec<f32> = (0..instance.n).into_iter().map(|_| 1.0).collect();
    let mut rng = rand::thread_rng();
    for j in 0..instance.n {
        let norm: f32 = (0..instance.n).into_iter().map(|i| weights.get(i, j) * not_used[i]).sum();
        if close(norm, 0.0) {
            return -f32::INFINITY;
        }
        let choice: f32 = rng.gen::<f32>() * norm;
        let mut cumulative: f32 = 0.0;
        for i in 0..instance.n {
            cumulative += weights.get(i, j) * not_used[i];
            if choice <= cumulative {
                not_used[i] = 0.0;
                importance_weight += instance.matrix.get(i, j).ln() - (weights.get(i, j) / norm).ln();
                break;
            }
        }
    }
    importance_weight
}

fn main() {
    let instance = read_input();
    let weights = preprocess(&instance);

    let mut count: f32 = 0.0;
    let mut cumulative: f32 = -f32::INFINITY;
    for _ in 0..10000 {
        count += 1.0;
        let sample = draw_sample(&instance, &weights);
        cumulative = log_sum_exp(cumulative, sample);
    }

    println!("{}", cumulative.exp() / count);
}