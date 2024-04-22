use std::{io, f32};
use rand::Rng;
use rand_distr::{Poisson, Distribution};
use std::thread;
use std::time::Duration;
use std::process;

type Matrix = Vec<Vec<f32>>;

#[derive(Debug)]
struct Instance {
    n: usize,
    epsilon: f32,
    delta: f32,
    time_limit: u64,
    matrix: Matrix,
    coef: f32,
}

#[derive(Debug)]
struct Helper {
    upper_bound: f32,
    row_bounds: Vec<f32>,
    deep_bound: Matrix,
    depth: usize,
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

fn log_sub_exp(x: f32, y: f32) -> f32 {
    if !x.is_finite() {
        y
    } else if !y.is_finite() {
        x
    } else {
        let z = f32::max(x, y);
        z + ((x - z).exp() - (y - z).exp()).ln()
    }
}

fn bernoulli(instance: &Instance, weights: &Matrix, helper: &Helper) -> bool {
    let mut rng = rand::thread_rng();
    let u = rng.gen::<f32>();
    let x = draw_sample(instance, weights, helper);
    if u.ln() < x {
        true
    } else {
        false
    }
}

fn var_bernoulli(instance: &Instance, weights: &Matrix, helper: &Helper) -> f32 {
    let mut rng = rand::thread_rng();
	if rng.gen::<f32>() < 0.5 {
        return 0.0;
    }
	let u = rng.gen::<f32>();
	let x1 = draw_sample(instance, weights, helper);
	let x2 = draw_sample(instance, weights, helper);
	if u.ln() <= 2.0 * log_sub_exp(x1.max(x2), x1.min(x2)) {
        1.0
    } else {
        0.0
    }
}

fn gbas(k: u64, instance: &Instance, weights: &Matrix, helper: &Helper) -> f32 {
	let mut r: f32 = 0.0;
	let mut cnt: u64 = 0;
    let mut rng = rand::thread_rng();

	loop {
		if bernoulli(instance, weights, helper) {
			cnt += 1;
		}
		r -= (1.0 - rng.gen::<f32>()).ln(); // r += Exp(1)
		if cnt == k {
            break;
        }
	}
	(k as f32 + 2.0) / r
}

fn psi(s: f32) -> f32 {
	if s >= 0.0 {
        (1.0 + s + s * s / 2.).ln()
    } else {
	    -((1.0 - s + s * s / 2.).ln())
    }
}

fn ltsa(n: u64, cc: f32, mu0: f32, epsilon: f32, epsilon0: f32, instance: &Instance, weights: &Matrix, helper: &Helper) -> f32 {
	let mut mu: f32 = 0.0;
	let mu0t: f32 = mu0 / (1.0 - epsilon0 * epsilon0);
	let alpha: f32 = epsilon / (cc * mu0t);
	for _ in 0..n {
        let sample = draw_sample(instance, weights, helper);
		let x: f32 = if sample.is_finite() {
            sample.exp()
        } else {
            0.0
        };
		let w: f32 = mu0t + 1.0 / alpha * psi(alpha * (x - mu0t));
        assert!(x.is_finite());
		assert!(w.is_finite());
		mu += w / (n as f32);
	}
	mu
}

fn estimate(instance: &Instance, weights: &Matrix, helper: &Helper) -> f32 {
	let k: u64 = ((2.0 * instance.epsilon.powf(-2./3.) * (6. / instance.delta).ln()).ceil() + 0.1) as u64;
	println!("GBAS... {}", k);
	let mu0: f32 = gbas(k, instance, weights, helper);
	println!("Bad estimate: {}", (mu0.ln() + helper.upper_bound + instance.coef));

	let poi = Poisson::new(2. * (3. / instance.delta).ln() / (instance.epsilon * mu0)).unwrap();
    let mut rng = rand::thread_rng();
	let n: u64 = (poi.sample(&mut rng) + 0.1) as u64;
	let mut a: f32 = 0.0;
	println!("Bern(Var)... {}", n);
	for _ in 0..n {
		a += var_bernoulli(instance, weights, helper);
	}
	let c1: f32 = 2.0 * (3.0 / instance.delta).ln();
	let cc: f32 = (a / c1 + 0.5 + (a / c1 + 0.25).sqrt()) * (1. + instance.epsilon.powf(1./3.)) * (1. + instance.epsilon.powf(1./3.)) * instance.epsilon / mu0;
	let n: u64 = (0.1 + (2. / instance.epsilon / instance.epsilon * (6. / instance.delta).ln() * cc / (1. - instance.epsilon.powf(1./3.))).ceil()) as u64;

	println!("LTSA... {}", n);
	ltsa(n, cc, mu0, instance.epsilon, instance.epsilon.powf(1.0 / 3.0), instance, weights, helper)
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
    let time_limit: u64 = args[3].parse().unwrap();

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

    let mut coef: f32 = 0.0;
    for _ in 0..100 {
        for i in 0..n {
            let row_sum: f32 = matrix[i].iter().sum();
            matrix[i] = matrix[i].iter().map(|x| x / row_sum).collect();
            coef += row_sum.ln();
        }
        for j in 0..n {
            let col_sum: f32 = (0..n).map(|i| matrix[i][j]).sum();
            for i in 0..n {
                matrix[i][j] /= col_sum;
            }
            coef += col_sum.ln();
        }
    }

    Instance { n, epsilon, delta, time_limit, matrix, coef }
}

fn preprocess(instance: &Instance) -> (Matrix, Helper) {
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

    let depth: usize = 16.min(instance.n - 1);

    let mut row_bounds: Vec<f32> = Vec::with_capacity(instance.n);
    for i in 0..instance.n {
        let mut line = instance.matrix[i][depth..].to_vec();
        line.sort_by(|x, y| y.partial_cmp(x).unwrap());
        row_bounds.push((1..=instance.n - depth).map(|j| line[j - 1] * (row_bound[j] - row_bound[j - 1])).sum::<f32>().ln());
    }

    let mut deep_bound: Matrix = Vec::with_capacity(instance.n);
    for i in 0..instance.n {
        deep_bound.push(Vec::with_capacity(1<<depth));
        match i {
            0 => {
                for _ in 0..(1<<depth) {
                    deep_bound[i].push(f32::NEG_INFINITY);
                }
                deep_bound[i][0] = row_bounds[0];
                for j in 0..depth {
                    deep_bound[i][1<<j] = instance.matrix[i][j].ln();
                }
            },
            _ => {
                for mask in 0..(1<<depth) {
                    let prev = deep_bound[i - 1][mask];
                    deep_bound[i].push(prev + row_bounds[i]);
                }
                for mask in 1..(1<<depth) {
                    for j in 0..depth {
                        if (mask & (1<<j)) > 0 {
                            let prev = deep_bound[i - 1][mask ^ (1<<j)] + instance.matrix[i][j].ln();
                            let cur = deep_bound[i][mask];
                            deep_bound[i][mask] = log_sum_exp(prev, cur);
                        }
                    }
                }
            }
        }
    }

    (weights, Helper {upper_bound: deep_bound[instance.n - 1][(1<<depth) - 1], row_bounds, deep_bound, depth })
}

fn draw_sample(instance: &Instance, weights: &Matrix, helper: &Helper) -> f32 {
    let mut importance_weight: f32 = 0.0;
    let mut not_used = vec![1.0; instance.n];
    let mut rng = rand::thread_rng();

	// Exact sampling part O(n * depth)
	let depth = helper.depth;
    let mut subset = (1<<depth) - 1;
	for i in (0..instance.n).rev() {
		if subset == 0 {
			importance_weight -= helper.row_bounds[i];
		} else {
			if i > 0 {
                let foo = rng.gen::<f32>();
				let lhs = helper.deep_bound[i - 1][subset] + helper.row_bounds[i];
                if lhs.is_infinite() || foo >= (lhs - helper.deep_bound[i][subset]).exp() {
					not_used[i] = 0.0;
					let mut max_j: usize = instance.n + 1;
					let mut max_weight: f32 = f32::NEG_INFINITY;
					for j in 0..depth {
						if (subset & (1<<j)) > 0 && !is_zero(instance.matrix[i][j]) {
							let weight = -f32::ln(-f32::ln(rng.gen::<f32>())) + instance.matrix[i][j].ln() + helper.deep_bound[i - 1][subset ^ (1<<j)];
							if max_j == instance.n + 1 || weight > max_weight {
								max_j = j;
								max_weight = weight;
							}
						}
					}
					if max_j == instance.n + 1 {
                        return f32::NEG_INFINITY;
                    }
					subset ^= 1<<max_j;
				} else {
					importance_weight -= helper.row_bounds[i];
				}
			} else {
                let isubset = subset as isize;
				if isubset != (isubset & -isubset) { // at most one 1 bit remaining
					panic!("Exact sampling failed: {}", subset);
				}
				not_used[i] = 0.0;
			}
		}
	}

	if instance.n == depth {
        return 0.0;
    }

    for j in depth..instance.n {
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
    let (weights, helper) = preprocess(&instance);

    thread::spawn(move || {
        thread::sleep(Duration::from_secs(instance.time_limit));
        println!("timeout");
        process::exit(0);
    });

    println!("{}", helper.upper_bound + estimate(&instance, &weights, &helper).ln() + instance.coef);
}