//! Sub-sample closest approach.
//!
//! The trajectory is discrete, so the samples around the closest one are rotated about z into a vertical plane
//! and fitted with quadratics x'(t) and z(t). The squared distance to the target is then a quartic in t,
//! which is minimised within the fitted time window.

use crate::pymath::{pow, square};
use crate::types::{ClosestPoint, Error, Target, Trajectory, Vector3};

/// Number of samples taken on each side of the closest sample.
const FIT_WIDTH: usize = 2;
const MIN_SAMPLES_FOR_FIT: usize = 3;

/// `a t^2 + b t + c`
#[derive(Debug, Clone, Copy, PartialEq)]
struct Quadratic {
    a: f64,
    b: f64,
    c: f64,
}

impl Quadratic {
    fn eval(&self, t: f64) -> f64 {
        self.a * square(t) + self.b * t + self.c
    }
}

/// Least-squares fit of a quadratic to up to 5 points.
///
/// The time is shifted by its mean and scaled by its half-width before solving, to keep the normal equations
/// well conditioned. The coefficients are then converted back to the unshifted time, as the Python version has them.
fn fit_quadratic(t: &[f64], y: &[f64]) -> Quadratic {
    let n = t.len() as f64;
    let mean = t.iter().sum::<f64>() / n;
    let scale = t
        .iter()
        .map(|v| (v - mean).abs())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);

    // Normal equations for y = alpha u^2 + beta u + gamma with u = (t - mean) / scale
    let mut s = [0.0; 5]; // sums of u^0 .. u^4
    let mut r = [0.0; 3]; // sums of y u^0 .. y u^2
    for (&ti, &yi) in t.iter().zip(y.iter()) {
        let u = (ti - mean) / scale;
        let mut power = 1.0;
        for (k, sum) in s.iter_mut().enumerate() {
            *sum += power;
            if k < 3 {
                r[k] += yi * power;
            }
            power *= u;
        }
    }
    let mut matrix = [
        [s[4], s[3], s[2], r[2]],
        [s[3], s[2], s[1], r[1]],
        [s[2], s[1], s[0], r[0]],
    ];
    let [alpha, beta, gamma] = solve_3x3(&mut matrix);

    let a = alpha / (scale * scale);
    let b = beta / scale - 2.0 * a * mean;
    let c = gamma - (beta / scale) * mean + a * mean * mean;
    Quadratic { a, b, c }
}

/// Gaussian elimination with partial pivoting on an augmented 3x4 matrix.
fn solve_3x3(m: &mut [[f64; 4]; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap_or(col);
        m.swap(col, pivot);
        for row in (col + 1)..3 {
            let factor = m[row][col] / m[col][col];
            for k in col..4 {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut sum = m[row][3];
        for k in (row + 1)..3 {
            sum -= m[row][k] * x[k];
        }
        x[row] = sum / m[row][row];
    }
    x
}

/// Real parts of the three roots of `c3 x^3 + c2 x^2 + c1 x + c0 = 0` (`c3 != 0`).
///
/// The Python version takes `np.roots(...).real`; the real parts of a complex pair are included in the same way.
fn cubic_root_real_parts(c3: f64, c2: f64, c1: f64, c0: f64) -> [f64; 3] {
    let p = c2 / c3;
    let q = c1 / c3;
    let r = c0 / c3;

    let big_q = (p * p - 3.0 * q) / 9.0;
    let big_r = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0;
    let shift = p / 3.0;

    let polish = |x: f64| -> f64 {
        // Newton's method on the original cubic, to remove the rounding error of the closed form.
        let mut x = x;
        for _ in 0..3 {
            let f = ((c3 * x + c2) * x + c1) * x + c0;
            let df = (3.0 * c3 * x + 2.0 * c2) * x + c1;
            if df == 0.0 || !f.is_finite() {
                break;
            }
            let next = x - f / df;
            if !next.is_finite() {
                break;
            }
            x = next;
        }
        x
    };

    if big_r * big_r < big_q * big_q * big_q {
        // Three distinct real roots
        let theta = (big_r / (big_q * big_q * big_q).sqrt()).acos();
        let m = -2.0 * big_q.sqrt();
        let tau = 2.0 * std::f64::consts::PI;
        [
            polish(m * (theta / 3.0).cos() - shift),
            polish(m * ((theta + tau) / 3.0).cos() - shift),
            polish(m * ((theta - tau) / 3.0).cos() - shift),
        ]
    } else {
        // One real root and a complex pair (or a repeated real root)
        let big_a =
            -big_r.signum() * (big_r.abs() + (big_r * big_r - big_q * big_q * big_q).sqrt()).cbrt();
        let big_b = if big_a == 0.0 { 0.0 } else { big_q / big_a };
        let real = polish((big_a + big_b) - shift);
        // The sum of the roots is -p, so the real part of the pair follows from the real root.
        let pair = (-p - real) / 2.0;
        [real, pair, pair]
    }
}

/// Minimise `a4 x^4 + a3 x^3 + a2 x^2 + a1 x + a0` over the closed interval `bounds`.
/// Returns the minimum point and the value there.
fn find_minimum_quartic(coefficients: [f64; 5], bounds: (f64, f64)) -> Result<(f64, f64), Error> {
    let [a4, a3, a2, a1, a0] = coefficients;
    if a4 <= 0.0 {
        return Err(Error::Numerical(
            "the leading coefficient of the quartic is not positive".to_string(),
        ));
    }

    // The minimum within the interval is at an extremum inside it or at one of its ends.
    // Roots of the derivative are the candidates for the extrema.
    let (lower, upper) = bounds;
    let roots = cubic_root_real_parts(4.0 * a4, 3.0 * a3, 2.0 * a2, a1);
    let candidates = [
        clip(roots[0], lower, upper),
        clip(roots[1], lower, upper),
        clip(roots[2], lower, upper),
        lower,
        upper,
    ];

    // NumPy evaluates `x**4` and `x**3` of an array with `pow` and `x**2` as a multiplication.
    let quartic = |x: f64| a4 * pow(x, 4.0) + a3 * pow(x, 3.0) + a2 * (x * x) + a1 * x + a0;
    let mut best = (candidates[0], quartic(candidates[0]));
    for &x in &candidates[1..] {
        let y = quartic(x);
        if y < best.1 || (y.is_nan() && !best.1.is_nan()) {
            best = (x, y);
        }
    }
    Ok(best)
}

/// `np.clip`: NaN stays NaN.
fn clip(x: f64, lower: f64, upper: f64) -> f64 {
    if x < lower {
        lower
    } else if x > upper {
        upper
    } else {
        x
    }
}

fn fit_and_minimise(trajectory: &Trajectory, target: &Target) -> Result<ClosestPoint, Error> {
    let start = trajectory.closest_sample.saturating_sub(FIT_WIDTH);
    let end = (trajectory.closest_sample + FIT_WIDTH + 1).min(trajectory.t.len());
    let t = &trajectory.t[start..end];
    let points = &trajectory.projectile[start..end];

    // The azimuth of the fitting plane is the direction of the sum of the position vectors.
    // Averaging the azimuths of the points would go wrong at the launch point (the origin has no azimuth)
    // and at the wrap-around at +-pi.
    let sum_x: f64 = points.iter().map(|p| p[0]).sum();
    let sum_y: f64 = points.iter().map(|p| p[1]).sum();
    let phi = sum_y.atan2(sum_x);
    let (sin, cos) = phi.sin_cos();

    // Rotate by -phi: x' = x cos(phi) + y sin(phi)
    let x_rotated: Vec<f64> = points.iter().map(|p| p[0] * cos + p[1] * sin).collect();
    let z: Vec<f64> = points.iter().map(|p| p[2]).collect();
    let x_param = fit_quadratic(t, &x_rotated);
    let z_param = fit_quadratic(t, &z);
    let domain = (
        t.iter().copied().fold(f64::INFINITY, f64::min),
        t.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );

    let t_pos = target.position;
    let t_dir = target.velocity;

    let m11 = x_param.a;
    let m12 = x_param.b - t_dir[0] * cos - t_dir[1] * sin;
    let m13 = x_param.c - t_pos[0] * cos - t_pos[1] * sin;
    let m22 = t_dir[0] * sin - t_dir[1] * cos;
    let m23 = t_pos[0] * sin - t_pos[1] * cos;
    let m31 = z_param.a;
    let m32 = z_param.b - t_dir[2];
    let m33 = z_param.c - t_pos[2];

    let a4 = square(m11) + square(m31);
    let a3 = 2.0 * (m11 * m12 + m31 * m32);
    let a2 = 2.0 * m11 * m13 + square(m12) + square(m22) + 2.0 * m31 * m33 + square(m32);
    let a1 = 2.0 * (m12 * m13 + m22 * m23 + m32 * m33);
    let a0 = square(m13) + square(m23) + square(m33);

    // The quartic is the squared distance. The fit can make its minimum slightly negative, so clamp at 0.
    // The quadratics are valid only within the fitted time window: outside it they extrapolate to a trajectory
    // that does not exist (after landing, or after the calculation was cut off).
    let (t_min, sq_distance_min) = find_minimum_quartic([a4, a3, a2, a1, a0], domain)?;
    let distance = sq_distance_min.max(0.0).sqrt();

    let x_at_min = x_param.eval(t_min);
    Ok(ClosestPoint {
        t: t_min,
        distance,
        target_pos: [
            t_pos[0] + t_min * t_dir[0],
            t_pos[1] + t_min * t_dir[1],
            t_pos[2] + t_min * t_dir[2],
        ],
        projectile_pos: [x_at_min * cos, x_at_min * sin, z_param.eval(t_min)],
    })
}

/// Find the closest point between the projectile and the target, without checking the inputs.
///
/// Uses the quadratic approximation near the closest sample. If the trajectory has too few samples to fit
/// a quadratic (the projectile lands immediately after launch), the closest sample itself is returned.
pub(crate) fn find_unchecked(
    trajectory: &Trajectory,
    target: &Target,
) -> Result<ClosestPoint, Error> {
    if trajectory.t.len() < MIN_SAMPLES_FOR_FIT {
        let index = trajectory.closest_sample;
        let p: Vector3 = trajectory.projectile[index];
        let q: Vector3 = trajectory.target[index];
        let squared = square(p[0] - q[0]) + square(p[1] - q[1]) + square(p[2] - q[2]);
        return Ok(ClosestPoint {
            t: trajectory.t[index],
            distance: squared.sqrt(),
            target_pos: q,
            projectile_pos: p,
        });
    }
    fit_and_minimise(trajectory, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn fit_recovers_a_quadratic() {
        let q = Quadratic {
            a: -4.9,
            b: 120.0,
            c: 3.5,
        };
        let t = [10.0, 10.3, 10.5, 11.1, 11.2];
        let y: Vec<f64> = t.iter().map(|&v| q.eval(v)).collect();
        let fitted = fit_quadratic(&t, &y);
        assert_close(fitted.a, q.a, 1e-9);
        assert_close(fitted.b, q.b, 1e-7);
        assert_close(fitted.c, q.c, 1e-6);

        let fitted = fit_quadratic(&t[..3], &y[..3]);
        assert_close(fitted.a, q.a, 1e-8);
    }

    #[test]
    fn fit_is_least_squares() {
        // Symmetric points: the residual of the best fit is orthogonal to 1, t and t^2.
        let t = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = [4.1, 0.9, 0.2, 1.1, 3.9];
        let q = fit_quadratic(&t, &y);
        for power in 0..3 {
            let dot: f64 = t
                .iter()
                .zip(y.iter())
                .map(|(&ti, &yi)| (yi - q.eval(ti)) * ti.powi(power))
                .sum();
            assert_close(dot, 0.0, 1e-12);
        }
    }

    #[test]
    fn cubic_with_three_real_roots() {
        // (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6
        let mut roots = cubic_root_real_parts(1.0, 0.0, -7.0, 6.0);
        roots.sort_by(f64::total_cmp);
        assert_close(roots[0], -3.0, 1e-12);
        assert_close(roots[1], 1.0, 1e-12);
        assert_close(roots[2], 2.0, 1e-12);
    }

    #[test]
    fn cubic_with_a_complex_pair() {
        // (x - 2)(x^2 + 2x + 5): roots 2 and -1 +- 2i
        let roots = cubic_root_real_parts(2.0, 0.0, 2.0, -20.0);
        assert_close(roots[0], 2.0, 1e-12);
        assert_close(roots[1], -1.0, 1e-12);
        assert_close(roots[2], -1.0, 1e-12);
    }

    #[test]
    fn quartic_minimum_inside_the_interval() {
        // (x - 1)^2 (x + 2)^2 + 0.5 x: the lower of the two minima is near x = -2.
        let (x, y) = find_minimum_quartic([1.0, 2.0, -3.0, -3.5, 4.0], (-10.0, 10.0)).unwrap();
        assert!((-2.2..-1.9).contains(&x));
        let f = |x: f64| x.powi(4) + 2.0 * x.powi(3) - 3.0 * x.powi(2) - 3.5 * x + 4.0;
        assert_close(y, f(x), 1e-12);
        assert!(f(x - 1e-4) > y && f(x + 1e-4) > y);
    }

    #[test]
    fn quartic_minimum_at_a_repeated_root() {
        // (x - 3)^4: the derivative has a triple root at 3.
        let (x, y) = find_minimum_quartic([1.0, -12.0, 54.0, -108.0, 81.0], (-10.0, 10.0)).unwrap();
        assert_close(x, 3.0, 1e-3);
        assert_close(y, 0.0, 1e-9);
    }

    #[test]
    fn quartic_minimum_at_the_ends_of_the_interval() {
        // x^4 has its minimum at 0, outside both intervals.
        let (x, y) = find_minimum_quartic([1.0, 0.0, 0.0, 0.0, 0.0], (1.0, 2.0)).unwrap();
        assert_eq!((x, y), (1.0, 1.0));
        let (x, y) = find_minimum_quartic([1.0, 0.0, 0.0, 0.0, 0.0], (-3.0, -2.0)).unwrap();
        assert_eq!((x, y), (-2.0, 16.0));
    }

    #[test]
    fn quartic_local_minimum_outside_the_interval_is_not_used() {
        // Minima at -2 (lower) and 1; restricted to [0, 5] the answer is the one at 1.
        let (x, _) = find_minimum_quartic([1.0, 2.0, -3.0, -3.5, 4.0], (0.0, 5.0)).unwrap();
        assert!((0.8..1.2).contains(&x));
    }

    #[test]
    fn quartic_needs_a_positive_leading_coefficient() {
        assert!(find_minimum_quartic([0.0, 1.0, 1.0, 1.0, 1.0], (0.0, 1.0)).is_err());
        assert!(find_minimum_quartic([-1.0, 1.0, 1.0, 1.0, 1.0], (0.0, 1.0)).is_err());
    }
}
