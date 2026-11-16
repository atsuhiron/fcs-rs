//! Explicit Runge-Kutta integration (Dormand-Prince 5(4)) with a terminal event.
//!
//! This is a port of SciPy 1.17's `solve_ivp(method="RK45")`: `RungeKutta._step_impl`, `select_initial_step`,
//! `RkDenseOutput` and the event handling of `solve_ivp` (root finding with `brentq` on the dense output).
//! The order of the floating-point operations follows SciPy so that the accepted steps are the same.
//! Nothing here depends on ballistics.

use std::fmt;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
/// `-1 / (error_estimator_order + 1)`
const ERROR_EXPONENT: f64 = -1.0 / 5.0;

const N_STAGES: usize = 6;

const C: [f64; N_STAGES] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];
const A: [[f64; 5]; N_STAGES] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
];
const B: [f64; N_STAGES] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];
const E: [f64; N_STAGES + 1] = [
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0,
];
/// Coefficients of the quartic interpolant (Shampine 1986).
const P: [[f64; 4]; N_STAGES + 1] = [
    [
        1.0,
        -8048581381.0 / 2820520608.0,
        8663915743.0 / 2820520608.0,
        -12715105075.0 / 11282082432.0,
    ],
    [0.0, 0.0, 0.0, 0.0],
    [
        0.0,
        131558114200.0 / 32700410799.0,
        -68118460800.0 / 10900136933.0,
        87487479700.0 / 32700410799.0,
    ],
    [
        0.0,
        -1754552775.0 / 470086768.0,
        14199869525.0 / 1410260304.0,
        -10690763975.0 / 1880347072.0,
    ],
    [
        0.0,
        127303824393.0 / 49829197408.0,
        -318862633887.0 / 49829197408.0,
        701980252875.0 / 199316789632.0,
    ],
    [
        0.0,
        -282668133.0 / 205662961.0,
        2019193451.0 / 616988883.0,
        -1453857185.0 / 822651844.0,
    ],
    [
        0.0,
        40617522.0 / 29380423.0,
        -110615467.0 / 29380423.0,
        69997945.0 / 29380423.0,
    ],
];

/// Which sign change of the event function counts as an occurrence (SciPy's `event.direction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossing {
    /// From negative to positive.
    Rising,
    /// From positive to negative.
    Falling,
    Either,
}

/// A terminal event: the integration stops at the first root of `function(t, y)` with the given crossing.
pub struct TerminalEvent<G> {
    pub function: G,
    pub crossing: Crossing,
}

/// Why the integration ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// Reached the end of the time span.
    Finished,
    /// The terminal event occurred. The last sample is at the event.
    Event,
    /// The required step size became smaller than the spacing between numbers. The samples so far are returned.
    StepTooSmall,
}

/// Samples of the solution: the initial point and the end of every accepted step.
#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution<const N: usize> {
    pub t: Vec<f64>,
    pub y: Vec<[f64; N]>,
    pub termination: Termination,
}

/// Failures of the root finding for the event time. SciPy raises an exception in these cases.
#[derive(Debug, Clone, PartialEq)]
pub enum OdeError {
    /// The event function has the same sign at both ends of the step when evaluated on the interpolant.
    EventNotBracketed,
    /// The root finding did not converge.
    EventNotConverged,
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeError::EventNotBracketed => {
                f.write_str("the event function does not change sign within the step")
            }
            OdeError::EventNotConverged => {
                f.write_str("the root finding for the event time did not converge")
            }
        }
    }
}

impl std::error::Error for OdeError {}

/// `sum_j k[j][i] * c[j]`, rounded the way NumPy rounds `np.dot(K[:s].T, c)` in `rk_step`.
///
/// The step sizes depend on the error estimate, which is a sum that cancels almost completely, so a different
/// last bit in these products changes the sample times by far more than a last bit (around 1e-9 relative).
/// To take the same steps as the Python version, this follows the order of operations of the BLAS that NumPy
/// ships (`dgemv` of OpenBLAS 0.3.30, Haswell kernel; found by comparing against `np.dot` bit for bit):
/// the rows that fill whole groups of 4 take the columns in blocks of 4 with fused multiply-adds,
/// then a block of 2, then a single column; the remaining rows are summed in order.
fn stage_dot<const N: usize>(k: &[[f64; N]; N_STAGES + 1], c: &[f64], i: usize) -> f64 {
    let n = c.len();
    if i >= N - N % 4 {
        let mut total = 0.0;
        for j in 0..n {
            total += k[j][i] * c[j];
        }
        return total;
    }

    let mut total = 0.0;
    let mut j = 0;
    while n - j >= 4 {
        let even = k[j + 2][i].mul_add(c[j + 2], k[j][i] * c[j]);
        let odd = k[j + 3][i].mul_add(c[j + 3], k[j + 1][i] * c[j + 1]);
        total += even + odd;
        j += 4;
    }
    if n - j >= 2 {
        total += k[j][i] * c[j] + k[j + 1][i] * c[j + 1];
        j += 2;
    }
    if n - j == 1 {
        total += k[j][i] * c[j];
    }
    total
}

/// Root mean square norm (SciPy's `norm`).
fn rms_norm<const N: usize>(x: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    for v in x {
        sum += v * v;
    }
    sum.sqrt() / (N as f64).sqrt()
}

/// Port of SciPy's `select_initial_step` (with `max_step = inf`).
#[allow(clippy::too_many_arguments)]
fn select_initial_step<const N: usize, F>(
    fun: &mut F,
    t0: f64,
    y0: &[f64; N],
    t_bound: f64,
    f0: &[f64; N],
    direction: f64,
    rtol: f64,
    atol: f64,
) -> f64
where
    F: FnMut(f64, &[f64; N]) -> [f64; N],
{
    if N == 0 {
        return f64::INFINITY;
    }
    let interval_length = (t_bound - t0).abs();
    if interval_length == 0.0 {
        return 0.0;
    }

    let scale: [f64; N] = std::array::from_fn(|i| atol + y0[i].abs() * rtol);
    let d0 = rms_norm::<N>(&std::array::from_fn(|i| y0[i] / scale[i]));
    let d1 = rms_norm::<N>(&std::array::from_fn(|i| f0[i] / scale[i]));
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    let h0 = h0.min(interval_length);
    let y1: [f64; N] = std::array::from_fn(|i| y0[i] + h0 * direction * f0[i]);
    let f1 = fun(t0 + h0 * direction, &y1);
    let d2 = rms_norm::<N>(&std::array::from_fn(|i| (f1[i] - f0[i]) / scale[i])) / h0;

    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (1e-6f64).max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / 5.0)
    };
    (100.0 * h0).min(h1).min(interval_length)
}

/// The quartic interpolant over one step (SciPy's `RkDenseOutput`).
struct DenseOutput<const N: usize> {
    t_old: f64,
    h: f64,
    y_old: [f64; N],
    q: [[f64; 4]; N],
}

impl<const N: usize> DenseOutput<N> {
    fn new(t_old: f64, t: f64, y_old: &[f64; N], k: &[[f64; N]; N_STAGES + 1]) -> Self {
        let mut q = [[0.0; 4]; N];
        for (i, q_row) in q.iter_mut().enumerate() {
            for (j, q_value) in q_row.iter_mut().enumerate() {
                // `K.T.dot(P)` (dgemm): fused multiply-adds in order
                let mut sum = 0.0;
                for (k_row, p_row) in k.iter().zip(P.iter()) {
                    sum = k_row[i].mul_add(p_row[j], sum);
                }
                *q_value = sum;
            }
        }
        Self {
            t_old,
            h: t - t_old,
            y_old: *y_old,
            q,
        }
    }

    fn eval(&self, t: f64) -> [f64; N] {
        let x = (t - self.t_old) / self.h;
        let mut p = [x; 4];
        for i in 1..4 {
            p[i] *= p[i - 1];
        }
        std::array::from_fn(|i| {
            // `np.dot(Q, p)` (dgemv): the even and the odd terms are summed separately
            let q = &self.q[i];
            let sum = (q[0] * p[0] + q[2] * p[2]) + (q[1] * p[1] + q[3] * p[3]);
            self.h * sum + self.y_old[i]
        })
    }
}

/// Port of SciPy's `brentq` (`scipy/optimize/Zeros/brentq.c`) with `maxiter = 100`.
fn brentq<F>(mut f: F, xa: f64, xb: f64, xtol: f64, rtol: f64) -> Result<f64, OdeError>
where
    F: FnMut(f64) -> f64,
{
    const MAX_ITER: usize = 100;

    let mut xpre = xa;
    let mut xcur = xb;
    let mut xblk = 0.0;
    let mut fblk = 0.0;
    let mut spre = 0.0;
    let mut scur = 0.0;

    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err(OdeError::EventNotBracketed);
    }

    for _ in 0..MAX_ITER {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    Err(OdeError::EventNotConverged)
}

fn is_active(g: f64, g_new: f64, crossing: Crossing) -> bool {
    let up = g <= 0.0 && g_new >= 0.0;
    let down = g >= 0.0 && g_new <= 0.0;
    match crossing {
        Crossing::Rising => up,
        Crossing::Falling => down,
        Crossing::Either => up || down,
    }
}

/// Integrate `dy/dt = fun(t, y)` from `t0` to `t_bound` with the Dormand-Prince 5(4) pair.
///
/// The local error estimate of each step is kept below `atol + rtol * |y|`.
/// The samples are the initial point and the end of every accepted step; the step sizes are chosen by the solver.
/// If `event` occurs, the integration stops there and the last sample is at the event,
/// evaluated on the quartic interpolant of the step.
pub fn solve_ivp<const N: usize, F, G>(
    mut fun: F,
    t0: f64,
    t_bound: f64,
    y0: [f64; N],
    rtol: f64,
    atol: f64,
    mut event: Option<TerminalEvent<G>>,
) -> Result<OdeSolution<N>, OdeError>
where
    F: FnMut(f64, &[f64; N]) -> [f64; N],
    G: FnMut(f64, &[f64; N]) -> f64,
{
    // validate_tol
    let rtol = rtol.max(100.0 * f64::EPSILON);

    let direction = if t_bound != t0 {
        (t_bound - t0).signum()
    } else {
        1.0
    };

    let mut t = t0;
    let mut y = y0;
    let mut f = fun(t, &y);
    let mut h_abs = select_initial_step(&mut fun, t, &y, t_bound, &f, direction, rtol, atol);
    let mut k = [[0.0; N]; N_STAGES + 1];

    let mut ts = vec![t0];
    let mut ys = vec![y0];
    let mut g = event.as_mut().map(|e| (e.function)(t0, &y0));

    let termination = loop {
        // OdeSolver.step
        if N == 0 || t == t_bound {
            ts.push(t_bound);
            ys.push(y);
            break Termination::Finished;
        }

        // RungeKutta._step_impl
        let t_old = t;
        let y_old = y;
        let min_step = 10.0 * (next_after(t, direction) - t).abs();
        if h_abs < min_step {
            h_abs = min_step;
        }

        let mut step_rejected = false;
        let (t_new, y_new, f_new) = loop {
            if h_abs < min_step {
                return Ok(OdeSolution {
                    t: ts,
                    y: ys,
                    termination: Termination::StepTooSmall,
                });
            }

            let h = h_abs * direction;
            let mut t_new = t + h;
            if direction * (t_new - t_bound) > 0.0 {
                t_new = t_bound;
            }
            let h = t_new - t;
            h_abs = h.abs();

            // rk_step
            k[0] = f;
            for s in 1..N_STAGES {
                let stage_y: [f64; N] = std::array::from_fn(|i| {
                    let sum = stage_dot(&k, &A[s][..s], i);
                    y[i] + sum * h
                });
                k[s] = fun(t + C[s] * h, &stage_y);
            }
            let y_new: [f64; N] = std::array::from_fn(|i| {
                let sum = stage_dot(&k, &B, i);
                y[i] + h * sum
            });
            let f_new = fun(t + h, &y_new);
            k[N_STAGES] = f_new;

            let error: [f64; N] = std::array::from_fn(|i| {
                let sum = stage_dot(&k, &E, i);
                let scale = atol + y[i].abs().max(y_new[i].abs()) * rtol;
                sum * h / scale
            });
            let error_norm = rms_norm(&error);

            if error_norm < 1.0 {
                let mut factor = if error_norm == 0.0 {
                    MAX_FACTOR
                } else {
                    MAX_FACTOR.min(SAFETY * error_norm.powf(ERROR_EXPONENT))
                };
                if step_rejected {
                    factor = factor.min(1.0);
                }
                h_abs *= factor;
                break (t_new, y_new, f_new);
            }
            h_abs *= MIN_FACTOR.max(SAFETY * error_norm.powf(ERROR_EXPONENT));
            step_rejected = true;
        };

        t = t_new;
        y = y_new;
        f = f_new;
        let finished = direction * (t - t_bound) >= 0.0;

        // Event handling of solve_ivp
        let mut sample_t = t;
        let mut sample_y = y;
        let mut terminated = false;
        if let (Some(event), Some(g_old)) = (event.as_mut(), g) {
            let g_new = (event.function)(t, &y);
            if is_active(g_old, g_new, event.crossing) {
                let sol = DenseOutput::new(t_old, t, &y_old, &k);
                let root = brentq(
                    |tau| (event.function)(tau, &sol.eval(tau)),
                    t_old,
                    t,
                    4.0 * f64::EPSILON,
                    4.0 * f64::EPSILON,
                )?;
                sample_t = root;
                sample_y = sol.eval(root);
                terminated = true;
            }
            g = Some(g_new);
        }

        ts.push(sample_t);
        ys.push(sample_y);

        if terminated {
            break Termination::Event;
        }
        if finished {
            break Termination::Finished;
        }
    };

    Ok(OdeSolution {
        t: ts,
        y: ys,
        termination,
    })
}

/// The next representable number after `x` in the given direction (`np.nextafter(x, direction * inf)`).
fn next_after(x: f64, direction: f64) -> f64 {
    if direction > 0.0 {
        x.next_up()
    } else {
        x.next_down()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type NoEvent = fn(f64, &[f64; 1]) -> f64;

    #[test]
    fn exponential_decay() {
        let sol = solve_ivp(
            |_t, y: &[f64; 1]| [-0.5 * y[0]],
            0.0,
            10.0,
            [2.0],
            1e-8,
            1e-10,
            None::<TerminalEvent<NoEvent>>,
        )
        .unwrap();
        assert_eq!(sol.termination, Termination::Finished);
        assert_eq!(*sol.t.last().unwrap(), 10.0);
        for (t, y) in sol.t.iter().zip(sol.y.iter()) {
            let expected = 2.0 * (-0.5 * t).exp();
            assert!((y[0] - expected).abs() < 1e-7, "t = {t}");
        }
    }

    #[test]
    fn uniform_acceleration_stops_at_the_ground() {
        // y = [z, vz], thrown upwards at 20 m/s under g = 10 m/s^2: lands at t = 4.
        let sol = solve_ivp(
            |_t, y: &[f64; 2]| [y[1], -10.0],
            0.0,
            100.0,
            [0.0, 20.0],
            1e-8,
            1e-6,
            Some(TerminalEvent {
                function: |t: f64, y: &[f64; 2]| if t <= 0.0 { 1.0 } else { y[0] },
                crossing: Crossing::Falling,
            }),
        )
        .unwrap();
        assert_eq!(sol.termination, Termination::Event);
        let t_end = *sol.t.last().unwrap();
        let y_end = sol.y.last().unwrap();
        assert!((t_end - 4.0).abs() < 1e-9);
        assert!(y_end[0].abs() < 1e-9);
        assert!((y_end[1] + 20.0).abs() < 1e-9);
        // Every sample lies on the parabola.
        for (t, y) in sol.t.iter().zip(sol.y.iter()) {
            assert!((y[0] - (20.0 * t - 5.0 * t * t)).abs() < 1e-9);
        }
    }

    #[test]
    fn rising_event_is_ignored_by_a_falling_one() {
        // z starts below 0 and rises: a falling event must not fire.
        let sol = solve_ivp(
            |_t, _y: &[f64; 1]| [1.0],
            0.0,
            2.0,
            [-1.0],
            1e-8,
            1e-6,
            Some(TerminalEvent {
                function: |_t: f64, y: &[f64; 1]| y[0],
                crossing: Crossing::Falling,
            }),
        )
        .unwrap();
        assert_eq!(sol.termination, Termination::Finished);
        assert!((sol.y.last().unwrap()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn backward_integration() {
        let sol = solve_ivp(
            |_t, y: &[f64; 1]| [y[0]],
            1.0,
            0.0,
            [std::f64::consts::E],
            1e-10,
            1e-12,
            None::<TerminalEvent<NoEvent>>,
        )
        .unwrap();
        assert_eq!(*sol.t.last().unwrap(), 0.0);
        assert!((sol.y.last().unwrap()[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn brentq_finds_a_simple_root() {
        let root = brentq(
            |x| x * x - 2.0,
            0.0,
            2.0,
            4.0 * f64::EPSILON,
            4.0 * f64::EPSILON,
        )
        .unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-14);
        assert_eq!(
            brentq(|x| x * x + 1.0, 0.0, 2.0, 1e-12, 1e-12),
            Err(OdeError::EventNotBracketed)
        );
    }
}
