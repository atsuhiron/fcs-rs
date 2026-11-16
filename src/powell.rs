//! Powell's conjugate direction method with an unbounded Brent line search.
//!
//! This is a port of SciPy 1.17's `minimize(method="Powell")` without bounds: `_minimize_powell`,
//! `_linesearch_powell`, `Brent.optimize` and `bracket`. The sequence of evaluated points is the same as SciPy's.
//! Nothing here depends on ballistics.

use std::fmt;

/// Settings of the search (the options of SciPy's Powell method with the same names).
#[derive(Debug, Clone, PartialEq)]
pub struct PowellOptions {
    /// Tolerance of the line searches (SciPy's `xtol`; each line search runs Brent's method with `100 * xtol`).
    pub xtol: f64,
    /// Relative decrease of the function value in one iteration below which the search stops.
    pub ftol: f64,
    /// Maximum number of function evaluations. The search stops as soon as one more would be needed.
    pub max_evaluations: usize,
    /// Initial search directions, one row per direction.
    pub directions: Vec<Vec<f64>>,
}

/// Why the search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowellStatus {
    /// The decrease in one iteration fell below `ftol`.
    Converged,
    /// The limit on function evaluations was reached.
    MaxEvaluations,
    /// The function value became NaN.
    NanEncountered,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowellResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: PowellStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PowellError<E> {
    /// The objective function returned an error.
    Objective(E),
    /// `bracket` exceeded its iteration limit (SciPy raises `RuntimeError`).
    BracketIterationLimit,
}

impl<E: fmt::Display> fmt::Display for PowellError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PowellError::Objective(e) => e.fmt(f),
            PowellError::BracketIterationLimit => f.write_str(
                "no valid bracket was found for the line search before the iteration limit",
            ),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for PowellError<E> {}

/// Reasons to unwind out of the search.
enum Stop<E> {
    /// SciPy's `_MaxFuncCallError`: ends the search normally with the current point.
    MaxEvaluations,
    Error(PowellError<E>),
}

/// Counts the evaluations and refuses to go beyond the limit (SciPy's `_wrap_scalar_function_maxfun_validation`).
struct Counted<F> {
    function: F,
    calls: usize,
    limit: usize,
}

impl<F> Counted<F> {
    fn call<E>(&mut self, x: &[f64]) -> Result<f64, Stop<E>>
    where
        F: FnMut(&[f64]) -> Result<f64, E>,
    {
        if self.calls >= self.limit {
            return Err(Stop::MaxEvaluations);
        }
        self.calls += 1;
        (self.function)(x).map_err(|e| Stop::Error(PowellError::Objective(e)))
    }
}

struct Bracket {
    xa: f64,
    xb: f64,
    xc: f64,
    fa: f64,
    fb: f64,
    fc: f64,
}

/// Port of SciPy's `bracket` with `xa = 0`, `xb = 1`.
///
/// Returns the three points and whether they form a valid bracket (SciPy raises `BracketError` if not).
fn bracket<E>(
    func: &mut impl FnMut(f64) -> Result<f64, Stop<E>>,
) -> Result<(Bracket, bool), Stop<E>> {
    // SciPy uses this rounded value of the golden ratio; the exact constant would give different points.
    #[allow(clippy::approx_constant)]
    const GOLD: f64 = 1.618034;
    const VERY_SMALL: f64 = 1e-21;
    const GROW_LIMIT: f64 = 110.0;
    const MAX_ITER: usize = 1000;

    let mut xa = 0.0;
    let mut xb = 1.0;
    let mut fa = func(xa)?;
    let mut fb = func(xb)?;
    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut xc = xb + GOLD * (xb - xa);
    let mut fc = func(xc)?;
    let mut iter = 0;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < VERY_SMALL {
            2.0 * VERY_SMALL
        } else {
            2.0 * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + GROW_LIMIT * (xc - xb);
        if iter > MAX_ITER {
            return Err(Stop::Error(PowellError::BracketIterationLimit));
        }
        iter += 1;
        let fw;
        if (w - xc) * (xb - w) > 0.0 {
            let fw_inner = func(w)?;
            if fw_inner < fc {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw_inner;
                break;
            } else if fw_inner > fb {
                xc = w;
                fc = fw_inner;
                break;
            }
            w = xc + GOLD * (xc - xb);
            fw = func(w)?;
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = func(w)?;
        } else if (w - wlim) * (xc - w) > 0.0 {
            let fw_inner = func(w)?;
            if fw_inner < fc {
                xb = xc;
                xc = w;
                w = xc + GOLD * (xc - xb);
                fb = fc;
                fc = fw_inner;
                fw = func(w)?;
            } else {
                fw = fw_inner;
            }
        } else {
            w = xc + GOLD * (xc - xb);
            fw = func(w)?;
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }

    let cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    let cond2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
    let cond3 = xa.is_finite() && xb.is_finite() && xc.is_finite();
    Ok((
        Bracket {
            xa,
            xb,
            xc,
            fa,
            fb,
            fc,
        },
        cond1 && cond2 && cond3,
    ))
}

/// Port of SciPy's `_minimize_scalar_brent` with `brack=None`, wrapped in `_recover_from_bracket_error`.
/// Returns the minimum point and the value there.
fn minimize_scalar_brent<E>(
    func: &mut impl FnMut(f64) -> Result<f64, Stop<E>>,
    tol: f64,
) -> Result<(f64, f64), Stop<E>> {
    const MIN_TOL: f64 = 1.0e-11;
    const CG: f64 = 0.3819660;
    const MAX_ITER: usize = 500;

    let (br, valid) = bracket(func)?;
    if !valid {
        // _recover_from_bracket_error: the best of the three points, or NaN if anything is NaN
        let xs = [br.xa, br.xb, br.xc];
        let fs = [br.fa, br.fb, br.fc];
        if xs.iter().chain(fs.iter()).any(|v| v.is_nan()) {
            return Ok((f64::NAN, f64::NAN));
        }
        let mut imin = 0;
        for i in 1..3 {
            if fs[i] < fs[imin] {
                imin = i;
            }
        }
        return Ok((xs[imin], fs[imin]));
    }

    let mut x = br.xb;
    let mut w = br.xb;
    let mut v = br.xb;
    let mut fx = br.fb;
    let mut fw = br.fb;
    let mut fv = br.fb;
    let (mut a, mut b) = if br.xa < br.xc {
        (br.xa, br.xc)
    } else {
        (br.xc, br.xa)
    };
    let mut deltax: f64 = 0.0;
    let mut rat: f64 = 0.0;
    let mut iter = 0;

    while iter < MAX_ITER {
        let tol1 = tol * x.abs() + MIN_TOL;
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        // check for convergence
        if (x - xmid).abs() < (tol2 - 0.5 * (b - a)) {
            break;
        }
        if deltax.abs() <= tol1 {
            // do a golden section step
            deltax = if x >= xmid { a - x } else { b - x };
            rat = CG * deltax;
        } else {
            // do a parabolic step
            let tmp1 = (x - w) * (fx - fv);
            let mut tmp2 = (x - v) * (fx - fw);
            let mut p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if tmp2 > 0.0 {
                p = -p;
            }
            tmp2 = tmp2.abs();
            let dx_temp = deltax;
            deltax = rat;
            // check parabolic fit
            if p > tmp2 * (a - x) && p < tmp2 * (b - x) && p.abs() < (0.5 * tmp2 * dx_temp).abs() {
                rat = p * 1.0 / tmp2; // if parabolic step is useful
                let u = x + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                }
            } else {
                // if it's not, do a golden section step
                deltax = if x >= xmid { a - x } else { b - x };
                rat = CG * deltax;
            }
        }

        // update by at least tol1
        let u = if rat.abs() < tol1 {
            if rat >= 0.0 { x + tol1 } else { x - tol1 }
        } else {
            x + rat
        };
        let fu = func(u)?;

        if fu > fx {
            // if it's bigger than current
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        } else {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }

        iter += 1;
    }

    Ok((x, fx))
}

/// Port of SciPy's `_linesearch_powell` without bounds: minimise `func(p + alpha * xi)` over `alpha`.
/// Returns the minimum value, the point and the displacement.
fn line_search<F, E>(
    func: &mut Counted<F>,
    p: &[f64],
    xi: &[f64],
    tol: f64,
    fval: f64,
) -> Result<(f64, Vec<f64>, Vec<f64>), Stop<E>>
where
    F: FnMut(&[f64]) -> Result<f64, E>,
{
    // if xi is zero, then don't optimize
    if xi.iter().all(|v| *v == 0.0) {
        return Ok((fval, p.to_vec(), xi.to_vec()));
    }

    let mut point = vec![0.0; p.len()];
    let mut along = |alpha: f64| {
        for ((out, &pi), &di) in point.iter_mut().zip(p.iter()).zip(xi.iter()) {
            *out = pi + alpha * di;
        }
        func.call(&point)
    };
    let (alpha_min, fret) = minimize_scalar_brent(&mut along, tol)?;
    let xi: Vec<f64> = xi.iter().map(|v| alpha_min * v).collect();
    let p: Vec<f64> = p.iter().zip(xi.iter()).map(|(a, b)| a + b).collect();
    Ok((fret, p, xi))
}

/// Minimise `function` starting from `x0` with Powell's method.
///
/// The function may fail; the first error ends the search and is returned as [`PowellError::Objective`].
pub fn minimize<F, E>(
    function: F,
    x0: &[f64],
    options: &PowellOptions,
) -> Result<PowellResult, PowellError<E>>
where
    F: FnMut(&[f64]) -> Result<f64, E>,
{
    let n = x0.len();
    assert_eq!(
        options.directions.len(),
        n,
        "one direction per variable is required"
    );
    assert!(
        options.directions.iter().all(|d| d.len() == n),
        "each direction must have one component per variable"
    );

    let mut func = Counted {
        function,
        calls: 0,
        limit: options.max_evaluations,
    };
    let mut direc = options.directions.clone();
    let mut x = x0.to_vec();
    let mut fval = f64::NAN;
    let mut iterations = 0;

    // The body of SciPy's `while True` loop, including the evaluation at x0 before it.
    // An error here is either the evaluation limit (the search ends with the current x) or a real failure.
    let outcome: Result<(), Stop<E>> = (|| {
        fval = func.call(&x)?;
        let mut x1 = x.clone();
        loop {
            let fx = fval;
            let mut bigind = 0;
            let mut delta = 0.0;
            for (i, direction) in direc.iter().enumerate() {
                let fx2 = fval;
                let (f_new, x_new, _) =
                    line_search(&mut func, &x, direction, options.xtol * 100.0, fval)?;
                fval = f_new;
                x = x_new;
                if (fx2 - fval) > delta {
                    delta = fx2 - fval;
                    bigind = i;
                }
            }
            iterations += 1;
            let bnd = options.ftol * (fx.abs() + fval.abs()) + 1e-20;
            if 2.0 * (fx - fval) <= bnd {
                break;
            }
            if func.calls >= func.limit {
                break;
            }
            if fx.is_nan() && fval.is_nan() {
                // Ended up in a nan-region: bail out
                break;
            }

            // Construct the extrapolated point
            let direc1: Vec<f64> = x.iter().zip(x1.iter()).map(|(a, b)| a - b).collect();
            x1.clone_from(&x);
            let x2: Vec<f64> = x.iter().zip(direc1.iter()).map(|(a, b)| a + b).collect();
            let fx2 = func.call(&x2)?;

            if fx > fx2 {
                let mut t = 2.0 * (fx + fx2 - 2.0 * fval);
                let mut temp = fx - fval - delta;
                t *= temp * temp;
                temp = fx - fx2;
                t -= delta * temp * temp;
                if t < 0.0 {
                    let (f_new, x_new, direc1) =
                        line_search(&mut func, &x, &direc1, options.xtol * 100.0, fval)?;
                    fval = f_new;
                    x = x_new;
                    if direc1.iter().any(|v| *v != 0.0) {
                        direc[bigind] = direc[n - 1].clone();
                        direc[n - 1] = direc1;
                    }
                }
            }
        }
        Ok(())
    })();

    match outcome {
        Ok(()) | Err(Stop::MaxEvaluations) => {}
        Err(Stop::Error(e)) => return Err(e),
    }

    let status = if func.calls >= func.limit {
        PowellStatus::MaxEvaluations
    } else if fval.is_nan() || x.iter().any(|v| v.is_nan()) {
        PowellStatus::NanEncountered
    } else {
        PowellStatus::Converged
    };
    Ok(PowellResult {
        x,
        fun: fval,
        iterations,
        evaluations: func.calls,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    fn identity(n: usize, scale: f64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { scale } else { 0.0 }).collect())
            .collect()
    }

    fn options(n: usize, max_evaluations: usize) -> PowellOptions {
        PowellOptions {
            xtol: 1e-4,
            ftol: 1e-4,
            max_evaluations,
            directions: identity(n, 1.0),
        }
    }

    // The expected evaluation counts below are those of
    // scipy.optimize.minimize(f, x0, method="Powell") in SciPy 1.17.0 with the default options.

    #[test]
    fn quadratic_form() {
        let f = |x: &[f64]| -> Result<f64, Infallible> {
            Ok((x[0] - 1.0).powi(2) + 10.0 * (x[1] + 2.0).powi(2) + x[0] * x[1])
        };
        let result = minimize(f, &[0.0, 0.0], &options(2, 2000)).unwrap();
        // Stationary point of the quadratic form: x = 80/39, y = -82/39
        assert!((result.x[0] - 80.0 / 39.0).abs() < 1e-4, "{:?}", result.x);
        assert!((result.x[1] + 82.0 / 39.0).abs() < 1e-4, "{:?}", result.x);
        assert_eq!(result.status, PowellStatus::Converged);
        assert_eq!(result.evaluations, SCIPY_QUADRATIC_NFEV);
    }

    #[test]
    fn rosenbrock() {
        let f = |x: &[f64]| -> Result<f64, Infallible> {
            Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
        };
        let result = minimize(f, &[-1.2, 1.0], &options(2, 2000)).unwrap();
        assert!((result.x[0] - 1.0).abs() < 1e-3, "{:?}", result.x);
        assert!((result.x[1] - 1.0).abs() < 1e-3, "{:?}", result.x);
        assert_eq!(result.status, PowellStatus::Converged);
        assert_eq!(result.evaluations, SCIPY_ROSENBROCK_NFEV);
    }

    #[test]
    fn stops_at_the_evaluation_limit() {
        let mut calls = 0;
        let f = |x: &[f64]| -> Result<f64, Infallible> {
            calls += 1;
            Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
        };
        let result = minimize(f, &[-1.2, 1.0], &options(2, 50)).unwrap();
        assert_eq!(result.status, PowellStatus::MaxEvaluations);
        assert_eq!(result.evaluations, 50);
        assert_eq!(calls, 50);
    }

    #[test]
    fn objective_errors_are_returned() {
        let f = |x: &[f64]| -> Result<f64, &'static str> {
            if x[0] > 0.5 { Err("boom") } else { Ok(x[0]) }
        };
        let result = minimize(f, &[0.0], &options(1, 100));
        assert_eq!(result, Err(PowellError::Objective("boom")));
    }

    const SCIPY_QUADRATIC_NFEV: usize = 81;
    const SCIPY_ROSENBROCK_NFEV: usize = 607;
}
