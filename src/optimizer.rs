//! Aim search: find the launch direction whose trajectory passes closest to the target.

use std::f64::consts::PI;

use crate::closest;
use crate::powell::{self, PowellOptions};
use crate::pymath::square;
use crate::trajectory;
use crate::types::{
    Aim, Atmosphere, ClosestPoint, Error, Projectile, Settings, Solution, Status, Target,
};

/// Maximum number of trajectories computed during the search. A hit usually converges in 100-250;
/// the limit keeps the search short for targets that cannot be reached.
pub const MAX_EVALUATIONS: usize = 300;
/// Length of the initial search directions (rad). The solution is usually within a few degrees of the initial
/// guess (the direction of the target's current position); SciPy's default of 1 rad overshoots it.
pub const INITIAL_STEP: f64 = 0.02;
/// `xtol` and `ftol` of Powell's method.
pub const SEARCH_TOLERANCE: f64 = 1e-6;

/// One evaluation of the search: the direction tried and the closest approach it gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluation {
    pub aim: Aim,
    pub closest: ClosestPoint,
}

/// Wrap an angle into (-pi, pi].
fn wrap_angle(phi: f64) -> f64 {
    PI - python_mod(PI - phi, 2.0 * PI)
}

/// Python's `%` for floats with a positive divisor: the result has the sign of the divisor.
fn python_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 && m < 0.0 { m + b } else { m }
}

/// The elevation may pass the zenith (pi/2) to the other side. At the zenith the azimuth has no effect,
/// so with an upper limit of pi/2 a search that reaches it with the azimuth pointing the wrong way
/// could not leave.
fn clip_elevation(theta: f64) -> f64 {
    // min(max(theta, 0), pi) as Python evaluates it: NaN stays NaN
    let lower = if 0.0 > theta { 0.0 } else { theta };
    if PI < lower { PI } else { lower }
}

/// Run the search without checking the inputs. `observer` is called for every trajectory computed.
pub(crate) fn search_unchecked(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
    observer: &mut dyn FnMut(&Evaluation),
) -> Result<Solution, Error> {
    let mut evaluations = 0;
    // Returns the closest point and whether the trajectory was cut off at span_end.
    let mut evaluate = |aim: Aim| -> Result<(ClosestPoint, bool), Error> {
        let traj =
            trajectory::solve_unchecked(projectile, target, atmosphere, aim, settings.span_end)?;
        // A trajectory that ends without landing has its last sample exactly at span_end.
        let cut_off = traj.t.last().is_some_and(|&t| t >= settings.span_end);
        let closest = closest::find_unchecked(&traj, target)?;
        evaluations += 1;
        observer(&Evaluation { aim, closest });
        Ok((closest, cut_off))
    };

    // The initial guess points straight at the target's current position.
    let [dx, dy, dz] = target.position;
    let r_xy = (square(dx) + square(dy)).sqrt();
    let x0 = [dy.atan2(dx), dz.atan2(r_xy)];

    let options = PowellOptions {
        xtol: SEARCH_TOLERANCE,
        ftol: SEARCH_TOLERANCE,
        max_evaluations: MAX_EVALUATIONS,
        directions: vec![vec![INITIAL_STEP, 0.0], vec![0.0, INITIAL_STEP]],
    };
    // The squared distance is smooth at a hit. The azimuth is periodic, so it is left unbounded and wrapped
    // afterwards. The elevation is limited here rather than by bounds of the search: with bounds the line
    // search spans the whole interval and jumps to another valley.
    let result = powell::minimize(
        |x: &[f64]| -> Result<f64, Error> {
            let (closest, _) = evaluate(Aim {
                azimuth: x[0],
                elevation: clip_elevation(x[1]),
            })?;
            Ok(square(closest.distance))
        },
        &x0,
        &options,
    )
    .map_err(|e| match e {
        powell::PowellError::Objective(e) => e,
        other => Error::Numerical(other.to_string()),
    })?;

    // Whether the search "converged" does not tell whether it is a hit (a hit can reach the evaluation limit,
    // and an unreachable target can converge), so the closest distance itself decides.
    let mut azimuth = result.x[0];
    let mut elevation = clip_elevation(result.x[1]);
    if elevation > PI / 2.0 {
        // A direction beyond the zenith is the same as the opposite azimuth with the elevation folded back.
        azimuth += PI;
        elevation = PI - elevation;
    }
    let (closest, cut_off) = evaluate(Aim { azimuth, elevation })?;
    let status = if closest.distance <= settings.tolerance {
        Status::Hit
    } else if cut_off {
        Status::OutOfTime
    } else {
        Status::OutOfRange
    };

    Ok(Solution {
        aim: Aim {
            azimuth: wrap_angle(azimuth),
            elevation,
        },
        distance: closest.distance,
        eta_msec: closest.t * 1000.0,
        status,
        evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_angle_range() {
        assert_eq!(wrap_angle(0.0), 0.0);
        assert_eq!(wrap_angle(PI), PI);
        assert!((wrap_angle(-PI) - PI).abs() < 1e-15);
        assert!((wrap_angle(3.0 * PI / 2.0) + PI / 2.0).abs() < 1e-15);
        assert!((wrap_angle(-3.0 * PI / 2.0) - PI / 2.0).abs() < 1e-15);
        assert!((wrap_angle(5.0 * PI + 0.25) - (PI + 0.25 - 2.0 * PI)).abs() < 1e-14);
    }

    #[test]
    fn clip_elevation_range() {
        assert_eq!(clip_elevation(-0.1), 0.0);
        assert_eq!(clip_elevation(0.3), 0.3);
        assert_eq!(clip_elevation(4.0), PI);
    }
}
