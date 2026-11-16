//! The state equation, the landing event and the trajectory of a projectile.
//!
//! The state is in the target-relative frame: the initial position is `-target.position` and the target's velocity
//! is subtracted from the muzzle velocity, so the norm of the state position is the projectile-target distance.
//! Drag is the exception: it depends on the motion through the air, so the state equation converts back to the
//! absolute velocity and altitude before computing it.

use crate::ode::{self, Crossing, TerminalEvent};
use crate::pymath::square;
use crate::types::{Aim, Atmosphere, DragType, Error, Projectile, Target, Trajectory};

/// Standard acceleration of gravity (m/s^2), `scipy.constants.g`.
pub const G: f64 = 9.80665;

const RTOL: f64 = 1e-8;
/// SciPy's default `atol`; the Python version does not set it.
const ATOL: f64 = 1e-6;

type State = [f64; 6];

fn initial_state(projectile: &Projectile, target: &Target, aim: Aim) -> State {
    let v0 = projectile.initial_velocity;
    let (theta, phi) = (aim.elevation, aim.azimuth);
    [
        -target.position[0],
        -target.position[1],
        -target.position[2],
        v0 * theta.cos() * phi.cos() - target.velocity[0],
        v0 * theta.cos() * phi.sin() - target.velocity[1],
        v0 * theta.sin() - target.velocity[2],
    ]
}

/// Compute the trajectory for a launch direction, without checking the inputs.
///
/// The integration runs from launch (t = 0) until the projectile comes back to z = 0 or until `span_end`.
pub(crate) fn solve_unchecked(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    aim: Aim,
    span_end: f64,
) -> Result<Trajectory, Error> {
    let area = projectile.area;
    let cd = projectile.drag_coefficient;
    let m = projectile.mass;
    let t_z0 = target.position[2];
    let [t_vx, t_vy, t_vz] = target.velocity;
    let drag_type = projectile.drag_type;

    let state_eq = |t: f64, state: &State| -> State {
        let [_, _, z, vx, vy, vz] = *state;

        let abs_vx = vx + t_vx;
        let abs_vy = vy + t_vy;
        let abs_vz = vz + t_vz;
        let abs_z = z + t_z0 + t_vz * t;

        let v_norm = (square(abs_vx) + square(abs_vy) + square(abs_vz)).sqrt();
        let k = area * cd * atmosphere.density(abs_z) / 2.0;
        let v_coef = match drag_type {
            DragType::Pressure => k * v_norm / m,
            DragType::Friction => k / m,
        };

        // The target moves at a constant velocity, so the relative acceleration equals the absolute one.
        [
            vx,
            vy,
            vz,
            -v_coef * abs_vx,
            -v_coef * abs_vy,
            -v_coef * abs_vz - G,
        ]
    };

    // Fires when the absolute z of the projectile crosses zero downwards. The launch point is also at z = 0,
    // so a positive constant is returned at t = 0 to keep the launch from being taken for a landing.
    let fall_event = TerminalEvent {
        function: |t: f64, state: &State| -> f64 {
            if t <= 0.0 {
                return 1.0;
            }
            state[2] + target.position_at(t)[2]
        },
        crossing: Crossing::Falling,
    };

    let sol = ode::solve_ivp(
        state_eq,
        0.0,
        span_end,
        initial_state(projectile, target, aim),
        RTOL,
        ATOL,
        Some(fall_event),
    )
    .map_err(|e| Error::Numerical(e.to_string()))?;

    let mut target_traj = Vec::with_capacity(sol.t.len());
    let mut projectile_traj = Vec::with_capacity(sol.t.len());
    let mut closest_sample = 0;
    let mut min_norm = f64::INFINITY;
    for (index, (&t, state)) in sol.t.iter().zip(sol.y.iter()).enumerate() {
        let target_pos = target.position_at(t);
        projectile_traj.push([
            state[0] + target_pos[0],
            state[1] + target_pos[1],
            state[2] + target_pos[2],
        ]);
        target_traj.push(target_pos);

        let norm = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
        // The first minimum wins, and NaN is the minimum (np.argmin).
        if norm < min_norm || (norm.is_nan() && !min_norm.is_nan()) {
            min_norm = norm;
            closest_sample = index;
        }
    }

    Ok(Trajectory {
        t: sol.t,
        projectile: projectile_traj,
        target: target_traj,
        closest_sample,
    })
}
