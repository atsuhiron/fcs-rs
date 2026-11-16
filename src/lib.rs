//! Fire-control (lead) calculator.
//!
//! Given the specifications of a projectile and the position and constant velocity of a target, [`solve_lead`]
//! finds the azimuth and elevation whose trajectory, with altitude-dependent air drag, passes closest to the target.
//!
//! This is a port of the Python implementation at <https://github.com/atsuhiron/fcs>, whose README holds the
//! mathematical derivation. The calculation is three nested layers:
//!
//! 1. [`solve_trajectory`] integrates the equation of motion for one launch direction.
//! 2. [`find_closest_point`] refines the closest approach between the samples of a trajectory.
//! 3. [`solve_lead`] searches for the launch direction that minimises the closest distance.
//!
//! Every function checks its inputs first and returns [`Error::InvalidInput`] without calculating anything
//! if they are meaningless.

mod closest;
pub mod ode;
mod optimizer;
pub mod powell;
mod pymath;
mod trajectory;
mod types;
mod validate;

pub use optimizer::{Evaluation, INITIAL_STEP, MAX_EVALUATIONS, SEARCH_TOLERANCE};
pub use trajectory::G;
pub use types::{
    Aim, Atmosphere, ClosestPoint, DragType, Error, Projectile, Settings, Solution, Status, Target,
    Trajectory, Vector3,
};
pub use validate::{InvalidInput, Parameter, Requirement, find_invalid_inputs};

fn check(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
) -> Result<(), Error> {
    let errors = find_invalid_inputs(projectile, target, atmosphere, settings);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(Error::InvalidInput(errors))
    }
}

/// Find the launch direction that brings the projectile closest to the target.
///
/// The result is a [`Status::Hit`] if the closest distance is within `settings.tolerance`.
/// Otherwise the direction of the closest approach found is still returned, with the reason in `status`.
pub fn solve_lead(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
) -> Result<Solution, Error> {
    solve_lead_observed(projectile, target, atmosphere, settings, |_| {})
}

/// Same as [`solve_lead`], calling `observer` for every trajectory computed during the search, in order.
/// The last call is the check at the returned direction.
pub fn solve_lead_observed(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
    mut observer: impl FnMut(&Evaluation),
) -> Result<Solution, Error> {
    check(projectile, target, atmosphere, settings)?;
    optimizer::search_unchecked(projectile, target, atmosphere, settings, &mut observer)
}

/// Compute the trajectory of the projectile launched in the direction `aim`, and the path of the target.
///
/// The trajectory runs from launch (t = 0) until the projectile comes back to z = 0 or until `settings.span_end`,
/// sampled at the steps of the integrator.
pub fn solve_trajectory(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
    aim: Aim,
) -> Result<Trajectory, Error> {
    check(projectile, target, atmosphere, settings)?;
    trajectory::solve_unchecked(projectile, target, atmosphere, aim, settings.span_end)
}

/// Find the closest approach of the projectile to the target on a trajectory from [`solve_trajectory`].
///
/// `target` must be the one the trajectory was computed for.
pub fn find_closest_point(trajectory: &Trajectory, target: &Target) -> Result<ClosestPoint, Error> {
    if trajectory.t.is_empty()
        || trajectory.projectile.len() != trajectory.t.len()
        || trajectory.target.len() != trajectory.t.len()
        || trajectory.closest_sample >= trajectory.t.len()
    {
        return Err(Error::Numerical(
            "the trajectory is empty or inconsistent".to_string(),
        ));
    }
    closest::find_unchecked(trajectory, target)
}
