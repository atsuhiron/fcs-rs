//! Layers 1 and 2 against the Python version: trajectories and the closest point on each.

// `!(a <= b)` is deliberate: a NaN must count as a failure.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod common;

use common::TrajectoryCase;
use fcs::{Aim, Target, Trajectory, find_closest_point, solve_trajectory};

/// Relative tolerance for times and positions of the samples (layer 1).
/// Values smaller than 1 are compared with this as an absolute tolerance.
const SAMPLE_TOLERANCE: f64 = 1e-9;
/// Absolute tolerance for the time and the distance of the closest point (layer 2).
const CLOSEST_TOLERANCE: f64 = 1e-7;

fn solve(case: &TrajectoryCase) -> (Trajectory, Target) {
    let target = common::target(case.t_pos, case.t_dir);
    let trajectory = solve_trajectory(
        &common::projectile(&case.runtime_const),
        &target,
        &common::atmosphere(&case.atmosphere),
        &common::settings(&case.runtime_const, 0.1),
        Aim {
            azimuth: case.phi,
            elevation: case.theta,
        },
    )
    .expect("the trajectory is computed");
    (trajectory, target)
}

fn describe(index: usize, case: &TrajectoryCase) -> String {
    format!(
        "case {index} ({}, theta = {:.4} deg, phi = {:.2} deg, target = {:?})",
        case.model,
        case.theta.to_degrees(),
        case.phi.to_degrees(),
        case.t_pos
    )
}

fn error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

#[test]
fn trajectories_match_the_python_version() {
    let cases: Vec<TrajectoryCase> = common::load("trajectories.json");
    assert_eq!(cases.len(), 396);

    let mut failures = Vec::new();
    let mut max_error: f64 = 0.0;
    let mut identical = 0;
    for (index, case) in cases.iter().enumerate() {
        let (trajectory, _) = solve(case);
        let name = describe(index, case);

        if trajectory.t.len() != case.t.len() {
            failures.push(format!(
                "{name}: {} samples, expected {}",
                trajectory.t.len(),
                case.t.len()
            ));
            continue;
        }
        if trajectory.closest_sample != case.approx_index {
            failures.push(format!(
                "{name}: closest sample {}, expected {}",
                trajectory.closest_sample, case.approx_index
            ));
        }

        let mut case_error: f64 = 0.0;
        for i in 0..case.t.len() {
            case_error = case_error.max(error(trajectory.t[i], case.t[i]));
            for axis in 0..3 {
                case_error = case_error.max(error(
                    trajectory.projectile[i][axis],
                    case.projectile[axis][i],
                ));
                case_error =
                    case_error.max(error(trajectory.target[i][axis], case.target[axis][i]));
            }
        }
        // NaN must fail too
        if case_error.is_nan() || case_error > SAMPLE_TOLERANCE {
            failures.push(format!("{name}: error {case_error:e}"));
        }
        max_error = max_error.max(case_error);
        if case_error == 0.0 {
            identical += 1;
        }
    }

    // On Windows x86-64, where the reference data was generated, every trajectory is identical to the last bit.
    println!(
        "largest error over all samples: {max_error:e}; {identical} of {} trajectories are bit-identical",
        cases.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {} trajectories differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn closest_points_match_the_python_version() {
    let cases: Vec<TrajectoryCase> = common::load("trajectories.json");

    let mut failures = Vec::new();
    let mut max_t_error: f64 = 0.0;
    let mut max_distance_error: f64 = 0.0;
    let mut max_position_error: f64 = 0.0;
    let mut few_samples = 0;
    for (index, case) in cases.iter().enumerate() {
        let (trajectory, target) = solve(case);
        let closest = find_closest_point(&trajectory, &target).expect("a closest point is found");
        if case.t.len() < 3 {
            few_samples += 1;
        }

        let t_error = (closest.t - case.closest.t).abs();
        let distance_error = (closest.distance - case.closest.distance).abs();
        let mut position_error: f64 = 0.0;
        for axis in 0..3 {
            position_error = position_error
                .max((closest.target_pos[axis] - case.closest.target_pos[axis]).abs())
                .max((closest.projectile_pos[axis] - case.closest.proj_pos[axis]).abs());
        }
        if !(t_error <= CLOSEST_TOLERANCE && distance_error <= CLOSEST_TOLERANCE) {
            failures.push(format!(
                "{}: t {} (expected {}), distance {} (expected {})",
                describe(index, case),
                closest.t,
                case.closest.t,
                closest.distance,
                case.closest.distance
            ));
        }
        max_t_error = max_t_error.max(t_error);
        max_distance_error = max_distance_error.max(distance_error);
        max_position_error = max_position_error.max(position_error);
    }

    println!(
        "largest errors: t {max_t_error:e}, distance {max_distance_error:e}, positions {max_position_error:e}"
    );
    assert_eq!(few_samples, 72, "cases with fewer than 3 samples");
    assert!(
        failures.is_empty(),
        "{} of {} closest points differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
