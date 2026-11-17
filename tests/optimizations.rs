//! Layer 3 against the Python version: results of the aim search and the sequence of evaluated points.

// `!(a <= b)` is deliberate: a NaN must count as a failure.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod common;

use common::OptimizationCase;
use fcs::{Evaluation, Solution, solve_lead_observed};

/// Tolerance for azimuth and elevation of a hit (degrees).
const ANGLE_TOLERANCE_DEG: f64 = 0.005;
/// Tolerance for the arrival time of a hit (ms).
const ETA_TOLERANCE_MSEC: f64 = 0.005;

fn run(case: &OptimizationCase) -> (Solution, Vec<[f64; 3]>) {
    let mut trace = Vec::new();
    let solution = solve_lead_observed(
        &common::projectile(&case.runtime_const),
        &common::target(case.t_pos, case.t_dir),
        &common::atmosphere(&case.atmosphere),
        &common::settings(&case.runtime_const, case.tolerance),
        |evaluation: &Evaluation| {
            trace.push([
                evaluation.aim.azimuth,
                evaluation.aim.elevation,
                evaluation.closest.distance,
            ]);
        },
    )
    .expect("the search runs");
    (solution, trace)
}

/// The line the Python version prints, without the calculation time.
fn format_cl(solution: &Solution) -> String {
    format!(
        "{:.2} {:.2} {:.2} {:.2}",
        solution.aim.azimuth.to_degrees(),
        solution.aim.elevation.to_degrees(),
        solution.distance,
        solution.eta_msec
    )
}

fn angle_difference_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).to_degrees().abs() % 360.0;
    d.min(360.0 - d)
}

#[test]
fn search_results_match_the_python_version() {
    let cases: Vec<OptimizationCase> = common::load("optimizations.json");
    assert_eq!(cases.len(), 282);

    let mut failures = Vec::new();
    let mut notes = Vec::new();
    let mut counts = [0usize; 3]; // hit, out_of_time, out_of_range
    let mut max_angle: f64 = 0.0;
    let mut max_eta: f64 = 0.0;
    let mut same_line = 0;
    let mut same_evaluations = 0;
    for (index, case) in cases.iter().enumerate() {
        let (solution, trace) = run(case);
        let name = format!("case {index} ({}, target = {:?})", case.set, case.t_pos);
        let status = solution.status.as_str();
        match status {
            "hit" => counts[0] += 1,
            "out_of_time" => counts[1] += 1,
            _ => counts[2] += 1,
        }
        assert_eq!(solution.evaluations, trace.len());

        // Hit or not must agree for every case.
        if (status == "hit") != (case.status == "hit") {
            failures.push(format!("{name}: status {status}, expected {}", case.status));
            continue;
        }
        if status == "hit" {
            let d_phi = angle_difference_deg(solution.aim.azimuth, case.phi);
            let d_theta = angle_difference_deg(solution.aim.elevation, case.theta);
            let d_eta = (solution.eta_msec - case.eta_msec).abs();
            max_angle = max_angle.max(d_phi).max(d_theta);
            max_eta = max_eta.max(d_eta);
            if !(d_phi <= ANGLE_TOLERANCE_DEG
                && d_theta <= ANGLE_TOLERANCE_DEG
                && d_eta <= ETA_TOLERANCE_MSEC)
            {
                failures.push(format!(
                    "{name}: azimuth differs by {d_phi:e} deg, elevation by {d_theta:e} deg, eta by {d_eta:e} ms"
                ));
            }
        }

        // The rest is expected to agree as well; report where it does not.
        if status != case.status {
            notes.push(format!("{name}: status {status}, expected {}", case.status));
        }
        if solution.evaluations == case.evaluations {
            same_evaluations += 1;
        } else {
            notes.push(format!(
                "{name}: {} evaluations, expected {}",
                solution.evaluations, case.evaluations
            ));
        }
        let line = format_cl(&solution);
        if line == case.format_cl {
            same_line += 1;
        } else {
            notes.push(format!(
                "{name} [{}]: output \"{line}\", expected \"{}\"",
                case.status, case.format_cl
            ));
        }
    }

    println!(
        "hit {}, out_of_time {}, out_of_range {}",
        counts[0], counts[1], counts[2]
    );
    println!("largest difference among hits: angle {max_angle:e} deg, eta {max_eta:e} ms");
    println!(
        "same number of evaluations: {same_evaluations} of {}; same output line: {same_line} of {}",
        cases.len(),
        cases.len()
    );
    for note in &notes {
        println!("note: {note}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} search results differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    // The split between out_of_time and out_of_range, the number of evaluations and the printed line are not
    // required to agree (see `searches_start_with_the_same_evaluations`): they are
    // printed above with `--nocapture` so that a change in them is visible.
}

/// The search must take the same path as the Python version for as long as the objective is well determined.
///
/// The two versions cannot agree to the end. Near the minimum the closest distance is the square root of a quartic
/// evaluated with cancellation, so it carries rounding noise of about 1e-6 m, and the line search then follows
/// that noise. The noise differs because the Python version fits the quadratics with `np.polyfit` and finds the
/// extrema with `np.roots` (LAPACK routines whose rounding cannot be reproduced), while this crate uses closed
/// forms. With those two replaced by the same closed forms, the Python version gives bit-identical results and
/// evaluation counts for all 282 cases.
///
/// So this test checks that every search starts with the same evaluations and gets the same distances there.
/// Where the paths part, they may part by a whole step: a comparison between two noisy values decides
/// which way the line search goes. Powell's method itself is checked to the end by
/// `powell_requests_the_same_points_as_scipy_when_given_the_same_values`.
#[test]
fn searches_start_with_the_same_evaluations() {
    /// Angles closer than this are the same evaluation (rad).
    const SAME_POINT: f64 = 1e-8;
    /// Distances at the same point agree within this (m): the rounding noise near a hit.
    const DISTANCE_NOISE: f64 = 1e-4;
    /// Every search agrees at least this long: the first line search along each axis.
    const MIN_COMMON: usize = 8;

    let cases: Vec<OptimizationCase> = common::load("optimizations.json");

    let mut traced = 0;
    let mut failures = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let Some(expected) = &case.trace else {
            continue;
        };
        traced += 1;
        let (_, trace) = run(case);
        let name = format!("case {index} ({}, {})", case.set, case.status);

        let angle_difference =
            |a: &[f64; 3], b: &[f64; 3]| (a[0] - b[0]).abs().max((a[1] - b[1]).abs());
        let common = trace
            .iter()
            .zip(expected.iter())
            .take_while(|(a, b)| angle_difference(a, b) <= SAME_POINT)
            .count();
        let distance_difference = trace
            .iter()
            .zip(expected.iter())
            .take(common)
            .map(|(a, b)| (a[2] - b[2]).abs())
            .fold(0.0, f64::max);
        println!(
            "{name}: first {common} of {} evaluations are the same (Python: {}), \
             distances within {distance_difference:.1e} m",
            trace.len(),
            expected.len()
        );

        if common < MIN_COMMON {
            failures.push(format!("{name}: only the first {common} evaluations agree"));
        }
        if !(distance_difference <= DISTANCE_NOISE) {
            failures.push(format!(
                "{name}: distances differ by {distance_difference:e} m at the same points"
            ));
        }
    }

    assert_eq!(traced, 20);
    assert!(
        failures.is_empty(),
        "{} problems in {traced} traces:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Powell's method itself, separated from the trajectory calculation: when the objective returns the values
/// the Python version computed, the port must ask for exactly the points SciPy asked for, to the last bit.
#[test]
fn powell_requests_the_same_points_as_scipy_when_given_the_same_values() {
    use fcs::powell::{PowellOptions, minimize};
    use std::f64::consts::PI;

    let cases: Vec<OptimizationCase> = common::load("optimizations.json");
    let mut traced = 0;
    let mut points = 0;
    for (index, case) in cases.iter().enumerate() {
        let Some(expected) = &case.trace else {
            continue;
        };
        traced += 1;
        // The last entry is the check at the optimum after the search.
        let search = &expected[..expected.len() - 1];

        let mut step = 0;
        let options = PowellOptions {
            xtol: fcs::SEARCH_TOLERANCE,
            ftol: fcs::SEARCH_TOLERANCE,
            max_evaluations: fcs::MAX_EVALUATIONS,
            directions: vec![vec![fcs::INITIAL_STEP, 0.0], vec![0.0, fcs::INITIAL_STEP]],
        };
        let result = minimize(
            |x: &[f64]| -> Result<f64, String> {
                let Some(wanted) = search.get(step) else {
                    return Err(format!(
                        "case {index}: more than {} evaluations",
                        search.len()
                    ));
                };
                // The trace holds the elevation after it was limited to [0, pi].
                let point = [x[0], x[1].clamp(0.0, PI)];
                if point != [wanted[0], wanted[1]] {
                    return Err(format!(
                        "case {index}: evaluation {step} is at {point:?}, expected {:?}",
                        &wanted[..2]
                    ));
                }
                step += 1;
                // distance ** 2 in Python is pow(distance, 2.0)
                Ok(wanted[2].powf(std::hint::black_box(2.0)))
            },
            &search[0][..2],
            &options,
        );
        let result = result.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(step, search.len(), "case {index}: number of evaluations");
        assert_eq!(result.evaluations, search.len());
        points += step;
    }
    println!("{traced} traced cases, {points} points, all bit-identical");
    assert_eq!(traced, 20);
}
