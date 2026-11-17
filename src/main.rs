//! `lead`: command-line interface compatible with the Python version's `lead`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{CommandFactory, Parser, ValueEnum};
use fcs::{
    Atmosphere, DragType, Error, Parameter, Projectile, Settings, Solution, Status, Target,
    Trajectory, find_closest_point, solve_lead, solve_trajectory,
};
use serde::Serialize;

/// Exit status for an unexpected error (what Python exits with on an uncaught exception).
const EXIT_UNEXPECTED: u8 = 1;
/// Exit status for invalid arguments (what argparse and clap exit with).
const EXIT_INVALID_ARGUMENTS: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DragTypeArg {
    Pressure,
    Friction,
}

/// Fire Control System Simulation
#[derive(Debug, Parser)]
#[command(
    name = "lead",
    bin_name = "lead",
    version,
    infer_long_args = true,
    after_help = "exit status: 0 hit, 1 unexpected error, 2 invalid arguments, \
                  3 out_of_range (the projectile cannot reach the target), \
                  4 out_of_time (it does not reach the target within --span-end). \
                  For 3 and 4 the result is still printed and \"lead: <status>: <message>\" is written to stderr."
)]
struct Args {
    /// Mass of the projectile (kg)
    #[arg(value_name = "mass", allow_negative_numbers = true)]
    mass: f64,
    /// Cross-sectional area of projectile (m^2)
    #[arg(value_name = "area", allow_negative_numbers = true)]
    area: f64,
    /// Drag coefficient of projectile
    #[arg(value_name = "drag_coefficient", allow_negative_numbers = true)]
    drag_coefficient: f64,
    /// Initial velocity of projectile (m/s)
    #[arg(value_name = "initial_velocity", allow_negative_numbers = true)]
    initial_velocity: f64,
    /// Target X position (m)
    #[arg(value_name = "target_x", allow_negative_numbers = true)]
    target_x: f64,
    /// Target Y position (m)
    #[arg(value_name = "target_y", allow_negative_numbers = true)]
    target_y: f64,
    /// Target Z position (m)
    #[arg(value_name = "target_z", allow_negative_numbers = true)]
    target_z: f64,
    /// Target direction X component
    #[arg(value_name = "direction_x", allow_negative_numbers = true)]
    direction_x: f64,
    /// Target direction Y component
    #[arg(value_name = "direction_y", allow_negative_numbers = true)]
    direction_y: f64,
    /// Target direction Z component
    #[arg(value_name = "direction_z", allow_negative_numbers = true)]
    direction_z: f64,

    /// Maximum flight time to simulate (s)
    #[arg(long, default_value_t = 10.0, allow_negative_numbers = true)]
    span_end: f64,
    /// Closest distance to the target that counts as a hit (m)
    #[arg(long, default_value_t = 0.1, allow_negative_numbers = true)]
    tolerance: f64,
    /// Standard atmospheric density (kg/m^3)
    #[arg(long, default_value_t = 1.225, allow_negative_numbers = true)]
    std_density: f64,
    /// Standard atmospheric temperature (K)
    #[arg(long, default_value_t = 288.15, allow_negative_numbers = true)]
    std_temp: f64,
    /// Atmospheric lapse rate (K/m)
    #[arg(long, default_value_t = 0.0065, allow_negative_numbers = true)]
    lapse_rate: f64,
    /// Atmospheric index
    #[arg(long, default_value_t = 5.256, allow_negative_numbers = true)]
    lapse_index: f64,
    /// Type of drag model to use
    #[arg(long, value_enum, default_value_t = DragTypeArg::Pressure)]
    drag_type: DragTypeArg,
    /// Write the trajectory at the result and the closest point to this file as JSON, for plotting
    #[arg(long, value_name = "PATH")]
    dump: Option<PathBuf>,
    /// Output in a human-readable format
    #[arg(long, short = 'r')]
    readable: bool,
}

/// The name of a parameter in a message about a non-finite value.
/// The Python version uses the attribute names of the parsed arguments there.
fn attribute_name(parameter: Parameter) -> &'static str {
    match parameter {
        Parameter::Mass => "mass",
        Parameter::Area => "area",
        Parameter::DragCoefficient => "drag_coefficient",
        Parameter::InitialVelocity => "initial_velocity",
        Parameter::TargetX => "target_x",
        Parameter::TargetY => "target_y",
        Parameter::TargetZ => "target_z",
        Parameter::VelocityX => "direction_x",
        Parameter::VelocityY => "direction_y",
        Parameter::VelocityZ => "direction_z",
        Parameter::SpanEnd => "span_end",
        Parameter::Tolerance => "tolerance",
        Parameter::StdDensity => "std_density",
        Parameter::StdTemp => "std_temp",
        Parameter::LapseRate => "lapse_rate",
        Parameter::LapseIndex => "lapse_index",
    }
}

/// The name of a parameter as it is written on the command line.
fn argument_name(parameter: Parameter) -> &'static str {
    match parameter {
        Parameter::SpanEnd => "--span-end",
        Parameter::Tolerance => "--tolerance",
        Parameter::StdDensity => "--std-density",
        Parameter::StdTemp => "--std-temp",
        Parameter::LapseRate => "--lapse-rate",
        Parameter::LapseIndex => "--lapse-index",
        positional => attribute_name(positional),
    }
}

fn describe_invalid_inputs(errors: &[fcs::InvalidInput]) -> String {
    let messages: Vec<String> = errors
        .iter()
        .map(|error| {
            error.describe(|parameter| {
                let non_finite = matches!(
                    error,
                    fcs::InvalidInput::Value {
                        requirement: fcs::Requirement::Finite,
                        ..
                    }
                );
                if non_finite {
                    attribute_name(parameter).to_string()
                } else {
                    argument_name(parameter).to_string()
                }
            })
        })
        .collect();
    messages.join("; ")
}

fn status_message(status: Status) -> &'static str {
    match status {
        Status::Hit => "the projectile hits the target",
        Status::OutOfRange => "the projectile cannot reach the target",
        Status::OutOfTime => "the projectile does not reach the target within --span-end",
    }
}

fn exit_code(status: Status) -> u8 {
    match status {
        Status::Hit => 0,
        Status::OutOfRange => 3,
        Status::OutOfTime => 4,
    }
}

fn format_result(solution: &Solution, calc_msec: f64, readable: bool) -> String {
    // np.rad2deg multiplies by 180 / pi
    let to_deg = 180.0 / std::f64::consts::PI;
    let phi = solution.aim.azimuth * to_deg;
    let theta = solution.aim.elevation * to_deg;
    if readable {
        format!(
            "phi: {phi:.2}°, theta: {theta:.2}°, dist: {:.2}, eta_time: {:.2} ms, calc_time: {calc_msec:.2} ms",
            solution.distance, solution.eta_msec
        )
    } else {
        format!(
            "{phi:.2} {theta:.2} {:.2} {:.2} {calc_msec:.2}",
            solution.distance, solution.eta_msec
        )
    }
}

/// Contents of the `--dump` file. Angles are in radians.
///
/// `trajectory` and `closest` have the fields of the Python version's `TrajectoryResult` and `ApproximateResult`,
/// which its `plot_3d` takes.
#[derive(Serialize)]
struct Dump {
    format_version: u32,
    input: DumpInput,
    result: DumpResult,
    trajectory: DumpTrajectory,
    closest: DumpClosest,
}

#[derive(Serialize)]
struct DumpInput {
    mass: f64,
    area: f64,
    drag_coefficient: f64,
    initial_velocity: f64,
    drag_type: &'static str,
    target_position: [f64; 3],
    target_velocity: [f64; 3],
    span_end: f64,
    tolerance: f64,
    std_density: f64,
    std_temp: f64,
    lapse_rate: f64,
    lapse_index: f64,
}

#[derive(Serialize)]
struct DumpResult {
    status: &'static str,
    phi: f64,
    theta: f64,
    distance: f64,
    eta_msec: f64,
}

#[derive(Serialize)]
struct DumpTrajectory {
    t: Vec<f64>,
    /// `[x, y, z]`, each with one value per sample
    projectile_traj: [Vec<f64>; 3],
    target_traj: [Vec<f64>; 3],
    approx_index: usize,
}

#[derive(Serialize)]
struct DumpClosest {
    t: f64,
    distance: f64,
    target_pos: [f64; 3],
    proj_pos: [f64; 3],
}

fn by_axis(points: &[[f64; 3]]) -> [Vec<f64>; 3] {
    std::array::from_fn(|axis| points.iter().map(|p| p[axis]).collect())
}

/// Solve once more at the result and write the trajectory, as the Python version does for `--plot`.
fn write_dump(
    path: &Path,
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
    solution: &Solution,
) -> Result<(), String> {
    let trajectory: Trajectory =
        solve_trajectory(projectile, target, atmosphere, settings, solution.aim)
            .map_err(|e| e.to_string())?;
    let closest = find_closest_point(&trajectory, target).map_err(|e| e.to_string())?;

    let dump = Dump {
        format_version: 1,
        input: DumpInput {
            mass: projectile.mass,
            area: projectile.area,
            drag_coefficient: projectile.drag_coefficient,
            initial_velocity: projectile.initial_velocity,
            drag_type: match projectile.drag_type {
                DragType::Pressure => "pressure",
                DragType::Friction => "friction",
            },
            target_position: target.position,
            target_velocity: target.velocity,
            span_end: settings.span_end,
            tolerance: settings.tolerance,
            std_density: atmosphere.std_density,
            std_temp: atmosphere.std_temp,
            lapse_rate: atmosphere.lapse_rate,
            lapse_index: atmosphere.lapse_index,
        },
        result: DumpResult {
            status: solution.status.as_str(),
            phi: solution.aim.azimuth,
            theta: solution.aim.elevation,
            distance: solution.distance,
            eta_msec: solution.eta_msec,
        },
        trajectory: DumpTrajectory {
            projectile_traj: by_axis(&trajectory.projectile),
            target_traj: by_axis(&trajectory.target),
            approx_index: trajectory.closest_sample,
            t: trajectory.t,
        },
        closest: DumpClosest {
            t: closest.t,
            distance: closest.distance,
            target_pos: closest.target_pos,
            proj_pos: closest.projectile_pos,
        },
    };
    let text = serde_json::to_string(&dump).map_err(|e| e.to_string())?;
    fs::write(path, text + "\n").map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn main() -> ExitCode {
    // Wrong usage (missing or malformed arguments) ends here with exit status 2.
    let args = Args::parse();

    let projectile = Projectile {
        mass: args.mass,
        area: args.area,
        drag_coefficient: args.drag_coefficient,
        initial_velocity: args.initial_velocity,
        drag_type: match args.drag_type {
            DragTypeArg::Pressure => DragType::Pressure,
            DragTypeArg::Friction => DragType::Friction,
        },
    };
    let target = Target {
        position: [args.target_x, args.target_y, args.target_z],
        velocity: [args.direction_x, args.direction_y, args.direction_z],
    };
    let atmosphere = Atmosphere {
        std_density: args.std_density,
        std_temp: args.std_temp,
        lapse_rate: args.lapse_rate,
        lapse_index: args.lapse_index,
    };
    let settings = Settings {
        span_end: args.span_end,
        tolerance: args.tolerance,
    };

    let start = Instant::now();
    let solution = match solve_lead(&projectile, &target, &atmosphere, &settings) {
        Ok(solution) => solution,
        Err(Error::InvalidInput(errors)) => {
            // Same shape as argparse's `parser.error`: the usage, then "lead: error: <message>".
            eprintln!("{}", Args::command().render_usage());
            eprintln!("lead: error: {}", describe_invalid_inputs(&errors));
            return ExitCode::from(EXIT_INVALID_ARGUMENTS);
        }
        Err(error) => {
            eprintln!("lead: error: {error}");
            return ExitCode::from(EXIT_UNEXPECTED);
        }
    };
    let calc_msec = start.elapsed().as_secs_f64() * 1000.0;

    println!("{}", format_result(&solution, calc_msec, args.readable));

    if let Some(path) = &args.dump
        && let Err(message) = write_dump(
            path,
            &projectile,
            &target,
            &atmosphere,
            &settings,
            &solution,
        )
    {
        eprintln!("lead: error: {message}");
        return ExitCode::from(EXIT_UNEXPECTED);
    }

    if solution.status != Status::Hit {
        eprintln!(
            "lead: {}: {}",
            solution.status,
            status_message(solution.status)
        );
    }
    ExitCode::from(exit_code(solution.status))
}
