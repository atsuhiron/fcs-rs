//! The `lead` command: stdout, stderr and exit status, compared with the Python version's `lead`.

mod common;

use std::process::{Command, Output};

use common::OptimizationCase;

const README_ARGS: [&str; 10] = [
    "0.1", "0.001", "0.1", "300", "100", "100", "100", "-10", "1", "0",
];

struct Run {
    stdout: String,
    stderr: String,
    code: i32,
}

fn lead<S: AsRef<str>>(args: &[S]) -> Run {
    let output: Output = Command::new(env!("CARGO_BIN_EXE_lead"))
        .args(args.iter().map(AsRef::as_ref))
        .output()
        .expect("lead runs");
    Run {
        stdout: String::from_utf8(output.stdout).expect("stdout is UTF-8"),
        stderr: String::from_utf8(output.stderr).expect("stderr is UTF-8"),
        code: output.status.code().expect("lead exits with a status"),
    }
}

/// The printed line without its last column, the calculation time.
fn without_calc_time(stdout: &str) -> &str {
    let line = stdout
        .strip_suffix('\n')
        .expect("one line ending in a newline");
    assert!(!line.contains('\n'), "more than one line: {stdout:?}");
    let (rest, calc_time) = line.rsplit_once(' ').expect("columns separated by spaces");
    let calc_time: f64 = calc_time.parse().expect("the calculation time is a number");
    assert!(calc_time >= 0.0);
    rest
}

fn last_line(text: &str) -> &str {
    text.trim_end().lines().last().unwrap_or("")
}

#[test]
fn readme_example() {
    let run = lead(&README_ARGS);
    assert_eq!(without_calc_time(&run.stdout), "46.95 36.46 0.00 601.31");
    assert_eq!(run.stderr, "");
    assert_eq!(run.code, 0);
}

#[test]
fn readable_output() {
    for flag in ["-r", "--readable"] {
        let mut args = vec![flag];
        args.extend(README_ARGS);
        let run = lead(&args);
        let line = run.stdout.strip_suffix('\n').unwrap();
        let rest = line
            .strip_prefix(
                "phi: 46.95°, theta: 36.46°, dist: 0.00, eta_time: 601.31 ms, calc_time: ",
            )
            .unwrap_or_else(|| panic!("unexpected output: {line}"));
        let calc_time = rest.strip_suffix(" ms").expect("ends with the unit");
        assert!(calc_time.parse::<f64>().is_ok());
        assert_eq!(run.code, 0);
    }
}

#[test]
fn options_may_come_after_the_positional_arguments() {
    let mut args = README_ARGS.to_vec();
    args.extend(["--tolerance", "0.5", "--drag-type", "pressure", "-r"]);
    let run = lead(&args);
    assert!(run.stdout.starts_with("phi: 46.95°"));
    assert_eq!(run.code, 0);
}

#[test]
fn out_of_time_prints_the_result_and_exits_with_4() {
    let run = lead(&[
        "0.1", "0.001", "0.1", "300", "4000", "0", "500", "0", "50", "0",
    ]);
    assert_eq!(
        without_calc_time(&run.stdout),
        "7.13 14.48 2403.53 10000.00"
    );
    assert_eq!(
        run.stderr,
        "lead: out_of_time: the projectile does not reach the target within --span-end\n"
    );
    assert_eq!(run.code, 4);
}

#[test]
fn out_of_range_prints_the_result_and_exits_with_3() {
    let run = lead(&[
        "0.1",
        "0.001",
        "0.1",
        "300",
        "300",
        "-1021",
        "0.9",
        "75",
        "-32",
        "-3.7",
        "--drag-type",
        "friction",
        "--tolerance",
        "0.5",
    ]);
    assert_eq!(without_calc_time(&run.stdout), "-61.47 4.15 15.49 4428.18");
    assert_eq!(
        run.stderr,
        "lead: out_of_range: the projectile cannot reach the target\n"
    );
    assert_eq!(run.code, 3);
}

#[test]
fn a_longer_span_end_turns_out_of_time_into_a_hit() {
    let args = [
        "0.1", "0.001", "0.1", "300", "2000", "0", "100", "0", "0", "0",
    ];
    let mut short = args.to_vec();
    short.extend(["--span-end", "3"]);
    assert_eq!(lead(&short).code, 4);
    // Long options can be abbreviated, as with argparse.
    let mut long = args.to_vec();
    long.extend(["--span", "30"]);
    assert_eq!(lead(&long).code, 0);
}

#[test]
fn every_invalid_value_is_reported_at_once() {
    let run = lead(&[
        "0",
        "0.001",
        "-0.5",
        "300",
        "0",
        "0",
        "0",
        "-10",
        "1",
        "0",
        "--span-end",
        "-2",
    ]);
    assert_eq!(run.stdout, "");
    assert_eq!(
        last_line(&run.stderr),
        "lead: error: mass must be greater than 0: 0.0; drag_coefficient must be 0 or greater: -0.5; \
         --span-end must be greater than 0: -2.0; the target must not be at the launch point (0, 0, 0)"
    );
    assert!(run.stderr.to_lowercase().starts_with("usage: lead "));
    assert_eq!(run.code, 2);
}

#[test]
fn each_rule_has_its_message() {
    let cases: [(&[&str], &str); 10] = [
        (
            &["-1", "0.001", "0.1", "300", "1", "1", "1", "0", "0", "0"],
            "mass must be greater than 0: -1.0",
        ),
        (
            &["0.1", "0", "0.1", "300", "1", "1", "1", "0", "0", "0"],
            "area must be greater than 0: 0.0",
        ),
        (
            &["0.1", "0.001", "-0.25", "300", "1", "1", "1", "0", "0", "0"],
            "drag_coefficient must be 0 or greater: -0.25",
        ),
        (
            &["0.1", "0.001", "0.1", "0", "1", "1", "1", "0", "0", "0"],
            "initial_velocity must be greater than 0: 0.0",
        ),
        (
            &["0.1", "0.001", "0.1", "300", "1", "1", "-5", "0", "0", "0"],
            "target_z must be 0 or greater: -5.0",
        ),
        (
            &[
                "0.1",
                "0.001",
                "0.1",
                "300",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
                "--tolerance",
                "-0.00001",
            ],
            "--tolerance must be 0 or greater: -1e-05",
        ),
        (
            &[
                "0.1",
                "0.001",
                "0.1",
                "300",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
                "--std-density",
                "-1",
            ],
            "--std-density must be 0 or greater: -1.0",
        ),
        (
            &[
                "0.1",
                "0.001",
                "0.1",
                "300",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
                "--std-temp",
                "0",
            ],
            "--std-temp must be greater than 0: 0.0",
        ),
        (
            &[
                "0.1",
                "0.001",
                "0.1",
                "300",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
                "--lapse-rate",
                "-0.5",
            ],
            "--lapse-rate must be 0 or greater: -0.5",
        ),
        (
            &["0.1", "0.001", "0.1", "300", "0", "0", "0", "0", "0", "0"],
            "the target must not be at the launch point (0, 0, 0)",
        ),
    ];
    for (args, message) in cases {
        let run = lead(args);
        assert_eq!(last_line(&run.stderr), format!("lead: error: {message}"));
        assert_eq!(run.stdout, "");
        assert_eq!(run.code, 2);
    }
}

#[test]
fn nan_and_inf_are_rejected() {
    let run = lead(&[
        "0.1",
        "0.001",
        "0.1",
        "300",
        "100",
        "nan",
        "100",
        "-10",
        "inf",
        "0",
        "--tolerance",
        "nan",
    ]);
    assert_eq!(run.stdout, "");
    // Only the non-finite values are reported, under the names the Python version uses for them.
    assert_eq!(
        last_line(&run.stderr),
        "lead: error: target_y must be a finite number: nan; direction_y must be a finite number: inf; \
         tolerance must be a finite number: nan"
    );
    assert_eq!(run.code, 2);

    let run = lead(&[
        "0.1",
        "0.001",
        "0.1",
        "300",
        "100",
        "100",
        "100",
        "-10",
        "1",
        "0",
        "--span-end",
        "inf",
    ]);
    assert_eq!(
        last_line(&run.stderr),
        "lead: error: span_end must be a finite number: inf"
    );
    assert_eq!(run.code, 2);
}

#[test]
fn wrong_usage_exits_with_2() {
    let missing = lead(&README_ARGS[..9]);
    assert_eq!(missing.code, 2);
    assert_eq!(missing.stdout, "");
    assert!(missing.stderr.contains("direction_z"));

    let mut not_a_number = README_ARGS.to_vec();
    not_a_number[1] = "abc";
    let run = lead(&not_a_number);
    assert_eq!(run.code, 2);
    assert!(run.stderr.contains("abc"));

    let mut bad_choice = README_ARGS.to_vec();
    bad_choice.extend(["--drag-type", "foo"]);
    let run = lead(&bad_choice);
    assert_eq!(run.code, 2);
    assert!(run.stderr.contains("pressure") && run.stderr.contains("friction"));

    let mut unknown = README_ARGS.to_vec();
    unknown.push("--plot");
    assert_eq!(lead(&unknown).code, 2);
}

#[test]
fn help_exits_with_0() {
    let run = lead(&["--help"]);
    assert_eq!(run.code, 0);
    for name in [
        "mass",
        "direction_z",
        "--span-end",
        "--tolerance",
        "--std-density",
        "--std-temp",
        "--lapse-rate",
        "--lapse-index",
        "--drag-type",
        "--dump",
        "--readable",
        "exit status:",
    ] {
        assert!(run.stdout.contains(name), "{name} is missing from the help");
    }
}

/// Every search of the reference data through the command line, negative positions and velocities included.
#[test]
fn reference_cases_print_the_same_line() {
    let cases: Vec<OptimizationCase> = common::load("optimizations.json");
    let mut identical = 0;
    for (index, case) in cases.iter().enumerate() {
        let rc = &case.runtime_const;
        let mut args: Vec<String> = [rc.m, rc.area, rc.cd, rc.v0]
            .iter()
            .chain(case.t_pos.iter())
            .chain(case.t_dir.iter())
            .map(|v| v.to_string())
            .collect();
        args.extend([
            "--span-end".to_string(),
            rc.span_end.to_string(),
            "--tolerance".to_string(),
            case.tolerance.to_string(),
            "--drag-type".to_string(),
            rc.drag_type.clone(),
        ]);
        let run = lead(&args);
        let line = without_calc_time(&run.stdout);
        let name = format!("case {index} ({}): {}", case.set, args.join(" "));

        // Hit or not must agree. Which of the two failures it is can differ when the projectile lands
        // exactly at span_end, and so can the values printed for a target that is not hit.
        if case.status == "hit" {
            assert_eq!(run.code, 0, "{name}");
            assert_eq!(run.stderr, "", "{name}");
            // The last digit may differ where the value is at the edge of rounding.
            let actual: Vec<f64> = line.split(' ').map(|v| v.parse().unwrap()).collect();
            let expected: Vec<f64> = case
                .format_cl
                .split(' ')
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(actual.len(), 4, "{name}");
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!(
                    (a - e).abs() < 0.0101,
                    "{name}: {line} != {}",
                    case.format_cl
                );
            }
        } else {
            assert!(
                run.code == 3 || run.code == 4,
                "{name}: exit status {}",
                run.code
            );
            assert!(
                run.stderr.starts_with("lead: out_of_"),
                "{name}: {}",
                run.stderr
            );
        }
        if line == case.format_cl {
            identical += 1;
        }
    }
    println!("{identical} of {} lines are identical", cases.len());
    assert!(identical >= cases.len() - 5);
}

#[test]
fn dump_writes_the_trajectory_and_the_closest_point() {
    let path = std::env::temp_dir().join(format!("fcs-lead-dump-{}.json", std::process::id()));
    let mut args: Vec<String> = README_ARGS.iter().map(|s| s.to_string()).collect();
    args.extend(["--dump".to_string(), path.display().to_string()]);
    let run = lead(&args);
    assert_eq!(without_calc_time(&run.stdout), "46.95 36.46 0.00 601.31");
    assert_eq!(run.code, 0);

    let text = std::fs::read_to_string(&path).expect("the dump file exists");
    std::fs::remove_file(&path).ok();
    let dump: serde_json::Value = serde_json::from_str(&text).expect("the dump is JSON");

    assert_eq!(dump["format_version"], 1);
    assert_eq!(
        dump["input"]["target_position"],
        serde_json::json!([100.0, 100.0, 100.0])
    );
    assert_eq!(
        dump["input"]["target_velocity"],
        serde_json::json!([-10.0, 1.0, 0.0])
    );
    assert_eq!(dump["input"]["drag_type"], "pressure");
    assert_eq!(dump["result"]["status"], "hit");
    let phi = dump["result"]["phi"].as_f64().unwrap();
    let theta = dump["result"]["theta"].as_f64().unwrap();
    assert_eq!(
        format!("{:.2} {:.2}", phi.to_degrees(), theta.to_degrees()),
        "46.95 36.46"
    );

    // The trajectory starts at the launch point and has one position per sample on each axis.
    let trajectory = &dump["trajectory"];
    let t = trajectory["t"].as_array().unwrap();
    assert!(t.len() >= 3);
    assert_eq!(t[0], 0.0);
    for key in ["projectile_traj", "target_traj"] {
        let axes = trajectory[key].as_array().unwrap();
        assert_eq!(axes.len(), 3, "{key}");
        for axis in axes {
            assert_eq!(axis.as_array().unwrap().len(), t.len(), "{key}");
        }
    }
    for axis in 0..3 {
        assert_eq!(trajectory["projectile_traj"][axis][0], 0.0);
        assert_eq!(trajectory["target_traj"][axis][0], 100.0);
    }
    assert!((trajectory["approx_index"].as_u64().unwrap() as usize) < t.len());

    // The closest point is the one the printed result describes: a hit, 601.31 ms after launch.
    // It is computed again at the wrapped azimuth (as the Python version does for --plot), so it agrees with
    // the result only up to the rounding noise of the closest distance.
    let closest = &dump["closest"];
    let distance = closest["distance"].as_f64().unwrap();
    assert!((distance - dump["result"]["distance"].as_f64().unwrap()).abs() < 1e-4);
    let eta = closest["t"].as_f64().unwrap() * 1000.0;
    assert_eq!(format!("{eta:.2}"), "601.31");
    assert!((eta - dump["result"]["eta_msec"].as_f64().unwrap()).abs() < 1e-3);
    for axis in 0..3 {
        let target = closest["target_pos"][axis].as_f64().unwrap();
        let projectile = closest["proj_pos"][axis].as_f64().unwrap();
        assert!((target - projectile).abs() < 1e-3);
    }
    // The target has moved from (100, 100, 100) at (-10, 1, 0) m/s for 0.6 s.
    assert!((closest["target_pos"][0].as_f64().unwrap() - 93.98691563224356).abs() < 1e-3);
}

#[test]
fn dump_to_an_unwritable_path_is_an_unexpected_error() {
    let path = std::env::temp_dir()
        .join("fcs-lead-no-such-directory")
        .join("dump.json");
    let mut args: Vec<String> = README_ARGS.iter().map(|s| s.to_string()).collect();
    args.extend(["--dump".to_string(), path.display().to_string()]);
    let run = lead(&args);
    assert_eq!(run.code, 1);
    assert!(run.stderr.starts_with("lead: error: cannot write "));
}
