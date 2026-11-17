# fcs-rs

Fire-control (lead) calculator. Given the specifications of a projectile and the position and constant velocity
of a target, it finds the azimuth and elevation whose trajectory, with altitude-dependent air drag,
passes closest to the target.

This is a Rust port of [fcs](https://github.com/atsuhiron/fcs), which is written in Python.
It runs the same algorithm, as a single binary that needs no Python, and is over 100 times faster.
**The mathematical model and its derivation are described in the
[README of the Python version](https://github.com/atsuhiron/fcs#readme)**; they are not repeated here.

## Download

Prebuilt binaries are attached to the [releases](https://github.com/atsuhiron/fcs-rs/releases),
for Windows x86-64 (`x86_64-pc-windows-msvc`), Linux x86-64 (`x86_64-unknown-linux-musl`)
and macOS on Apple silicon (`aarch64-apple-darwin`).
Each archive holds the single executable `lead` (`lead.exe` on Windows), which needs nothing else installed:
the Windows one links the C runtime statically, and the Linux one is a static musl binary.

## Build

```shell
cargo build --release
```

The binary is `target/release/lead` (`lead.exe` on Windows).

### Release

Pushing a tag `v<version>`, where the version is the one in `Cargo.toml`, runs `.github/workflows/release.yml`.
It builds the three binaries, checks that each prints the first example under Usage, and publishes them as a release.
Running the workflow by hand from the Actions tab does the same without publishing;
the archives are then kept as artifacts of the run.

## Usage

```shell
$ lead 0.1 0.001 0.1 300 100 100 100 -10 1 0
46.95 36.46 0.00 601.31 1.24
```

The positional arguments are, in order:

| Argument | Meaning |
|---|---|
| `mass` | Mass of the projectile (kg) |
| `area` | Cross-sectional area of the projectile (m^2) |
| `drag_coefficient` | Drag coefficient of the projectile |
| `initial_velocity` | Muzzle velocity (m/s) |
| `target_x` `target_y` `target_z` | Position of the target at launch (m). The shooter is at the origin, z is up |
| `direction_x` `direction_y` `direction_z` | Velocity of the target (m/s) |

The output is the azimuth (deg), the elevation (deg), the closest distance to the target (m),
the time from launch to the closest approach (ms) and the calculation time (ms).
`-r` (`--readable`) prints the same values with labels:

```shell
$ lead -r 0.1 0.001 0.1 300 100 100 100 -10 1 0
phi: 46.95°, theta: 36.46°, dist: 0.00, eta_time: 601.31 ms, calc_time: 1.24 ms
```

| Option | Default | Meaning |
|---|---|---|
| `--span-end` | 10 | Maximum flight time to simulate (s) |
| `--tolerance` | 0.1 | Closest distance to the target that counts as a hit (m) |
| `--drag-type` | `pressure` | `pressure` (drag proportional to the square of the airspeed) or `friction` (proportional to the airspeed) |
| `--std-density` | 1.225 | Air density at z = 0 (kg/m^3) |
| `--std-temp` | 288.15 | Temperature at z = 0 (K) |
| `--lapse-rate` | 0.0065 | Temperature lapse rate (K/m) |
| `--lapse-index` | 5.256 | Exponent of the density formula |
| `--dump <PATH>` | | Write the trajectory at the result and the closest point to a JSON file |

### Exit status

| Exit status | Name | Meaning |
|---|---|---|
| 0 | `hit` | The closest distance is within `--tolerance`. |
| 1 | | Unexpected error. |
| 2 | | Invalid arguments. They are checked before the calculation starts, and every problem is reported at once. |
| 3 | `out_of_range` | The projectile lands without reaching the target. |
| 4 | `out_of_time` | The projectile is still in flight at `--span-end` and has not reached the target. A larger `--span-end` may give a hit. |

For 3 and 4 the result for the closest approach is still printed to stdout, and `lead: <name>: <message>` is written to stderr.

### Plotting data

There is no `--plot`; plotting stays in Python. `--dump <PATH>` writes what a plot needs:

```json
{
  "format_version": 1,
  "input": { "mass": 0.1, "...": "the arguments" },
  "result": { "status": "hit", "phi": 0.819, "theta": 0.636, "distance": 0.0, "eta_msec": 601.3 },
  "trajectory": {
    "t": [0.0, "..."],
    "projectile_traj": [["x ..."], ["y ..."], ["z ..."]],
    "target_traj": [["x ..."], ["y ..."], ["z ..."]],
    "approx_index": 13
  },
  "closest": { "t": 0.601, "distance": 0.0, "target_pos": [0, 0, 0], "proj_pos": [0, 0, 0] }
}
```

Angles are in radians. `trajectory` and `closest` have the fields of `TrajectoryResult` and `ApproximateResult`
of the Python version, which its `plot_3d` takes.

## Library

The package is also a library crate, `fcs`. Every function checks its inputs and returns a `Result`.

```rust
use fcs::{Atmosphere, DragType, Projectile, Settings, Target, solve_lead};

let projectile = Projectile {
    mass: 0.1,
    area: 0.001,
    drag_coefficient: 0.1,
    initial_velocity: 300.0,
    drag_type: DragType::Pressure,
};
let target = Target {
    position: [100.0, 100.0, 100.0],
    velocity: [-10.0, 1.0, 0.0],
};
let solution = solve_lead(&projectile, &target, &Atmosphere::default(), &Settings::default())?;
println!("{:?} {:?}", solution.status, solution.aim);
```

`solve_trajectory` and `find_closest_point` give the trajectory for one launch direction and the closest approach on it.

## Relation to the Python version

The Python version is the reference implementation. This crate ports its algorithm without changes,
including the numerical routines it takes from SciPy, which are written out here by hand:
the RK45 integrator of `solve_ivp` with its event handling, and Powell's method of `minimize`.
The only dependencies are `clap`, `serde` and `serde_json`.

`lead` takes the same arguments and gives the same output, exit statuses and messages as the Python `lead`, except that

- `--plot` is replaced by `--dump`;
- messages about wrong usage (a missing argument, a value that is not a number) are worded by `clap`, not `argparse`.
  The exit status is 2 in both.

The tests compare the crate with reference data written by the Python version (`tests/fixtures/`,
from `uv run python -m fcs.scripts.dump_fixtures` at commit `af0c85e` with NumPy 2.4.1 and SciPy 1.17.0).

| Compared | Cases | Result |
|---|---|---|
| Trajectories | 396 | Same samples, to the last bit (on Windows x86-64, where the data was generated) |
| Closest point on each trajectory | 396 | Within 5e-11 m and 6e-14 s |
| Aim search, hit or not | 282 | Same for every case |
| Aim search, the hits | 249 | Within 0.0002° and 0.0002 ms |
| Printed line | 282 | Identical for 281; the other differs by 0.01 ms at the edge of rounding |
| Powell's method fed the Python objective values | 2985 points | Requests the same points, to the last bit |

Two things are not the same:

- **The number of trajectories computed in a search** differs in 274 of the 282 cases (by 10 % on average, in either direction; the totals differ by 0.7 %).
  Near a hit the closest distance is down at its rounding noise (about 1e-6 m), and the line search follows that noise.
  The noise is not the same because the Python version fits with `np.polyfit` and finds roots with `np.roots`
  (LAPACK routines), where this crate uses closed forms. With those two calls replaced by the same closed forms,
  the Python version gives bit-identical results and counts for all 282 cases.
- **`out_of_range` or `out_of_time`** can differ for a target that is not hit when the best trajectory lands exactly
  at `--span-end` (1 of 33 such cases: landing at 9.99999999999998 s against a cut-off at 10 s).

## Speed

Measured on one Windows PC (2026-11-17), release build against the Python version at `af0c85e` (Python 3.13).
The targets are those of the reference data, with the projectile of the example above.

Calculation time (the last column of the output), median and maximum over the targets:

| Targets | Python | Rust | Ratio of medians |
|---|---|---|---|
| The example above | 158 ms | 1.24 ms | 127 |
| Near, within 1.2 km (100 targets) | 197 ms (max 378) | 1.36 ms (max 2.39) | 145 |
| Overhead (40 targets) | 183 ms (max 321) | 1.24 ms (max 1.65) | 148 |
| Out of reach, 2.5-6 km (20 targets) | 198 ms (max 343) | 1.28 ms (max 2.23) | 155 |

Time from starting the command to its exit, for the example above (median of 10 runs):

| Command | Time |
|---|---|
| `uv run lead ...` (Python) | 1016 ms |
| `.venv\Scripts\lead.exe ...` (Python, without `uv run`) | 967 ms |
| `lead.exe ...` (Rust) | 49 ms |

The last table was measured from a shell script, where starting a process that does nothing takes 20 ms.

## License

Apache-2.0, the same as the Python version.
