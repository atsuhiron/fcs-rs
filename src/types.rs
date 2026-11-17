//! Data types shared by the layers of the calculation.

use std::fmt;

use crate::validate::InvalidInput;

/// A position or a velocity, `[x, y, z]`. z is up and the shooter is at the origin.
pub type Vector3 = [f64; 3];

/// How the drag force depends on the airspeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DragType {
    /// Pressure drag, proportional to the square of the airspeed.
    #[default]
    Pressure,
    /// Friction drag, proportional to the airspeed.
    Friction,
}

/// Specifications of the projectile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projectile {
    /// Mass (kg).
    pub mass: f64,
    /// Cross-sectional area (m^2).
    pub area: f64,
    /// Drag coefficient.
    pub drag_coefficient: f64,
    /// Muzzle velocity (m/s).
    pub initial_velocity: f64,
    pub drag_type: DragType,
}

/// A target moving at a constant velocity. `position` is where it is at launch (t = 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// Position at launch (m).
    pub position: Vector3,
    /// Velocity (m/s).
    pub velocity: Vector3,
}

impl Target {
    /// Position of the target at time `t`.
    pub fn position_at(&self, t: f64) -> Vector3 {
        // Same order of operations as the Python version (move + init).
        [
            self.velocity[0] * t + self.position[0],
            self.velocity[1] * t + self.position[1],
            self.velocity[2] * t + self.position[2],
        ]
    }
}

/// International Standard Atmosphere model: `density(z) = std_density * (1 - lapse_rate * z / std_temp) ^ lapse_index`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atmosphere {
    /// Density at z = 0 (kg/m^3).
    pub std_density: f64,
    /// Temperature at z = 0 (K).
    pub std_temp: f64,
    /// Temperature lapse rate (K/m).
    pub lapse_rate: f64,
    /// Exponent of the density formula.
    pub lapse_index: f64,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            std_density: 1.225,
            std_temp: 288.15,
            lapse_rate: 0.0065,
            lapse_index: 5.256,
        }
    }
}

impl Atmosphere {
    /// Air density at altitude `z`.
    pub fn density(&self, z: f64) -> f64 {
        let coef = (1.0 - (z * self.lapse_rate / self.std_temp)).powf(self.lapse_index);
        self.std_density * coef
    }
}

/// Settings of the calculation that are not properties of the projectile, the target or the air.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Maximum flight time to simulate (s). Launch is always at t = 0.
    pub span_end: f64,
    /// Closest distance to the target that counts as a hit (m).
    pub tolerance: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            span_end: 10.0,
            tolerance: 0.1,
        }
    }
}

/// Launch direction, in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aim {
    /// Azimuth (phi): angle from the x axis towards the y axis.
    pub azimuth: f64,
    /// Elevation (theta): angle from the horizontal plane.
    pub elevation: f64,
}

/// A trajectory in absolute coordinates, sampled at the steps the integrator took.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// Sample times (s). The first is 0 (launch).
    pub t: Vec<f64>,
    /// Position of the projectile at each sample.
    pub projectile: Vec<Vector3>,
    /// Position of the target at each sample.
    pub target: Vec<Vector3>,
    /// Index of the sample where the projectile is closest to the target.
    pub closest_sample: usize,
}

/// The closest approach of the projectile to the target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestPoint {
    /// Time of the closest approach (s).
    pub t: f64,
    /// Distance between the projectile and the target at that time (m).
    pub distance: f64,
    pub target_pos: Vector3,
    pub projectile_pos: Vector3,
}

/// Outcome of the aim search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The closest distance is within the tolerance.
    Hit,
    /// Not a hit, and the trajectory was cut off at `span_end` before landing. A larger `span_end` may give a hit.
    OutOfTime,
    /// Not a hit, and the projectile landed before `span_end`. This projectile cannot reach the target.
    OutOfRange,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Hit => "hit",
            Status::OutOfTime => "out_of_time",
            Status::OutOfRange => "out_of_range",
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of the aim search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solution {
    /// Launch direction: azimuth in (-pi, pi], elevation in [0, pi/2].
    pub aim: Aim,
    /// Closest distance between the projectile and the target (m).
    pub distance: f64,
    /// Time from launch to the closest approach (ms).
    pub eta_msec: f64,
    pub status: Status,
    /// Number of trajectories computed, including the final check at the optimum.
    pub evaluations: usize,
}

/// Errors of the public API.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The inputs make the calculation meaningless. Nothing was calculated.
    InvalidInput(Vec<InvalidInput>),
    /// A numerical routine failed. The Python version raises an exception in the same situations.
    Numerical(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(errors) => {
                let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
                f.write_str(&messages.join("; "))
            }
            Error::Numerical(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}
