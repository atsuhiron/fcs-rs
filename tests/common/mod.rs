//! Loading of the reference data written by the Python version (`fcs.scripts.dump_fixtures`).
#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

use fcs::{Atmosphere, DragType, Projectile, Settings, Target};
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[derive(Deserialize)]
pub struct Fixture<T> {
    pub cases: Vec<T>,
}

#[derive(Deserialize)]
pub struct RuntimeConst {
    pub m: f64,
    pub area: f64,
    pub cd: f64,
    pub v0: f64,
    pub span_end: f64,
    pub drag_type: String,
}

#[derive(Deserialize)]
pub struct AtmosphereModel {
    pub std_density: f64,
    pub std_temp: f64,
    pub lapse_rate: f64,
    pub index_: f64,
}

#[derive(Deserialize)]
pub struct Closest {
    pub t: f64,
    pub distance: f64,
    pub target_pos: [f64; 3],
    pub proj_pos: [f64; 3],
}

#[derive(Deserialize)]
pub struct TrajectoryCase {
    pub model: String,
    pub runtime_const: RuntimeConst,
    pub atmosphere: AtmosphereModel,
    pub theta: f64,
    pub phi: f64,
    pub t_pos: [f64; 3],
    pub t_dir: [f64; 3],
    pub t: Vec<f64>,
    /// `[x, y, z]`, each with one value per sample
    pub projectile: [Vec<f64>; 3],
    pub target: [Vec<f64>; 3],
    pub approx_index: usize,
    pub closest: Closest,
}

#[derive(Deserialize)]
pub struct OptimizationCase {
    pub set: String,
    pub runtime_const: RuntimeConst,
    pub atmosphere: AtmosphereModel,
    pub t_pos: [f64; 3],
    pub t_dir: [f64; 3],
    pub tolerance: f64,
    pub status: String,
    pub phi: f64,
    pub theta: f64,
    pub distance: f64,
    pub eta_msec: f64,
    pub format_cl: String,
    pub evaluations: usize,
    /// Every evaluated `(phi, theta, distance)` in order, for the first cases of each set
    pub trace: Option<Vec<[f64; 3]>>,
}

pub fn load<T: DeserializeOwned>(file_name: &str) -> Vec<T> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(file_name);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let fixture: Fixture<T> =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    fixture.cases
}

pub fn projectile(rc: &RuntimeConst) -> Projectile {
    Projectile {
        mass: rc.m,
        area: rc.area,
        drag_coefficient: rc.cd,
        initial_velocity: rc.v0,
        drag_type: match rc.drag_type.as_str() {
            "pressure" => DragType::Pressure,
            "friction" => DragType::Friction,
            other => panic!("unknown drag type: {other}"),
        },
    }
}

pub fn atmosphere(model: &AtmosphereModel) -> Atmosphere {
    Atmosphere {
        std_density: model.std_density,
        std_temp: model.std_temp,
        lapse_rate: model.lapse_rate,
        lapse_index: model.index_,
    }
}

pub fn target(t_pos: [f64; 3], t_dir: [f64; 3]) -> Target {
    Target {
        position: t_pos,
        velocity: t_dir,
    }
}

pub fn settings(rc: &RuntimeConst, tolerance: f64) -> Settings {
    Settings {
        span_end: rc.span_end,
        tolerance,
    }
}
