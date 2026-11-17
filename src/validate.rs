//! Checks of the inputs, done before any calculation.

use std::fmt;

use crate::types::{Atmosphere, Projectile, Settings, Target};

/// An input value of the calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameter {
    Mass,
    Area,
    DragCoefficient,
    InitialVelocity,
    TargetX,
    TargetY,
    TargetZ,
    VelocityX,
    VelocityY,
    VelocityZ,
    SpanEnd,
    Tolerance,
    StdDensity,
    StdTemp,
    LapseRate,
    LapseIndex,
}

impl Parameter {
    /// Name of the parameter in this library's API.
    pub fn name(self) -> &'static str {
        match self {
            Parameter::Mass => "projectile.mass",
            Parameter::Area => "projectile.area",
            Parameter::DragCoefficient => "projectile.drag_coefficient",
            Parameter::InitialVelocity => "projectile.initial_velocity",
            Parameter::TargetX => "target.position[0]",
            Parameter::TargetY => "target.position[1]",
            Parameter::TargetZ => "target.position[2]",
            Parameter::VelocityX => "target.velocity[0]",
            Parameter::VelocityY => "target.velocity[1]",
            Parameter::VelocityZ => "target.velocity[2]",
            Parameter::SpanEnd => "settings.span_end",
            Parameter::Tolerance => "settings.tolerance",
            Parameter::StdDensity => "atmosphere.std_density",
            Parameter::StdTemp => "atmosphere.std_temp",
            Parameter::LapseRate => "atmosphere.lapse_rate",
            Parameter::LapseIndex => "atmosphere.lapse_index",
        }
    }
}

/// What a value has to satisfy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    Finite,
    Positive,
    NonNegative,
}

/// One reason why the inputs are rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InvalidInput {
    Value {
        parameter: Parameter,
        value: f64,
        requirement: Requirement,
    },
    /// The target is at (0, 0, 0), where the shooter is.
    TargetAtLaunchPoint,
}

impl InvalidInput {
    /// The message for this error, with the parameter named by `name` (e.g. the name of a command-line argument).
    ///
    /// The wording and the number format are the same as in the Python version.
    pub fn describe(&self, name: impl Fn(Parameter) -> String) -> String {
        match *self {
            InvalidInput::Value {
                parameter,
                value,
                requirement,
            } => {
                let condition = match requirement {
                    Requirement::Finite => "a finite number",
                    Requirement::Positive => "greater than 0",
                    Requirement::NonNegative => "0 or greater",
                };
                format!(
                    "{} must be {}: {}",
                    name(parameter),
                    condition,
                    python_float_repr(value)
                )
            }
            InvalidInput::TargetAtLaunchPoint => {
                "the target must not be at the launch point (0, 0, 0)".to_string()
            }
        }
    }
}

impl fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe(|parameter| parameter.name().to_string()))
    }
}

/// Return every reason why the inputs make the calculation meaningless. Empty if they are valid.
///
/// If any value is not finite, only those are reported, because the other checks would be meaningless.
pub fn find_invalid_inputs(
    projectile: &Projectile,
    target: &Target,
    atmosphere: &Atmosphere,
    settings: &Settings,
) -> Vec<InvalidInput> {
    use Parameter as P;
    use Requirement as R;

    let values = [
        (P::Mass, projectile.mass),
        (P::Area, projectile.area),
        (P::DragCoefficient, projectile.drag_coefficient),
        (P::InitialVelocity, projectile.initial_velocity),
        (P::TargetX, target.position[0]),
        (P::TargetY, target.position[1]),
        (P::TargetZ, target.position[2]),
        (P::VelocityX, target.velocity[0]),
        (P::VelocityY, target.velocity[1]),
        (P::VelocityZ, target.velocity[2]),
        (P::SpanEnd, settings.span_end),
        (P::Tolerance, settings.tolerance),
        (P::StdDensity, atmosphere.std_density),
        (P::StdTemp, atmosphere.std_temp),
        (P::LapseRate, atmosphere.lapse_rate),
        (P::LapseIndex, atmosphere.lapse_index),
    ];
    let non_finite: Vec<InvalidInput> = values
        .iter()
        .filter(|(_, value)| !value.is_finite())
        .map(|&(parameter, value)| InvalidInput::Value {
            parameter,
            value,
            requirement: R::Finite,
        })
        .collect();
    if !non_finite.is_empty() {
        return non_finite;
    }

    let rules = [
        (P::Mass, projectile.mass, R::Positive),
        (P::Area, projectile.area, R::Positive),
        (
            P::DragCoefficient,
            projectile.drag_coefficient,
            R::NonNegative,
        ),
        (P::InitialVelocity, projectile.initial_velocity, R::Positive),
        // The shooter is on the ground (z = 0) and the projectile cannot go below it.
        (P::TargetZ, target.position[2], R::NonNegative),
        (P::SpanEnd, settings.span_end, R::Positive),
        (P::Tolerance, settings.tolerance, R::NonNegative),
        (P::StdDensity, atmosphere.std_density, R::NonNegative),
        (P::StdTemp, atmosphere.std_temp, R::Positive),
        (P::LapseRate, atmosphere.lapse_rate, R::NonNegative),
    ];
    let mut errors: Vec<InvalidInput> = rules
        .iter()
        .filter(|&&(_, value, requirement)| match requirement {
            R::Positive => value <= 0.0,
            R::NonNegative => value < 0.0,
            R::Finite => false,
        })
        .map(|&(parameter, value, requirement)| InvalidInput::Value {
            parameter,
            value,
            requirement,
        })
        .collect();

    if target.position == [0.0, 0.0, 0.0] {
        errors.push(InvalidInput::TargetAtLaunchPoint);
    }
    errors
}

/// Format a number the way Python's `repr(float)` does.
fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }

    // `{:e}` gives the shortest digits that round-trip, as repr does: "-1.2345e-5"
    let formatted = format!("{:e}", value.abs());
    let (mantissa, exponent) = formatted
        .split_once('e')
        .expect("`{:e}` always contains an exponent");
    let exponent: i32 = exponent.parse().expect("the exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if value.is_sign_negative() { "-" } else { "" };

    let body = if (-4..16).contains(&exponent) {
        if exponent < 0 {
            format!("0.{}{}", "0".repeat((-exponent - 1) as usize), digits)
        } else {
            let int_len = exponent as usize + 1;
            if digits.len() <= int_len {
                format!("{}{}.0", digits, "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        }
    } else {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mantissa, exponent_sign, exponent.abs())
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DragType;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.0, "-1.0"),
            (10.0, "10.0"),
            (0.1, "0.1"),
            (-0.5, "-0.5"),
            (123.456, "123.456"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-1.5e-7, "-1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.25e100, "1.25e+100"),
            (1e-100, "1e-100"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for (value, expected) in cases {
            assert_eq!(python_float_repr(value), expected);
        }
    }

    fn valid() -> (Projectile, Target, Atmosphere, Settings) {
        (
            Projectile {
                mass: 0.1,
                area: 0.001,
                drag_coefficient: 0.1,
                initial_velocity: 300.0,
                drag_type: DragType::Pressure,
            },
            Target {
                position: [100.0, 100.0, 100.0],
                velocity: [-10.0, 1.0, 0.0],
            },
            Atmosphere::default(),
            Settings::default(),
        )
    }

    #[test]
    fn valid_inputs_pass() {
        let (p, t, a, s) = valid();
        assert!(find_invalid_inputs(&p, &t, &a, &s).is_empty());
    }

    #[test]
    fn non_finite_values_hide_the_other_errors() {
        let (mut p, mut t, a, s) = valid();
        p.mass = -1.0;
        t.velocity[1] = f64::NAN;
        let errors = find_invalid_inputs(&p, &t, &a, &s);
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].to_string(),
            "target.velocity[1] must be a finite number: nan"
        );
    }

    #[test]
    fn every_violated_rule_is_reported() {
        let (mut p, mut t, a, mut s) = valid();
        p.mass = 0.0;
        p.drag_coefficient = -0.5;
        t.position = [0.0, 0.0, 0.0];
        s.span_end = -2.0;
        let messages: Vec<String> = find_invalid_inputs(&p, &t, &a, &s)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            messages,
            [
                "projectile.mass must be greater than 0: 0.0",
                "projectile.drag_coefficient must be 0 or greater: -0.5",
                "settings.span_end must be greater than 0: -2.0",
                "the target must not be at the launch point (0, 0, 0)",
            ]
        );
    }
}
