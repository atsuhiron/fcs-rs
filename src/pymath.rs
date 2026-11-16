//! Arithmetic that has to round the way the Python version does.

use std::hint::black_box;

/// `x ** 2` on a Python or NumPy scalar.
///
/// Python evaluates it with the C runtime's `pow(x, 2.0)`, which is not always the correctly rounded square:
/// with the Windows runtime it differs from `x * x` in the last bit for about 1 in 2000 values.
/// That bit matters because the step sizes of the integrator are sensitive to it (see `ode::stage_dot`).
pub(crate) fn square(x: f64) -> f64 {
    pow(x, 2.0)
}

/// The C runtime's `pow(x, y)`, kept as a call: the compiler would otherwise turn small integer exponents
/// into multiplications.
pub(crate) fn pow(x: f64, y: f64) -> f64 {
    x.powf(black_box(y))
}
