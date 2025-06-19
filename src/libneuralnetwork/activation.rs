use std::f64::consts::E;

pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| E.powf(-x) / (1.0 + E.powf(-x)).powi(2),
    // derivative: |y| y * (1.0 - y),
};
