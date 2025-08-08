use std::f64::consts::E;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub enum ActivationFunction {
    Sigmoid,
    Relu,
}

impl ActivationFunction {
    pub fn get_function(&self) -> fn(f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => SIGMOID.function,
            ActivationFunction::Relu => RELU.function,
        }
    }

    pub fn get_derivative(&self) -> fn(f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => SIGMOID.derivative,
            ActivationFunction::Relu => RELU.derivative,
        }
    }
}

pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| E.powf(-x) / (1.0 + E.powf(-x)).powi(2),
};

pub const RELU: Activation = Activation {
    function: |x| f64::max(0.0, x),
    derivative: |x| ((x > 0.0) as u8).into(),
};
