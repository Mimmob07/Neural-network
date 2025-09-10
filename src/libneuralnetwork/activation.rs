use std::f64::consts::E;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub enum ActivationFunction {
    Sigmoid,
    Relu,
}

impl ActivationFunction {
    pub fn get_function(&self) -> fn(f32) -> f32 {
        match self {
            ActivationFunction::Sigmoid => SIGMOID.function,
            ActivationFunction::Relu => RELU.function,
        }
    }

    pub fn get_derivative(&self) -> fn(f32) -> f32 {
        match self {
            ActivationFunction::Sigmoid => SIGMOID.derivative,
            ActivationFunction::Relu => RELU.derivative,
        }
    }
}

pub struct Activation {
    pub function: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x as f64) as f32),
    derivative: |x| (SIGMOID.function)(x) * (1.0 - (SIGMOID.function)(x)),
    // derivative: |x| E.powf(-x as f64) as f32 / (1.0 + E.powf(-x as f64)).powi(2) as f32,
};

pub const RELU: Activation = Activation {
    function: |x| f32::max(0.0, x),
    derivative: |x| ((x > 0.0) as u8).into(),
};
