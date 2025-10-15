use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{
    f32::consts::E,
    fs::{read, File},
    io::{self, Write},
    time::Instant,
};

use super::{activation::ActivationFunction, matrix::Matrix};

#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation_function: ActivationFunction,
    learning_rate: f32,
    #[serde(skip)]
    z_history: Vec<Matrix>,
    #[serde(skip)]
    activation_history: Vec<Matrix>,
}

#[allow(dead_code)]
impl Network {
    pub fn new(
        layers: Vec<usize>,
        activation_function: ActivationFunction,
        learning_rate: f32,
    ) -> Network {
        let mut weights: Vec<Matrix> = Vec::new();
        let mut biases: Vec<Matrix> = Vec::new();

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            activation_function,
            learning_rate,
            z_history: Vec::new(),
            activation_history: Vec::new(),
        }
    }

    pub fn from_file<T: AsRef<str>>(filename: T) -> io::Result<Self> {
        let start_time = Instant::now();
        let buf = read(filename.as_ref())?;
        let network = rmp_serde::from_slice(&buf).unwrap();

        println!(
            "Took {}s to load network",
            start_time.elapsed().as_millis() as f64 / 1000.0
        );

        Ok(network)
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert!(
            inputs.len() == self.layers[0],
            "Number of inputs does not match number of neurons in the first layer"
        );

        let mut activation = Matrix::from(inputs).transpose();

        for i in 0..self.layers.len() - 1 {
            activation = ((self.weights[i].clone() * &activation) + &self.biases[i].clone())
                .map(self.activation_function.get_function());
        }

        activation.transpose()[0].to_owned()
    }

    fn feed_forward_and_record(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        assert!(
            inputs.len() == self.layers[0],
            "Number of inputs does not match number of neurons in the first layer"
        );

        let mut current_activation = Matrix::from(inputs).transpose();
        self.activation_history.push(current_activation.clone());

        for i in 0..self.layers.len() - 1 {
            let z = (self.weights[i].clone() * &current_activation) + &self.biases[i];
            self.z_history.push(z.clone());

            current_activation = z.map(self.activation_function.get_function());
            self.activation_history.push(current_activation.clone());
        }

        current_activation.transpose()[0].to_owned()
    }

    fn back_propogate(
        &self,
        outputs: Vec<f32>,
        expected_outputs: Vec<f32>,
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        assert!(
            expected_outputs.len() == *self.layers.last().unwrap(),
            "Number of expected outputs does not match number of neurons in the last layer"
        );

        let outputs_matrix = Matrix::from(outputs).transpose();
        let expected_outputs_matrix = Matrix::from(expected_outputs).transpose();

        let mut nabla_w: Vec<Matrix> = vec![Matrix::zeros(0, 0); self.weights.len()];
        let mut nabla_b: Vec<Matrix> = vec![Matrix::zeros(0, 0); self.biases.len()];

        let mut error = ((outputs_matrix - &expected_outputs_matrix) * 2.0).dot(
            &self.z_history[self.z_history.len() - 1]
                .map(self.activation_function.get_derivative()),
        );

        let l = self.layers.len() - 2;
        nabla_w[l] = error.clone() * &self.activation_history[l].transpose();
        nabla_b[l] = error.clone();

        for l_rev in (0..l).rev() {
            error = (self.weights[l_rev + 1].transpose() * &error)
                .dot(&self.z_history[l_rev].map(self.activation_function.get_derivative()));

            nabla_w[l_rev] = error.clone() * &self.activation_history[l_rev].transpose();
            nabla_b[l_rev] = error.clone();
        }

        (nabla_w, nabla_b)
    }

    pub fn update_network(&mut self, nabla_w: Vec<Matrix>, nabla_b: Vec<Matrix>) {
        for i in 0..self.layers.len() - 1 {
            let adjusted_nabla_w = nabla_w[i].clone() * self.learning_rate;
            let adjusted_nabla_b = nabla_b[i].clone() * self.learning_rate;

            self.weights[i] = self.weights[i].clone() - &adjusted_nabla_w;
            self.biases[i] = self.biases[i].clone() - &adjusted_nabla_b;
        }
    }

    pub fn train(
        &mut self,
        training_inputs: Vec<Vec<f32>>,
        training_outputs: Vec<Vec<f32>>,
        epochs: u16,
    ) {
        let mut epoch_durations: Vec<u64> = Vec::new();

        for i in 1..=epochs {
            let start_time = Instant::now();
            if epochs <= 100 || i % 100 == 0 {
                println!("Epoch {i} of {epochs}");
            }

            for j in 0..training_inputs.len() {
                self.z_history.clear();
                self.activation_history.clear();

                let train_outputs = self.feed_forward_and_record(training_inputs[j].clone());
                let (nabla_w, nabla_b) =
                    self.back_propogate(train_outputs, training_outputs[j].clone());
                self.update_network(nabla_w, nabla_b);
            }

            let elapsed_time = start_time.elapsed().as_secs();
            if epochs <= 100 || i % 100 == 0 {
                println!("Epoch {i} took {elapsed_time}s");
            }
            epoch_durations.push(elapsed_time);
        }

        println!(
            "Average time to complete one epoch {}s",
            epoch_durations.iter().sum::<u64>() / epoch_durations.len() as u64
        );
    }

    pub fn stochastic_train(
        &mut self,
        training_inputs: Vec<Vec<f32>>,
        training_outputs: Vec<Vec<f32>>,
        epochs: u16,
        mini_batch_size: usize,
    ) {
        let mut training_data: Vec<(Vec<f32>, Vec<f32>)> =
            training_inputs.into_iter().zip(training_outputs).collect();

        for i in 1..=epochs {
            if epochs <= 100 || i % 100 == 0 {
                println!("Epoch {i} of {epochs}");
            }

            training_data.shuffle(&mut rand::rng());
            let mini_batches = training_data.windows(mini_batch_size);

            for mini_batch in mini_batches {
                // Update mini branch
                let mut weight_gradient: Vec<Matrix> = Vec::new();
                let mut bias_gradient: Vec<Matrix> = Vec::new();

                self.weights
                    .iter()
                    .for_each(|x| weight_gradient.push(Matrix::zeros(x.rows, x.cols)));
                self.biases
                    .iter()
                    .for_each(|x| bias_gradient.push(Matrix::zeros(x.rows, x.cols)));

                // Create sum of gradients
                for (inputs, labels) in mini_batch.iter().cloned() {
                    let output = self.feed_forward_and_record(inputs);
                    let (naive_weight_gradient, naive_bias_gradient) =
                        self.back_propogate(output, labels);

                    for j in 0..weight_gradient.len() {
                        weight_gradient[j] = weight_gradient[j].clone() + &naive_weight_gradient[j];
                        bias_gradient[j] = bias_gradient[j].clone() + &naive_bias_gradient[j];
                    }
                }

                // Gradient descent using average gradient
                for j in 0..self.layers.len() - 1 {
                    self.weights[j] = self.weights[j].clone()
                        - &(weight_gradient[j].clone()
                            * (self.learning_rate / mini_batch.len() as f32));
                    self.biases[j] = self.biases[j].clone()
                        - &(bias_gradient[j].clone()
                            * (self.learning_rate / mini_batch.len() as f32));
                }
            }
        }
    }

    pub fn test(&self, inputs_set: Vec<Vec<f32>>, expected_outputs_set: Vec<Vec<f32>>) -> usize {
        assert!(inputs_set.len() == expected_outputs_set.len());
        let start_time = Instant::now();
        let mut passes = 0;

        for (i, (inputs, label)) in inputs_set.iter().zip(expected_outputs_set).enumerate() {
            // let results = Network::softmax(self.feed_forward(inputs.clone()));
            let results = self.feed_forward(inputs.clone());
            let clamped_results: Vec<f32> = results.iter().map(|val| val.round()).collect();

            println!("Test {} of {}", i, inputs_set.len());
            println!("Results : {results:?}");
            println!("Clamped : {clamped_results:?}");
            println!("Expected: {label:?}");

            if clamped_results == label {
                passes += 1;
            }
        }

        println!(
            "Took {}s to test network",
            start_time.elapsed().as_millis() as f64 / 1000.0
        );

        passes
    }

    fn softmax(input_layer: Vec<f32>) -> Vec<f32> {
        let sum: f32 = input_layer.iter().map(|x| E.powf(*x)).sum();
        input_layer.iter().map(|x| E.powf(*x) / sum).collect()
    }

    pub fn save<T: AsRef<str>>(&self, filename: T) -> io::Result<()> {
        let start_time = Instant::now();

        let mut file = File::create(filename.as_ref())?;
        let mut buf = Vec::new();
        self.serialize(&mut rmp_serde::Serializer::new(&mut buf))
            .unwrap();

        println!(
            "Took {}s to serlialize network",
            start_time.elapsed().as_millis() as f64 / 1000.0
        );

        file.write_all(&buf)?;
        Ok(())
    }
}
