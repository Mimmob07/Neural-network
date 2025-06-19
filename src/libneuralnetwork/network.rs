use super::{activation::Activation, matrix::Matrix};

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation_function: Activation,
    learning_rate: f64,
    z_history: Vec<Matrix>,
    activation_history: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation_function: Activation, learning_rate: f64) -> Network {
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

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        assert!(
            inputs.len() == self.layers[0],
            "Number of inputs does not match number of neurons in the first layer"
        );

        let mut current_activation = Matrix::from(inputs).transpose();
        self.activation_history.push(current_activation.clone());

        for i in 0..self.layers.len() - 1 {
            // let z = current_activation * &self.weights[i] + &self.biases[i];
            let z = (self.weights[i].clone() * &current_activation) + &self.biases[i];
            self.z_history.push(z.clone());

            current_activation = z.map(self.activation_function.function);
            self.activation_history.push(current_activation.clone());
        }

        current_activation.transpose().data[0].to_owned()
    }

    // No idea if this works as intended
    pub fn back_propogate(
        &self,
        outputs: Vec<f64>,
        expected_outputs: Vec<f64>,
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
            &self.z_history[self.z_history.len() - 1].map(self.activation_function.derivative),
        );

        let l = self.layers.len() - 2;
        nabla_w[l] = error.clone() * &self.activation_history[l].transpose();
        nabla_b[l] = error.clone();

        for l_rev in (0..l).rev() {
            error = (self.weights[l_rev + 1].transpose() * &error)
                .dot(&self.z_history[l_rev].map(self.activation_function.derivative));

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
        training_inputs: Vec<Vec<f64>>,
        training_outputs: Vec<Vec<f64>>,
        epochs: u16,
    ) {
        for i in 1..=epochs {
            if epochs < 100 || i % 100 == 0 {
                println!("Epoch {i} of {epochs}");
            }

            for j in 0..training_inputs.len() {
                self.z_history.clear();
                self.activation_history.clear();

                let train_outputs = self.feed_forward(training_inputs[j].clone());
                let (nabla_w, nabla_b) =
                    self.back_propogate(train_outputs, training_outputs[j].clone());
                self.update_network(nabla_w, nabla_b);
            }
        }
    }
}
