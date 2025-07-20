use libneuralnetwork::{activation::SIGMOID, network::Network};

pub mod libneuralnetwork;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let mut network = Network::new(vec![2, 3, 1], SIGMOID, 1.0);
    network.train(inputs.clone(), outputs.clone(), 10000);

    println!(
        "Score: {}/{}",
        network.test(
            inputs.clone(),
            vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]
        ),
        inputs.len()
    );

    network.stohastic_train(inputs.clone(), outputs, 30, 1);

    println!(
        "Score: {}/{}",
        network.test(
            inputs.clone(),
            vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]
        ),
        inputs.len()
    );
}
