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

    test_network(&network);

    network.stohastic_train(inputs, outputs, 30, 1);

    test_network(&network);
}

fn test_network(network: &Network) {
    println!("0, 0 -> {:?}", network.feed_forward(vec![0.0, 0.0]));
    println!("0, 1 -> {:?}", network.feed_forward(vec![0.0, 1.0]));
    println!("1, 0 -> {:?}", network.feed_forward(vec![1.0, 0.0]));
    println!("1, 1 -> {:?}", network.feed_forward(vec![1.0, 1.0]));
}
