use std::io;

use libneuralnetwork::{activation::ActivationFunction, network::Network};
use mnist_unpacker::{unpack, MnistImages};

mod libneuralnetwork;
mod mnist_unpacker;

fn main() -> io::Result<()> {
    let train_data: MnistImages = unpack(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte",
    )?;

    let train_images = train_data
        .images
        .iter()
        .map(|vec| vec.iter().map(|val| *val as f32).collect())
        .collect();
    let train_labels = train_data
        .labels
        .iter()
        .map(|val| {
            let mut tmp = vec![0f32; 10];
            tmp[*val as usize] = 1.0;
            tmp
        })
        .collect();

    let mut network = Network::new(vec![784, 30, 10], ActivationFunction::Sigmoid, 1.0);
    // network.stochastic_train(train_images, train_labels, 30, 10);
    network.train(train_images, train_labels, 100);

    let test_data: MnistImages = unpack(
        "mnist/t10k-images.idx3-ubyte",
        "mnist/t10k-labels.idx1-ubyte",
    )?;

    let test_images: Vec<Vec<f32>> = test_data
        .images
        .iter()
        .map(|vec| vec.iter().map(|val| *val as f32).collect())
        .collect();
    let test_labels: Vec<Vec<f32>> = test_data
        .labels
        .iter()
        .map(|val| {
            let mut tmp = vec![0f32; 10];
            tmp[*val as usize] = 1.0;
            tmp
        })
        .collect();
    println!("Score: {}/10_000", network.test(test_images, test_labels));

    network.save("mnist_100_epochs_1.0")?;

    Ok(())
}
