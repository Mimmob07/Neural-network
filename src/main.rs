use std::io;

use libneuralnetwork::{activation::SIGMOID, network::Network};
use mnist_unpacker::{unpack, MnistImages};

mod libneuralnetwork;
mod mnist_unpacker;

fn main() -> io::Result<()> {
    let train_data: MnistImages = unpack(
        "../mnist/train-images.idx3-ubyte",
        "../mnist/train-labels.idx1-ubyte",
    )?;

    let train_images = train_data
        .images
        .iter()
        .map(|vec| vec.iter().map(|val| *val as f64).collect())
        .collect();
    let train_labels = train_data
        .labels
        .iter()
        .map(|val| {
            let mut tmp = vec![0f64; 10];
            tmp[*val as usize] = 1.0;
            tmp
        })
        .collect();

    let mut network = Network::new(vec![784, 16, 16, 10], SIGMOID, 0.01);
    network.train(train_images, train_labels, 100);

    let test_data: MnistImages = unpack(
        "../mnist/t10k-images.idx3-ubyte",
        "../mnist/t10k-labels.idx1-ubyte",
    )?;

    let test_images: Vec<Vec<f64>> = test_data
        .images
        .iter()
        .map(|vec| vec.iter().map(|val| *val as f64).collect())
        .collect();
    let test_labels: Vec<Vec<f64>> = test_data
        .labels
        .iter()
        .map(|val| {
            let mut tmp = vec![0f64; 10];
            tmp[*val as usize] = 1.0;
            tmp
        })
        .collect();
    println!("Score: {}/10_000", network.test(test_images, test_labels));

    Ok(())
}
