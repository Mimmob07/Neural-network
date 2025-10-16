use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    ops::{Add, Index, Mul, Sub},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let data = (0..rows * cols)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect::<Vec<f32>>();

        Matrix { rows, cols, data }
    }

    pub fn dot(&self, rhs: &Matrix) -> Matrix {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Attempt to dot multiply matrix of size {}x{} with matrix of size {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );

        let mut product: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                product.data[i * self.cols + j] =
                    self.data[i * self.cols + j] * rhs.data[i * self.cols + j];
            }
        }

        product
    }

    pub fn transpose(&self) -> Matrix {
        let mut transpose: Matrix = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                transpose.data[j * transpose.cols + i] = self.data[i * self.cols + j];
            }
        }

        transpose
    }

    // pub fn map(self, function: Box<dyn Fn(f64) -> f64>) -> Matrix {
    pub fn map(&self, function: impl Fn(f32) -> f32) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone().into_iter().map(&function).collect(),
        }
    }
}

impl From<&Vec<Vec<f32>>> for Matrix {
    fn from(data: &Vec<Vec<f32>>) -> Self {
        let flat_data = data.iter().flatten().copied().collect::<Vec<f32>>();

        Self {
            rows: data.len(),
            cols: data[0].len(),
            data: flat_data,
        }
    }
}

impl From<&Vec<f32>> for Matrix {
    fn from(data: &Vec<f32>) -> Self {
        Self {
            rows: 1,
            cols: data.len(),
            data: data.to_vec(),
        }
    }
}

impl From<Vec<Vec<f32>>> for Matrix {
    fn from(data: Vec<Vec<f32>>) -> Self {
        let flat_data = data.iter().flatten().copied().collect::<Vec<f32>>();

        Self {
            rows: data.len(),
            cols: data[0].len(),
            data: flat_data,
        }
    }
}

impl From<Vec<f32>> for Matrix {
    fn from(data: Vec<f32>) -> Self {
        Self {
            rows: 1,
            cols: data.len(),
            data,
        }
    }
}

// Matrix addition
fn matrix_add(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    assert!(
        lhs.rows == rhs.rows && lhs.cols == rhs.cols,
        "Attempt to add matrix of size {}x{} with matrix of size {}x{}",
        lhs.rows,
        lhs.cols,
        rhs.rows,
        rhs.cols
    );

    let mut sum: Matrix = Matrix::zeros(lhs.rows, lhs.cols);

    for i in 0..lhs.rows {
        for j in 0..lhs.cols {
            sum.data[i * lhs.cols + j] = lhs.data[i * lhs.cols + j] + rhs.data[i * lhs.cols + j];
        }
    }

    sum
}

impl Add<&Matrix> for Matrix {
    type Output = Self;

    fn add(self, rhs: &Matrix) -> Self::Output {
        matrix_add(&self, rhs)
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        matrix_add(self, rhs)
    }
}

// Matrix subtraction
fn matrix_subtract(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    assert!(
        lhs.rows == rhs.rows && lhs.cols == rhs.cols,
        "Attempt to subtract matrix of size {}x{} with matrix of size {}x{}",
        lhs.rows,
        lhs.cols,
        rhs.rows,
        rhs.cols
    );

    let mut difference: Matrix = Matrix::zeros(lhs.rows, lhs.cols);

    for i in 0..lhs.rows {
        for j in 0..lhs.cols {
            difference.data[i * lhs.cols + j] =
                lhs.data[i * lhs.cols + j] - rhs.data[i * lhs.cols + j];
        }
    }

    difference
}

impl Sub<&Matrix> for Matrix {
    type Output = Self;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        matrix_subtract(&self, rhs)
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        matrix_subtract(self, rhs)
    }
}

// Matrix multiplication
fn matrix_multiply(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    assert!(
        lhs.cols == rhs.rows,
        "Attempt to multiply matrix of size {}x{} with matrix of size {}x{}",
        lhs.rows,
        lhs.cols,
        rhs.rows,
        rhs.cols
    );

    let mut product: Matrix = Matrix::zeros(lhs.rows, rhs.cols);

    product
        .data
        .par_chunks_mut(rhs.cols)
        .enumerate()
        .for_each(|(i, row)| {
            for (j, val) in row.iter_mut().enumerate() {
                let mut sum: f32 = 0.0;

                for k in 0..lhs.cols {
                    sum += lhs.data[i * lhs.cols + k] * rhs.data[k * rhs.cols + j];
                }

                *val = sum;
            }
        });

    product
}

impl Mul<&Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        matrix_multiply(&self, rhs)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        matrix_multiply(self, rhs)
    }
}

// Scalar multiplication
fn matrix_scalar_multiplication(lhs: &Matrix, rhs: f32) -> Matrix {
    let mut product: Matrix = Matrix::zeros(lhs.rows, lhs.cols);

    for i in 0..lhs.rows {
        for j in 0..lhs.cols {
            product.data[i * lhs.cols + j] = rhs * lhs.data[i * lhs.cols + j];
        }
    }

    product
}

impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        matrix_scalar_multiplication(&self, rhs)
    }
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Self::Output {
        matrix_scalar_multiplication(&self, rhs)
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let linear_index = index * self.cols;
        &self.data[linear_index..linear_index + self.cols]
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix:").unwrap();
        for i in 0..self.data.len() {
            writeln!(f, "\t{:?}", self.data[i]).unwrap();
        }
        Ok(())
    }
}
