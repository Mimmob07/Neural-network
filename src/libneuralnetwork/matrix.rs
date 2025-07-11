use rand::Rng;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let data = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

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
                product.data[i][j] = self.data[i][j] * rhs.data[i][j];
            }
        }

        product
    }

    pub fn transpose(&self) -> Matrix {
        let mut transpose: Matrix = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                transpose.data[j][i] = self.data[i][j];
            }
        }

        transpose
    }

    // pub fn map(self, function: Box<dyn Fn(f64) -> f64>) -> Matrix {
    pub fn map(&self, function: impl Fn(f64) -> f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(&function).collect())
                .collect(),
        }
    }
}

impl From<Vec<Vec<f64>>> for Matrix {
    fn from(data: Vec<Vec<f64>>) -> Self {
        Self {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }
}

impl From<Vec<f64>> for Matrix {
    fn from(data: Vec<f64>) -> Self {
        Self {
            rows: 1,
            cols: data.len(),
            data: vec![data],
        }
    }
}

// Matrix addition
impl Add<&Matrix> for Matrix {
    type Output = Self;

    fn add(self, rhs: &Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Attempt to add matrix of size {}x{} with matrix of size {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );

        let mut sum: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                sum.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }

        sum
    }
}

// Matrix subtraction
impl Sub<&Matrix> for Matrix {
    type Output = Self;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Attempt to subtract matrix of size {}x{} with matrix of size {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );

        let mut difference: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                difference.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }

        difference
    }
}

// Matrix multiplication
impl Mul<&Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert!(
            self.cols == rhs.rows,
            "Attempt to multiply matrix of size {}x{} with matrix of size {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );

        let mut product: Matrix = Matrix::zeros(self.rows, rhs.cols);

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum: f64 = 0.0;

                for k in 0..self.cols {
                    sum += self.data[i][k] * rhs.data[k][j];
                }

                product.data[i][j] = sum;
            }
        }

        product
    }
}

// Scalar multiplication
impl Mul<f64> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut product: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                product.data[i][j] = rhs * self.data[i][j];
            }
        }

        product
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
