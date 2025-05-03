use core::fmt;
use rand::Rng;
use std::ops::{Add, Mul, Sub};

pub struct DimensionError(String);

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
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

    pub fn dot(self, rhs: Matrix) -> Result<Matrix, DimensionError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(DimensionError(format!(
                "Attempt to dot multiply matrix of size {}x{} with matrix of size {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }

        let mut product: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                product.data[i][j] = self.data[i][j] * rhs.data[i][j];
            }
        }

        Ok(product)
    }

    pub fn transpose(self) -> Matrix {
        let mut transpose: Matrix = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                transpose.data[j][i] = self.data[i][j];
            }
        }

        transpose
    }

    pub fn map(self, function: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .into_iter()
                .map(|row| row.into_iter().map(function).collect())
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

// Matrix addition
impl Add for Matrix {
    type Output = Result<Self, DimensionError>;

    fn add(self, rhs: Matrix) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(DimensionError(format!(
                "Attempt to add matrix of size {}x{} with matrix of size {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }

        let mut sum: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                sum.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }

        Ok(sum)
    }
}

// Matrix subtraction
impl Sub for Matrix {
    type Output = Result<Self, DimensionError>;

    fn sub(self, rhs: Matrix) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(DimensionError(format!(
                "Attempt to add matrix of size {}x{} with matrix of size {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }

        let mut difference: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                difference.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }

        Ok(difference)
    }
}

// Matrix multiplication
impl Mul<Matrix> for Matrix {
    type Output = Result<Self, DimensionError>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols != rhs.rows {
            return Err(DimensionError(format!(
                "Attempt to multiply matrix of size {}x{} with matrix of size {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }

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

        Ok(product)
    }
}

// Scalar multiplication
impl<T: AsRef<f64>> Mul<T> for Matrix {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut product: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                product.data[i][j] = rhs.as_ref() * self.data[i][j];
            }
        }

        product
    }
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
