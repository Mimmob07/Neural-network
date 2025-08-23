use ocl::{flags, Buffer, ProQue};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    fs,
    ops::{Add, Index, Mul, Sub},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

// TODO:
// ✅ Flatten Matrix.data into Vec<f64> using row major order
// ✅ Switch from f64 to f32 for better gpu compatibility
// ✅ Create opencl kernels
//  - Implement Matrix * Matrix using kernels

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
                sum.data[i * self.cols + j] =
                    self.data[i * self.cols + j] + rhs.data[i * self.cols + j];
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
                difference.data[i * self.cols + j] =
                    self.data[i * self.cols + j] - rhs.data[i * self.cols + j];
            }
        }

        difference
    }
}

// Matrix multiplication
// untested
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
        let src = fs::read_to_string("./kernel.cl").unwrap();
        let pro_que = ProQue::builder()
            .src(src)
            .dims([self.rows, rhs.cols])
            .build()
            .unwrap();

        let buffer_a = Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_ONLY)
            .len(self.data.len())
            .copy_host_slice(&self.data)
            .build()
            .unwrap();

        let buffer_b = Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_ONLY)
            .len(rhs.data.len())
            .copy_host_slice(&rhs.data)
            .build()
            .unwrap();

        let buffer_c = Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(product.data.len())
            .build()
            .unwrap();

        let kernel = pro_que
            .kernel_builder("mat_mul")
            .arg(&buffer_a)
            .arg(&buffer_b)
            .arg(&buffer_c)
            .arg(self.rows as i32)
            .arg(self.cols as i32)
            .arg(rhs.cols as i32)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        buffer_c.read(&mut product.data).enq().unwrap();

        product
    }
}

// Scalar multiplication
impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut product: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                product.data[i * self.cols + j] = rhs * self.data[i * self.cols + j];
            }
        }

        product
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
