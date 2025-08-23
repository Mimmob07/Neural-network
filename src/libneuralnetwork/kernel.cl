__kernel void mat_mul(
    __global const float* A,    // Matrix A
    __global const float* B,    // Matrix B
    __global float* C,          // Result
    const int A_ROWS,           // Rows in A
    const int A_COLS,           // Cols in A / Rows in B
    const int B_COLS)           // Cols in B
{
    int row = get_global_id(0); // Which row of C this work-item computes
    int col = get_global_id(1); // Which col of C this work-item computes

    if (row < A_ROWS && col < B_COLS) {
        float sum = 0.0f;
        for (int k = 0; k < A_COLS; k++) {
            // A[row, k] = A[row * A_COLS + k]
            // B[k, col] = B[k * B_COLS + col]
            sum += A[row * A_COLS + k] * B[k * B_COLS + col];
        }
        // C[row, col] = sum
        C[row * B_COLS + col] = sum;
    }
}
