/**
 * @file sparse_mult.cpp
 * @brief This file contains the implementation of a function that performs multiplication between a sparse matrix and a dense matrix of rank r.
 *
 * The function is designed to be compiled into a MEX file for use with MATLAB. It utilizes OpenMP for parallel computation to improve performance for large datasets.
 *
 * To compile this file into a MEX file, use the following commands in the MATLAB terminal:
 * 
 * ```matlab
 * mex -setup C++
 * mex sparse_mult.cpp
 * ```
 *
 * The main function `mexFunction` handles the input and output arguments from MATLAB, and performs the matrix multiplication based on the value of `r`.
 * For values of `r` from 1 to 10, specialized loops are used for optimization. For other values of `r`, a general case function is called.
 *
 * The function uses atomic operations to ensure thread safety when updating the result matrix in parallel.
 *
 * @param nlhs Number of left-hand side (output) arguments
 * @param plhs Array of pointers to the left-hand side (output) arguments
 * @param nrhs Number of right-hand side (input) arguments
 * @param prhs Array of pointers to the right-hand side (input) arguments
 */
#include "mex.h"
#include <omp.h>

// Function for handling general cases of `r`
inline void performGeneralCase(const double* Y_vec, const double* omega_row, const double* omega_col, 
                               const double* R, double* res, int n1, int n2, int r, int pn) {
    #pragma omp parallel for if(pn > 1000)  // Perform parallel computation only when `pn` is large
    for (int i = 0; i < pn; i++) {
        int row = static_cast<int>(omega_row[i]) - 1;
        int col = static_cast<int>(omega_col[i]) - 1;
        for (int j = 0; j < r; j++) {
            int res_index = row + j * n1; // Precompute result index
            int R_index = col + j * n2;   // Precompute R matrix index
            #pragma omp atomic            // Atomic to ensure thread safety
            res[res_index] += Y_vec[i] * R[R_index];
        }
    }
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Check the number of input arguments
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("MATLAB:mm_mex:nrhs", "Seven inputs required.");
    }

    // Check the number of output arguments
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mm_mex:nlhs", "One output required.");
    }
    
    // Get input arguments
    const double *Y_vec = mxGetPr(prhs[0]);
    const double *omega_row = mxGetPr(prhs[1]);
    const double *omega_col = mxGetPr(prhs[2]);
    const double *R = mxGetPr(prhs[3]);
    int n1 = static_cast<int>(mxGetScalar(prhs[4]));
    int n2 = static_cast<int>(mxGetScalar(prhs[5]));
    int r = static_cast<int>(mxGetScalar(prhs[6]));

    // Get the length of Y_vec
    int pn = static_cast<int>(mxGetNumberOfElements(prhs[0]));

    // Create output matrix
    plhs[0] = mxCreateDoubleMatrix(n1, r, mxREAL);
    double *res = mxGetPr(plhs[0]);

    // Choose the ideal loop strategy based on the value of r
    switch (r) {
        case 1: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
                int row = static_cast<int>(omega_row[i]) - 1;
                int col = static_cast<int>(omega_col[i]) - 1;
                #pragma omp atomic
                res[row] += Y_vec[i] * R[col];
            }
            break;
        }
        case 2: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
                int row = static_cast<int>(omega_row[i]) - 1;
                int col = static_cast<int>(omega_col[i]) - 1;
                #pragma omp atomic
                res[row] += Y_vec[i] * R[col];
                #pragma omp atomic
                res[row + n1] += Y_vec[i] * R[col + n2];
            }
            break;
        }
        case 3: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
                int row = static_cast<int>(omega_row[i]) - 1;
                int col = static_cast<int>(omega_col[i]) - 1;
                #pragma omp atomic
                res[row] += Y_vec[i] * R[col];
                #pragma omp atomic
                res[row + n1] += Y_vec[i] * R[col + n2];
                #pragma omp atomic
                res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
            }
            break;
        }
        case 4: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
                int row = static_cast<int>(omega_row[i]) - 1;
                int col = static_cast<int>(omega_col[i]) - 1;
                #pragma omp atomic
                res[row] += Y_vec[i] * R[col];
                #pragma omp atomic
                res[row + n1] += Y_vec[i] * R[col + n2];
                #pragma omp atomic
                res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
                #pragma omp atomic
                res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
            }
            break;
        }
        case 5: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
                int row = static_cast<int>(omega_row[i]) - 1;
                int col = static_cast<int>(omega_col[i]) - 1;
                #pragma omp atomic
                res[row] += Y_vec[i] * R[col];
                #pragma omp atomic
                res[row + n1] += Y_vec[i] * R[col + n2];
                #pragma omp atomic
                res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
                #pragma omp atomic
                res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
                #pragma omp atomic
                res[row + 4 * n1] += Y_vec[i] * R[col + 4 * n2];
            }
            break;
        }
        case 6: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
            int row = static_cast<int>(omega_row[i]) - 1;
            int col = static_cast<int>(omega_col[i]) - 1;
            #pragma omp atomic
            res[row] += Y_vec[i] * R[col];
            #pragma omp atomic
            res[row + n1] += Y_vec[i] * R[col + n2];
            #pragma omp atomic
            res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
            #pragma omp atomic
            res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
            #pragma omp atomic
            res[row + 4 * n1] += Y_vec[i] * R[col + 4 * n2];
            #pragma omp atomic
            res[row + 5 * n1] += Y_vec[i] * R[col + 5 * n2];
            }
            break;
        }
        case 7: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
            int row = static_cast<int>(omega_row[i]) - 1;
            int col = static_cast<int>(omega_col[i]) - 1;
            #pragma omp atomic
            res[row] += Y_vec[i] * R[col];
            #pragma omp atomic
            res[row + n1] += Y_vec[i] * R[col + n2];
            #pragma omp atomic
            res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
            #pragma omp atomic
            res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
            #pragma omp atomic
            res[row + 4 * n1] += Y_vec[i] * R[col + 4 * n2];
            #pragma omp atomic
            res[row + 5 * n1] += Y_vec[i] * R[col + 5 * n2];
            #pragma omp atomic
            res[row + 6 * n1] += Y_vec[i] * R[col + 6 * n2];
            }
            break;
        }
        case 8: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
            int row = static_cast<int>(omega_row[i]) - 1;
            int col = static_cast<int>(omega_col[i]) - 1;
            #pragma omp atomic
            res[row] += Y_vec[i] * R[col];
            #pragma omp atomic
            res[row + n1] += Y_vec[i] * R[col + n2];
            #pragma omp atomic
            res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
            #pragma omp atomic
            res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
            #pragma omp atomic
            res[row + 4 * n1] += Y_vec[i] * R[col + 4 * n2];
            #pragma omp atomic
            res[row + 5 * n1] += Y_vec[i] * R[col + 5 * n2];
            #pragma omp atomic
            res[row + 6 * n1] += Y_vec[i] * R[col + 6 * n2];
            #pragma omp atomic
            res[row + 7 * n1] += Y_vec[i] * R[col + 7 * n2];
            }
            break;
        }
        case 9: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
            int row = static_cast<int>(omega_row[i]) - 1;
            int col = static_cast<int>(omega_col[i]) - 1;
            #pragma omp atomic
            res[row] += Y_vec[i] * R[col];
            #pragma omp atomic
            res[row + n1] += Y_vec[i] * R[col + n2];
            #pragma omp atomic
            res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
            #pragma omp atomic
            res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
            #pragma omp atomic
            res[row + 4 * n1] += Y_vec[i] * R[col + 4 * n2];
            #pragma omp atomic
            res[row + 5 * n1] += Y_vec[i] * R[col + 5 * n2];
            #pragma omp atomic
            res[row + 6 * n1] += Y_vec[i] * R[col + 6 * n2];
            #pragma omp atomic
            res[row + 7 * n1] += Y_vec[i] * R[col + 7 * n2];
            #pragma omp atomic
            res[row + 8 * n1] += Y_vec[i] * R[col + 8 * n2];
            }
            break;
        }
        case 10: {
            #pragma omp parallel for if(pn > 1000)
            for (int i = 0; i < pn; i++) {
            int row = static_cast<int>(omega_row[i]) - 1;
            int col = static_cast<int>(omega_col[i]) - 1;
            #pragma omp atomic
            res[row] += Y_vec[i] * R[col];
            #pragma omp atomic
            res[row + n1] += Y_vec[i] * R[col + n2];
            #pragma omp atomic
            res[row + 2 * n1] += Y_vec[i] * R[col + 2 * n2];
            #pragma omp atomic
            res[row + 3 * n1] += Y_vec[i] * R[col + 3 * n2];
            #pragma omp atomic
            res[row + 4 * n1] += Y_vec[i] * R[col + 4 * n2];
            #pragma omp atomic
            res[row + 5 * n1] += Y_vec[i] * R[col + 5 * n2];
            #pragma omp atomic
            res[row + 6 * n1] += Y_vec[i] * R[col + 6 * n2];
            #pragma omp atomic
            res[row + 7 * n1] += Y_vec[i] * R[col + 7 * n2];
            #pragma omp atomic
            res[row + 8 * n1] += Y_vec[i] * R[col + 8 * n2];
            #pragma omp atomic
            res[row + 9 * n1] += Y_vec[i] * R[col + 9 * n2];
            }
            break;
        }
        default: {
            performGeneralCase(Y_vec, omega_row, omega_col, R, res, n1, n2, r, pn);
            break;
        }
    }
}
