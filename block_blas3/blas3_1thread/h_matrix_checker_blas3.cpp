#include <vector>
#include <cmath>
#include <algorithm>
#include <cblas.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

extern "C" bool is_H_matrix(double* real, double* imag, int n);

static inline void build_comparison_matrix_colmajor(const double* re, const double* im, int n, double* M, int lda) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
        const double* colRe = re + (size_t)j * n;
        const double* colIm = im + (size_t)j * n;
        double*       colM  = M  + (size_t)j * lda;
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            double rr  = colRe[i];
            double ii  = colIm[i];
            double mag = std::sqrt(rr * rr + ii * ii);
            colM[i] = (i == j) ? mag : -mag;
        }
    }
}

static bool is_M_matrix_colmajor(double* A, int n, int lda) {
    const int nb = 64;

    for (int k = 0; k < n; k += nb) {
        const int kb = std::min(nb, n - k);

        for (int j = k; j < k + kb && j < n; ++j) {
            double pivot = A[j + (size_t)j * lda];
            if (!(pivot > 0.0)) {
                return false;
            }

            const int m = n - (j + 1);
            if (m <= 0) {
                continue;
            }
            
            double inv_pivot = 1.0 / pivot;
            double* col_j_below = A + (size_t)(j + 1) + (size_t)j * lda;
            cblas_dscal(m, inv_pivot, col_j_below, 1);


            const int n_panel = k + kb - (j + 1);
            if (n_panel > 0) {
                double* row_j_panel = A + (size_t)j + (size_t)(j + 1) * lda;
                double* A_sub       = A + (size_t)(j + 1) + (size_t)(j + 1) * lda;

                cblas_dger(CblasColMajor, m, n_panel, -1.0, col_j_below, 1, row_j_panel, lda, A_sub, lda);
            }
        }

        const int m2 = n - (k + kb);
        const int n2 = n - (k + kb);

        if (n2 > 0) {
            double* L11 = A + (size_t)k + (size_t)k * lda;
            double* A12 = A + (size_t)k + (size_t)(k + kb) * lda;

            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, kb, n2, 1.0, L11, lda, A12, lda);
        }

        if (m2 > 0 && n2 > 0) {
            double* L21 = A + (size_t)(k + kb) + (size_t)k * lda;
            double* U12 = A + (size_t)k + (size_t)(k + kb) * lda;
            double* A22 = A + (size_t)(k + kb) + (size_t)(k + kb) * lda;

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m2, n2, kb, -1.0, L21, lda, U12, lda, 1.0, A22, lda);
        }
    }

    return true;
}

extern "C" bool is_H_matrix(double* real, double* imag, int n) {
    const int lda = n;
    std::vector<double> M((size_t)lda * n);

    build_comparison_matrix_colmajor(real, imag, n, M.data(), lda);

    return is_M_matrix_colmajor(M.data(), n, lda);
}
