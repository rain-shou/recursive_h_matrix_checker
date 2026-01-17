#include <vector>
#include <cmath>
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
            double rr = colRe[i], ii = colIm[i];
            double mag = std::sqrt(rr * rr + ii * ii);
            colM[i] = (i == j) ? mag : -mag;
        }
    }
}

static bool is_M_matrix_colmajor(double* M, int n, int lda) {
    for (int k = 0; k < n; ++k) {
        double pivot = M[k + (size_t)k * lda];
        if (!(pivot > 0.0)) {
            return false;
        }

        int m = n - (k + 1);
        if (m <= 0) {
            break;
        }

        double* x = M + (size_t)(k + 1) + (size_t)k * lda;

        double* y = M + k + (size_t)(k + 1) * lda;

        double* A22 = M + (size_t)(k + 1) + (size_t)(k + 1) * lda;

        double alpha = -1.0 / pivot;
        cblas_dger(CblasColMajor, m, m, alpha, x, 1, y, lda, A22, lda);
    }
    return true;
}

extern "C" bool is_H_matrix(double* real, double* imag, int n) {
    const int lda = n;
    std::vector<double> M((size_t)lda * n);

    build_comparison_matrix_colmajor(real, imag, n, M.data(), lda);

    return is_M_matrix_colmajor(M.data(), n, lda);
}
