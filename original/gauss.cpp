#include <vector>
#include <cmath>
#include <cstddef>

extern "C" bool is_H_matrix_gauss(double* real, double* imag, int n);

using namespace std;

static inline size_t idx(size_t i, size_t j, size_t ld) { return i * ld + j; }

static bool gauss_jordan_inverse(vector<double>& A, size_t n, vector<double>& Ainv, double eps = 1e-12) {
    Ainv.assign(n * n, 0.0);

    vector<double> aug(n * (2 * n), 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[idx(i, j, 2 * n)] = A[idx(i, j, n)];
        }
        aug[idx(i, n + i, 2 * n)] = 1.0;
    }

    for (size_t k = 0; k < n; ++k) {
        size_t piv = k;
        double best = fabs(aug[idx(k, k, 2 * n)]);
        for (size_t r = k + 1; r < n; ++r) {
            double v = fabs(aug[idx(r, k, 2 * n)]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < eps) {
            return false;
        }

        if (piv != k) {
            for (size_t j = 0; j < 2 * n; ++j) {
                swap(aug[idx(k, j, 2 * n)], aug[idx(piv, j, 2 * n)]);
            }
        }

        double pivot = aug[idx(k, k, 2 * n)];
        for (size_t j = 0; j < 2 * n; ++j) {
            aug[idx(k, j, 2 * n)] /= pivot;
        }

        for (size_t i = 0; i < n; ++i) {
            if (i == k) {
                continue;
            }
            double factor = aug[idx(i, k, 2 * n)];
            if (factor == 0.0) {
                continue;
            }
            for (size_t j = 0; j < 2 * n; ++j) {
                aug[idx(i, j, 2 * n)] -= factor * aug[idx(k, j, 2 * n)];
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Ainv[idx(i, j, n)] = aug[idx(i, n + j, 2 * n)];
        }
    }

    return true;
}

static void build_comparison_matrix_from_complex(const double* real, const double* imag, int n, vector<double>& M) {
    const size_t N = static_cast<size_t>(n);
    M.assign(N * N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const size_t k = idx(i, j, N);
            const double mag = hypot(real[k], imag ? imag[k] : 0.0);
            if (i == j) {
                M[k] = mag;
            } else {
                M[k] = -mag;
            }
        }
    }
}

extern "C" bool is_H_matrix_gauss(double* real, double* imag, int n) {
    if (real == nullptr || n <= 0) {
        return false;
    }

    std::vector<double> M, Minv;
    build_comparison_matrix_from_complex(real, imag, n, M);

    if (!gauss_jordan_inverse(M, static_cast<size_t>(n), Minv)) {
        return false;
    }

    const double tol = 1e-12;
    for (size_t i = 0; i < static_cast<size_t>(n) * static_cast<size_t>(n); ++i) {
        if (Minv[i] < -tol) {
            return false;
        }
    }
    return true;
}
