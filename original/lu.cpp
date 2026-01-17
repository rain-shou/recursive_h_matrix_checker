#include <vector>
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>

extern "C" bool is_H_matrix_lu(double* real, double* imag, int n);

using namespace std;

#ifndef HSWP_BLOCK_COLS
#define HSWP_BLOCK_COLS 64
#endif

static inline int ID(int i, int j, int ld) { return i * ld + j; }

static double norm_inf(const vector<double>& A, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int row = i * n;
        for (int j = 0; j < n; ++j) {
            s += fabs(A[row + j]);
        }
        if (s > m) {
            m = s;
        }
    }
    return m;
}

static bool lu_factor(vector<double>& A, int n, vector<int>& piv, double& Anorm, double& pivot_tol) {
    piv.resize(n);
    for (int i = 0; i < n; ++i) {
        piv[i] = i;
    }

    const double eps = numeric_limits<double>::epsilon();
    Anorm = max(1.0, norm_inf(A, n));
    pivot_tol = (double)n * eps * Anorm;

    for (int k = 0; k < n; ++k) {
        int piv_row = k;
        double best = fabs(A[ID(k,k,n)]);
        for (int i = k + 1; i < n; ++i) {
            double v = fabs(A[ID(i,k,n)]);
            if (v > best) {
                best = v;
                piv_row = i;
            }
        }
        if (best <= pivot_tol || !isfinite(best)) {
            return false;
        }

        if (piv_row != k) {
            for (int j = 0; j < n; ++j) {
                swap(A[ID(k,j,n)], A[ID(piv_row,j,n)]);
            }
            swap(piv[k], piv[piv_row]);
        }

        const double Akk = A[ID(k,k,n)];
        for (int i = k + 1; i < n; ++i) {
            A[ID(i,k,n)] /= Akk;
            const double lik = A[ID(i,k,n)];
            const int irow = i * n;
            const int krow = k * n;
            for (int j = k + 1; j < n; ++j)
                A[irow + j] -= lik * A[krow + j];
        }
    }
    return true;
}

static void lu_solve_multi(const vector<double>& LU, int n, const vector<int>& piv, vector<double>& B, int nb) {
    for (int k = 0; k < n; ++k) {
        int pk = piv[k];
        if (pk != k) {
            double* rowk = &B[ID(k,0,nb)];
            double* rowp = &B[ID(pk,0,nb)];
            for (int j = 0; j < nb; ++j) {
                std::swap(rowk[j], rowp[j]);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        const int irow = i * n;
        for (int k = 0; k < i; ++k) {
            const double lik = LU[ID(i,k,n)];
            if (lik == 0.0) {
                continue;
            }
            const double* rowk = &B[ID(k,0,nb)];
            double* rowi = &B[ID(i,0,nb)];
            for (int j = 0; j < nb; ++j) {
                rowi[j] -= lik * rowk[j];
            }
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        const int irow = i * n;
        const double Uii = LU[ID(i,i,n)];
        for (int k = i + 1; k < n; ++k) {
            const double uik = LU[ID(i,k,n)];
            if (uik == 0.0) {
                continue;
            }
            const double* rowk = &B[ID(k,0,nb)];
            double* rowi = &B[ID(i,0,nb)];
            for (int j = 0; j < nb; ++j) {
                rowi[j] -= uik * rowk[j];
            }
        }
        double* rowi = &B[ID(i,0,nb)];
        for (int j = 0; j < nb; ++j) {
            rowi[j] /= Uii;
        }
    }
}

static void build_comparison_matrix_from_complex(const double* real, const double* imag, int n, vector<double>& M) {
    M.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        const int row = i * n;
        for (int j = 0; j < n; ++j) {
            const int k = row + j;
            const double mag = std::hypot(real[k], imag ? imag[k] : 0.0);
            M[k] = (i == j) ? mag : -mag;
        }
    }
}

extern "C" bool is_H_matrix_lu(double* real, double* imag, int n) {
    if (!real || n <= 0) {
        return false;
    }

    vector<double> M;
    build_comparison_matrix_from_complex(real, imag, n, M);

    vector<int> piv;
    double Anorm = 1.0, pivot_tol = 0.0;
    if (!lu_factor(M, n, piv, Anorm, pivot_tol)) {
        return false;
    }

    const double eps = numeric_limits<double>::epsilon();
    const double col_base = (double)n * 50.0 * eps;

    std::vector<double> B;
    B.resize((size_t)n * HSWP_BLOCK_COLS, 0.0);

    for (int col0 = 0; col0 < n; col0 += HSWP_BLOCK_COLS) {
        const int nb = std::min(HSWP_BLOCK_COLS, n - col0);
        fill(B.begin(), B.begin() + (size_t)n * nb, 0.0);
        for (int j = 0; j < nb; ++j) {
            B[ID(col0 + j, j, nb)] = 1.0;
        }

        lu_solve_multi(M, n, piv, B, nb);

        for (int j = 0; j < nb; ++j) {
            double max_abs = 0.0;
            for (int i = 0; i < n; ++i) {
                double v = B[ID(i, j, nb)];
                if (!std::isfinite(v)) {
                    return false;
                }
                double av = std::fabs(v);
                if (av > max_abs) {
                    max_abs = av;
                }
            }
            const double col_tol = col_base * std::max(1.0, max_abs);
            for (int i = 0; i < n; ++i) {
                if (B[ID(i, j, nb)] < -col_tol) {
                    return false;
                }
            }
        }
    }

    return true;
}
