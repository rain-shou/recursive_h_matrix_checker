#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>

using namespace std;

extern "C" bool is_H_matrix_recursive(double* real, double* imag, int n);

static vector<double> comparisonMatrix(const vector<complex<double>>& A, int n) {
    vector<double> M(n * n);
    for (int i = 0; i < n; ++i) {
        const int row = i * n;
        for (int j = 0; j < n; ++j) {
            double v = abs(A[row + j]);
            M[row + j] = (i == j) ? v : -v;
        }
    }
    return M;
}

static bool isMMatrix_via_schur_iter(const vector<double>& Min, int n) {
    if (n == 1) {
        return Min[0] > 0;
    }

    vector<double> bufA, bufB;
    bufA.resize(n * n);
    bufB.resize(n * n);

    const double* rd = Min.data();
    vector<double>* wr_vec = &bufA;

    int curN = n;

    while (curN > 1) {
        const int m = curN - 1;
        double a11 = rd[0];

        if (a11 <= 0) {
            return false;
        }

        double* wr = wr_vec->data();

        for (int i = 0; i < m; ++i) {
            const int iA = (i + 1) * curN;
            const double c_i = rd[iA + 0];
            for (int j = 0; j < m; ++j) {
                const double a_ij = rd[iA + (j + 1)];
                const double b_j  = rd[0 * curN + (j + 1)];
                wr[i * m + j] = a_ij - (c_i * b_j) / a11;
            }
        }

        rd = wr;
        curN = m;
        wr_vec = (wr_vec == &bufA) ? &bufB : &bufA;
    }

    return rd[0] > 0;
}

extern "C" bool is_H_matrix_recursive(double* real, double* imag, int n) {
    vector<complex<double>> A(n * n);
    for (int i = 0; i < n * n; ++i) {
        double im = imag ? imag[i] : 0.0;
        A[i] = complex<double>(real[i], im);
    }
    vector<double> M = comparisonMatrix(A, n);
    return isMMatrix_via_schur_iter(M, n);
}
