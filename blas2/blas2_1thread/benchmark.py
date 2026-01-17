import os

os.environ["OMP_NUM_THREADS"] = "1"         # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"         # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Apple Accelerate / vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # numexprxw

import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing

lib = ctypes.cdll.LoadLibrary("./libhcheck.so")

is_h_matrix_cpp = lib.is_H_matrix
is_h_matrix_cpp.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
is_h_matrix_cpp.restype = ctypes.c_bool

def is_H_matrix_cpp(A):
    n = A.shape[0]
    real = np.ascontiguousarray(A.real.ravel(), dtype=np.float64)
    imag = np.ascontiguousarray(A.imag.ravel(), dtype=np.float64)
    return is_h_matrix_cpp(real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           n)

def is_H_matrix_numpy(A):
    M = -np.abs(A)
    np.fill_diagonal(M, np.abs(np.diag(A)))
    M = M.real
    try:
        M_inv = np.linalg.inv(M)
        return np.all(M_inv >= 0)
    except np.linalg.LinAlgError:
        return False

def _scale_to_spectral_radius(M, target_rho):
    n = M.shape[0]
    v = np.random.rand(n)
    v /= np.linalg.norm(v) + 1e-18
    for _ in range(50):
        v_next = M @ v
        nv = np.linalg.norm(v_next)
        if nv < 1e-30:
            return np.zeros_like(M)
        v = v_next / nv
    rho_est = float(v @ (M @ v))
    if rho_est <= 0:
        return M * 0.0
    return M * (target_rho / rho_est)

def _build_A_from_comparison(M, noise=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = M.shape[0]
    A = np.zeros((n, n), dtype=np.complex128)

    theta_diag = noise * rng.standard_normal(n)
    for i in range(n):
        mag = max(M[i, i], 0.0)
        A[i, i] = mag * np.exp(1j * theta_diag[i])

    theta_off = noise * rng.standard_normal((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mag = max(-M[i, j], 0.0)
            base_phase = np.pi  # -1
            A[i, j] = mag * np.exp(1j * (base_phase + theta_off[i, j]))
    return A

def generate_test_matrix(
    n,
    kind="hard_H",
    noise=1e-3,
    seed=None,
):
    rng = np.random.default_rng(seed)

    if kind == "easy_tridiag":
        A = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            A[i, i] = 4 + 1j * noise * rng.standard_normal()
            if i + 1 < n:
                A[i, i+1] = -1 + 1j * noise * rng.standard_normal()
                A[i+1, i] = -1 + 1j * noise * rng.standard_normal()
        return A

    elif kind in ("hard_H", "critical_H", "non_H"):
        B = rng.random((n, n))
        mask = (rng.random((n, n)) < 0.2).astype(np.float64)
        B *= (0.3 + 0.7 * mask)

        ratio = {"hard_H": 0.98, "critical_H": 0.999, "non_H": 1.02}[kind]
        alpha = 1.0
        B = _scale_to_spectral_radius(B, ratio * alpha)

        M = alpha * np.eye(n) - B

        A = _build_A_from_comparison(M, noise=noise, rng=rng)
        return A

    elif kind == "nondom_banded":
        w = max(3, n // 20)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if i == j:
                    continue
                M[i, j] = -rng.uniform(0.6, 1.2)
            row_sum = -M[i, :].sum()
            M[i, i] = row_sum * rng.uniform(0.98, 1.02)
        return _build_A_from_comparison(M, noise=noise, rng=rng)

    elif kind == "arrowhead":
        M = np.zeros((n, n))
        for j in range(1, n):
            M[0, j] = -rng.uniform(0.5, 1.5)
            M[j, 0] = -rng.uniform(0.5, 1.5)
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                if rng.random() < 0.05:
                    M[i, j] = -rng.uniform(0.1, 0.5)
        for i in range(n):
            row_sum = -M[i, :].sum()
            M[i, i] = row_sum * rng.uniform(0.98, 1.02)
        return _build_A_from_comparison(M, noise=noise, rng=rng)

    elif kind == "dense_random":
        M = -np.abs(rng.standard_normal((n, n)))
        np.fill_diagonal(M, 0.0)
        for i in range(n):
            row_sum = -M[i, :].sum()
            M[i, i] = row_sum * rng.uniform(0.95, 1.05)
        return _build_A_from_comparison(M, noise=noise, rng=rng)

    else:
        raise ValueError(f"Unknown kind={kind}")

def benchmark(n, kind, noise=1e-3, seed=42):
    A = generate_test_matrix(n, kind=kind, noise=noise, seed=seed)

    start = time.perf_counter()
    result_numpy = is_H_matrix_numpy(A)
    t_numpy = time.perf_counter() - start

    start = time.perf_counter()
    result_cpp = is_H_matrix_cpp(A)
    t_cpp = time.perf_counter() - start

    print(f"[n={n}] NumPy: {result_numpy}, C++: {result_cpp}")
    assert result_cpp == result_numpy, f"Mismatch at n={n}"

    return t_numpy, t_cpp

if __name__ == '__main__':
    # from threadpoolctl import threadpool_limits, threadpool_info
    # print("Threadpools:", threadpool_info()) 

    sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    kinds = ["non_H", "nondom_banded", "arrowhead", "dense_random", "easy_tridiag", "hard_H", "critical_H", ]

    # print(f"Detected {multiprocessing.cpu_count()} logical CPU cores.\n")
    # print("NumPy will be forced to use only 1 thread.\n")
    
    for kind in kinds:
        numpy_times, cpp_times = [], []
        print(f"\n=== Kind: {kind} ===")
        for n in sizes:
            print(f"Benchmarking n={n}...")
            t_numpy, t_cpp = benchmark(n, kind)
            numpy_times.append(t_numpy)
            cpp_times.append(t_cpp)

        plt.figure()
        plt.plot(sizes, numpy_times, label='NumPy (1-thread)', marker='o')
        plt.plot(sizes, cpp_times, label='C++ (iterative-refactored)', marker='s')
        plt.xlabel('Matrix size (n x n)')
        plt.ylabel('Execution time (seconds)')
        plt.title('H-Matrix Verification Performance (Single-core NumPy)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\nExecution Time Comparison:")
        print("{:>12}  {:>16}  {:>16}  {:>18}".format(
            "Matrix Size", "NumPy Time (s)", "C++ Time (s)", "Speedup (NumPy/C++)"))
        for i, n in enumerate(sizes):
            speedup = round(numpy_times[i] / cpp_times[i], 2) if cpp_times[i] > 0 else float('inf')
            print("{:>12}  {:>16.6f}  {:>16.6f}  {:>18.2f}".format(
                n, numpy_times[i], cpp_times[i], speedup))
