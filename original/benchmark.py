import os

os.environ["OMP_NUM_THREADS"] = "1"         # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"         # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Apple Accelerate / vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # numexpr

import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing

lib_recursive = ctypes.cdll.LoadLibrary("./libhcheck_recursive.so")
lib_gauss = ctypes.cdll.LoadLibrary("./libhcheck_gauss.so")
lib_lu = ctypes.cdll.LoadLibrary("./libhcheck_lu.so")

is_h_matrix_recursive = lib_recursive.is_H_matrix_recursive
is_h_matrix_recursive.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
is_h_matrix_recursive.restype = ctypes.c_bool

is_h_matrix_gauss = lib_gauss.is_H_matrix_gauss
is_h_matrix_gauss.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
is_h_matrix_gauss.restype = ctypes.c_bool

is_h_matrix_lu = lib_lu.is_H_matrix_lu
is_h_matrix_lu.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
is_h_matrix_lu.restype = ctypes.c_bool

def is_H_matrix_recursive(A):
    n = A.shape[0]
    real = np.ascontiguousarray(A.real.ravel(), dtype=np.float64)
    imag = np.ascontiguousarray(A.imag.ravel(), dtype=np.float64)
    return is_h_matrix_recursive(real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           n)

def is_H_matrix_gauss(A):
    n = A.shape[0]
    real = np.ascontiguousarray(A.real.ravel(), dtype=np.float64)
    imag = np.ascontiguousarray(A.imag.ravel(), dtype=np.float64)
    return is_h_matrix_gauss(real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           n)

def is_H_matrix_lu(A):
    n = A.shape[0]
    real = np.ascontiguousarray(A.real.ravel(), dtype=np.float64)
    imag = np.ascontiguousarray(A.imag.ravel(), dtype=np.float64)
    return is_h_matrix_lu(real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
            base_phase = np.pi
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
    result_recursive = is_H_matrix_recursive(A)
    t_recursive = time.perf_counter() - start

    start = time.perf_counter()
    result_gauss = is_H_matrix_gauss(A)
    t_gauss = time.perf_counter() - start

    start = time.perf_counter()
    result_lu = is_H_matrix_lu(A)
    t_lu = time.perf_counter() - start

    start = time.perf_counter()
    result_numpy = is_H_matrix_numpy(A)
    t_numpy = time.perf_counter() - start

    print(f"[n={n}] Recursive: {result_recursive}, Gauss: {result_gauss}, LU: {result_lu}, Numpy: {result_numpy}")
    assert result_recursive == result_gauss == result_lu == result_numpy, f"Mismatch at n={n}"

    return t_recursive, t_gauss, t_lu, t_numpy

if __name__ == '__main__':
    # from threadpoolctl import threadpool_limits, threadpool_info
    # print("Threadpools:", threadpool_info()) 

    sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    kinds = ["non_H", "nondom_banded", "arrowhead", "dense_random", "easy_tridiag", "hard_H", "critical_H", ]

    # print(f"Detected {multiprocessing.cpu_count()} logical CPU cores.\n")
    # print("NumPy will be forced to use only 1 thread.\n")
    
    for kind in kinds:
        recursive_times = []
        gauss_times = []
        lu_times = []
        numpy_times = []
        print(f"\n=== Kind: {kind} ===")
        for n in sizes:
            print(f"Benchmarking n={n}...")
            t_recursive, t_gauss, t_lu, t_numpy = benchmark(n, kind)
            recursive_times.append(t_recursive)
            gauss_times.append(t_gauss)
            lu_times.append(t_lu)
            numpy_times.append(t_numpy)

        plt.figure()
        plt.plot(sizes, recursive_times, label='Recursive', marker='o')
        plt.plot(sizes, gauss_times, label='Gauss', marker='s')
        plt.plot(sizes, lu_times, label='LU', marker='x')
        plt.plot(sizes, numpy_times, label='Numpy', marker='p')
        plt.xlabel('Matrix size (n x n)')
        plt.ylabel('Execution time (seconds)')
        plt.title('H-Matrix Verification Performance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\nExecution Time Comparison:")
        print("{:>12}  {:>20}  {:>20}  {:>20}  {:>20}  {:>20}  {:>20}  {:>20}".format("Matrix Size", "Recursive Time (s)", "Gauss Time (s)", "LU Time (S)", "Numpy Time (s)", "Speedup (Gauss/Recursive)", "Speedup (LU/Recursive)", "Speedup (Numpy/Recursive)"))
        for i, n in enumerate(sizes):
            speedup_gr = round(gauss_times[i] / recursive_times[i], 2) if recursive_times[i] > 0 else float('inf')
            speedup_lr = round(lu_times[i] / recursive_times[i], 2) if recursive_times[i] > 0 else float('inf')
            speedup_nr = round(numpy_times[i] / recursive_times[i], 2) if recursive_times[i] > 0 else float('inf')
            print("{:>12}  {:>18.6f}  {:>18.6f}  {:>18.6f}  {:>18.6f}  {:>18.2f}  {:>18.2f}  {:>18.2f}".format(
                n, recursive_times[i], gauss_times[i], lu_times[i], numpy_times[i], speedup_gr, speedup_lr, speedup_nr))
