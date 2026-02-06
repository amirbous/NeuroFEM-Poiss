
from matplotlib.pylab import gamma
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import scipy.sparse.linalg as spla

import subprocess
import sys
import re
import time


def convergence_diagnostic(rho, noise_estimate=1e-6):
    amplification = 1.0 / (1.0 - rho)
    steady_error = amplification * noise_estimate

    print("\n--- Practical convergence diagnostic ---")
    print(f"rho = {rho:.6f}")
    print(f"noise amplification ≈ {amplification:.2f}x")
    print(f"estimated steady error ≈ {steady_error:.2e}")

    if amplification > 100:
        print("⚠ Slow contraction — hardware noise sensitive")
    else:
        print("✓ Robust contraction")



def neurofem_quantization_check(A, A_quant, scale, dt=None):
    print("\n--- NeuroFEM stability verification ---")

    # reconstruct quantized float matrix
    A_hw = A_quant.astype(float) * scale

    # smallest eigenvalue of original A
    lam_min = spla.eigsh(A, k=1, which='SA',
                         return_eigenvectors=False)[0]

    print(f"λ_min(A) = {lam_min:.6e}")

    if lam_min <= 0:
        print("❌ Continuous system unstable (A not positive definite)")
        return

    # quantization perturbation norm
    delta = spla.norm(A - A_hw)
    print(f"||ΔA|| = {delta:.6e}")

    if delta >= lam_min:
        print("⚠ Quantization may destroy convergence")
    else:
        print("✓ Quantization preserves spectral margin")

    # optional discrete stability test
    if dt is not None:
        G = sp.sparse.eye(A.shape[0]) - dt * A_hw
        rho = abs(spla.eigs(G, k=1,
                            which='LM',
                            return_eigenvectors=False)[0])

        print(f"ρ(I-dt A_hw) = {rho:.6e}")

        if rho >= 1:
            print("⚠ Discrete timestep instability risk")
        else:
            print("✓ Discrete dynamics stable")

    print("--- end check ---\n")
    convergence_diagnostic(rho, 1e-6)


##########################################
## Utils section
###########################################
def float_to_signed_sparse(matrix, x_bits=6, scale=None):
    """
    Quantizize a sparse matrix to signed integers with specific bit-width.
    """
    if not sp.sparse.issparse(matrix):
        matrix = sp.sparse.csr_matrix(matrix)

    x_bits -= 1 # one bit for sign

    matrix = matrix.astype(np.float32).tocoo() # use float32, keep sparse
    max_int = 2**x_bits - 1
    min_int = -2**x_bits

    # compute scale if not given
    if scale is None:
        max_val = np.max(np.abs(matrix.data))
        scale = max_val / max_int if max_val != 0 else 1.0

    # efficient quantitization in sparse form
    int_data = np.empty_like(matrix.data, dtype=np.int32)
    for i in range(len(matrix.data)):
        val = matrix.data[i] / scale
        int_data[i] = int(np.clip(np.round(val), min_int, max_int))

    int_matrix = sp.sparse.coo_matrix((int_data, (matrix.row, matrix.col)), shape=matrix.shape, dtype=np.int32).tocsr()
    return int_matrix, scale





# IO read matrix
def LoadCSRMatrix(model_name="", data_dir="../../data"):
    
    model_matrix_filename = f"{data_dir}/matrix/{model_name}_mtx.txt"
    model_rhs_filename = f"{data_dir}/matrix/{model_name}_rhs.txt"
    model_solution_filename = f"" # TODO: Have to first store them in C++

    A_csr = []
    b = []

    if not os.path.exists(model_matrix_filename):
        print(f"Error: {model_matrix_filename} not found.")
        sys.exit(1)

    # --- 1. Load Matrix Data ---
    try:
        print("Loading matrix data...")
        raw_data = np.loadtxt(model_matrix_filename, skiprows=1, dtype=float)
        
        with open(model_matrix_filename, 'r') as f:
            header = f.readline().split()
            n_rows = int(header[0])
            n_cols = int(header[1])

        rows = raw_data[:, 0].astype(int) - 1
        cols = raw_data[:, 1].astype(int) - 1
        values = raw_data[:, 2]

        A_csr = sp.sparse.csr_matrix((values, (rows, cols)), shape=(n_rows, n_cols))
        
        A_csr.eliminate_zeros()

    except Exception as e:
        print(f"Failed to load matrix: {e}")
        sys.exit(1)

    # ---2. Load RHS Data ---
    try:
        print("Loading RHS data...")
        b = np.loadtxt(model_rhs_filename, skiprows=1)
        
        if A_csr.shape[0] != b.shape[0]:
            print(f"Error: Matrix rows ({A_csr.shape[0]}) != RHS length ({b.shape[0]})")
            sys.exit(1)

    except Exception as e:
        print(f"Failed to load RHS: {e}")
        sys.exit(1)

    return A_csr, b



def main():

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Sphere_00"


    print("Starting CSR to SNN pipeline...")

    #1. Load CSR Matrix and RHS
    A_csr, b = LoadCSRMatrix(model_name, "../data")
    n_mesh = A_csr.shape[0]

    print("1. Loaded matrix and RHS.")


    A_quant, A_scale = float_to_signed_sparse(A_csr, x_bits=21)


    npm = 8                            # neurons per node
    r = npm // 2                       # num neurons with positive sign

    num_timesteps = 10000              # timesteps
    sys_tick_in_s = 1e-3              # (default is 1 ms)

    gamma = 1e-5                      # neuron gain
    
    
    theta = 0.5 * (gamma ** 2)        # neuron spike threshold
    lambda_max, _ = spla.eigsh(A_csr, k=1, which='LM')
    lambda_max = np.abs(lambda_max[0])
    
    dt = 1.0 / (4 * lambda_max)
    #dt = 2e-2
    print(f"Computed dt: {dt:.6e} s")

    tau = 100
    lambda_d = 1 / (tau * dt)
    lambda_v = 2 / (tau * dt)

    omega_n = 2.0
    zeta = 4.0
    k_p = omega_n**2
    k_i = 2 * omega_n * zeta

    sigma = 0.05
    steady_state = 0.4
    neuron_params = {
        "gb": [b[i] * gamma for i in range(n_mesh)], # Mesh param
        "threshold": theta, # Global param
        "scale": A_scale * (gamma ** 2), # Global param
        "dt": dt, # Global param
        "gamma": gamma, # Global param
        "lambda_d": lambda_d, # Global param
        "lambda_v": lambda_v, # Global param
        "k_p": k_p, # Global param
        "k_i": k_i, # Global param
        "sigma": sigma, # Global param
        "steady_state": steady_state, # Global param
        "npm": npm
    }
    # print the log and properties for a matrix
    neurofem_quantization_check(A_csr, A_quant, A_scale, dt)


if __name__ == "__main__":      
    main()


