import sys
import os
import numpy as np
import scipy as sp


import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve


def main():
    problem_name = sys.argv[1] if len(sys.argv) > 1 else "Sphere_00"
    
    mtx_filename = f"{problem_name}_mtx.txt"
    rhs_filename = f"{problem_name}_rhs.txt"
    sol_filename = f"{problem_name}_sol.txt"

    if not os.path.exists(mtx_filename):
        print(f"Error: {mtx_filename} not found.")
        sys.exit(1)

    # --- 1. Load Data on CPU (Host) ---
    try:
        print("Loading matrix data...")
        raw_data = np.loadtxt(mtx_filename, skiprows=1, dtype=float)
        
        with open(mtx_filename, 'r') as f:
            header = f.readline().split()
            n_rows = int(header[0])
            n_cols = int(header[1])

        rows = raw_data[:, 0].astype(int) - 1
        cols = raw_data[:, 1].astype(int) - 1
        values = raw_data[:, 2]

        A_host = sp.sparse.csr_matrix((values, (rows, cols)), shape=(n_rows, n_cols))
        
        A_host.eliminate_zeros()

    except Exception as e:
        print(f"Failed to load matrix: {e}")
        sys.exit(1)

    try:
        b_host = np.loadtxt(rhs_filename, skiprows=1)
        
        if A_host.shape[0] != b_host.shape[0]:
            print(f"Error: Matrix rows ({A_host.shape[0]}) != RHS length ({b_host.shape[0]})")
            sys.exit(1)

    except Exception as e:
        print(f"Failed to load RHS: {e}")
        sys.exit(1)


    try:
        print("Transferring data to GPU...")

        A_device = csp.csr_matrix(A_host) 
        
        b_device = cp.asarray(b_host)

        print("Solving on GPU...")

        x_device = spsolve(A_device, b_device)
        
        if cp.any(cp.isnan(x_device)) or cp.any(cp.isinf(x_device)):
            print("Did not converge")
            sys.exit(1)
            

        x_host = x_device.get()

    except Exception as e:
        print(f"GPU Solver failed: {e}")

        sys.exit(1)

    try:
        with open(sol_filename, 'w') as f:
            f.write(f"{len(x_host)}\n")
            for val in x_host:
                f.write(f"{val:.16e}\n")
        print(f"Solution saved to {sol_filename}")

    except Exception as e:
        print(f"Failed to write solution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()