import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as linalg
import numpy as np
import time
from pathlib import Path
import pynvml
import sys

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

# Helper function to get current power in Watts
def get_power():
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W

# CUDA timing helper
def gpu_elapsed_ms(start_event, end_event):
    end_event.synchronize()
    return cp.cuda.get_elapsed_time(start_event, end_event)

def load_matrix_rhs(problem_name: str):
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "matrix"
    mtx_path = data_dir / f"{problem_name}_mtx.txt"
    rhs_path = data_dir / f"{problem_name}_rhs.txt"

    if not mtx_path.is_file():
        raise FileNotFoundError(f"Matrix file not found: {mtx_path}")
    if not rhs_path.is_file():
        raise FileNotFoundError(f"RHS file not found: {rhs_path}")

    with mtx_path.open("r") as f:
        header = f.readline().strip().split()
    n_rows, n_cols, _ = map(int, header)

    mtx_data = np.loadtxt(mtx_path, skiprows=1)
    rows = mtx_data[:, 0].astype(np.int64) - 1
    cols = mtx_data[:, 1].astype(np.int64) - 1
    vals = mtx_data[:, 2].astype(np.float32)

    rhs_data = np.loadtxt(rhs_path, skiprows=1).astype(np.float32)

    # ---- transfer timing ----
    t0 = time.perf_counter()

    vals_gpu = cp.asarray(vals)
    rows_gpu = cp.asarray(rows)
    cols_gpu = cp.asarray(cols)

    A = sp.coo_matrix(
        (vals_gpu, (rows_gpu, cols_gpu)),
        shape=(n_rows, n_cols)
    ).tocsr()

    b = cp.asarray(rhs_data)

    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    transfer_time = t1 - t0

    return A, b, transfer_time


# Load system
problem_name = "poisson2D_10x10" if len(sys.argv) < 2 else sys.argv[1]
A, b, transfer_time = load_matrix_rhs(problem_name)

# Initial residual norm (relative baseline)
r0 = b - A @ cp.zeros_like(b)
r0_norm = cp.linalg.norm(r0)

# ---- GPU solver timing ----
wall_start = time.perf_counter()

start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

start_event.record()

x, info = linalg.cg(A, b, tol=1e-6, maxiter=1000)

end_event.record()

gpu_time_ms = gpu_elapsed_ms(start_event, end_event)
wall_end = time.perf_counter()

wall_time = wall_end - wall_start

# Final residual norm
rf = b - A @ x
rf_norm = cp.linalg.norm(rf)

rel_reduction = (rf_norm / r0_norm).item()

# Power / energy estimate
power = get_power()
energy_joules = power * wall_time

# ---- reporting ----
print("\n=== Performance Metrics ===")
print(f"Transfer time (CPU → GPU): {transfer_time:.6f} s")
print(f"GPU solve time:            {gpu_time_ms / 1000:.6f} s")
print(f"Wall time (total):         {wall_time:.6f} s")
print(f"Power:                     {power:.1f} W")
print(f"Estimated energy:          {energy_joules:.3f} J")
print(f"Relative residual reduction: {rel_reduction:.3e}")

# Cleanup NVML
pynvml.nvmlShutdown()
