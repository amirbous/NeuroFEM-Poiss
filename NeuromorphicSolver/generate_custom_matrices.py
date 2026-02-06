#!/usr/bin/env python3
"""
Generate Poisson matrices with guaranteed diagonal dominance for testing.
These matrices are ideal for neuromorphic solver testing because:
1. They are diagonally dominant (stable for Jacobi-like methods)
2. SPD (Symmetric Positive Definite)
3. Similar structure to FEM matrices
"""

import numpy as np
import scipy.sparse as sp
import sys
import os

def generate_poisson_2d(nx, ny, save_path=None):
    """
    Generate 2D Poisson equation matrix on rectangular grid.
    -∇²u = f  discretized with 5-point stencil.
    
    Args:
        nx, ny: Grid dimensions
        save_path: Base filename (without extension) to save matrix
    
    Returns:
        A: Sparse matrix (n×n where n=nx*ny)
        b: RHS vector
    """
    n = nx * ny
    
    # Build matrix using 5-point stencil
    # Each interior point: 4u_i - u_left - u_right - u_up - u_down = h²*f
    diag_main = 4.0 * np.ones(n)
    diag_x = -1.0 * np.ones(n-1)
    diag_y = -1.0 * np.ones(n-nx)
    
    # Fix boundary connections (don't connect across grid wrap)
    for i in range(1, ny):
        diag_x[i*nx - 1] = 0.0  # No connection at end of row
    
    # Build sparse matrix
    A = sp.diags([diag_main, diag_x, diag_x, diag_y, diag_y],
                  [0, -1, 1, -nx, nx], 
                  shape=(n, n), format='csr')
    
    # Generate RHS (e.g., constant forcing or sin pattern)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    b = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel())
    
    # Scale to low magnitude
    b = b * 1e-3
    
    if save_path:
        save_matrix(A, b, save_path)
    
    return A, b

def generate_poisson_3d(nx, ny, nz, save_path=None):
    """
    Generate 3D Poisson equation matrix on box grid.
    -∇²u = f discretized with 7-point stencil.
    
    Args:
        nx, ny, nz: Grid dimensions
        save_path: Base filename to save matrix
    
    Returns:
        A: Sparse matrix (n×n where n=nx*ny*nz)
        b: RHS vector
    """
    n = nx * ny * nz
    
    # 7-point stencil: 6 neighbors + center
    diag_main = 6.0 * np.ones(n)
    diag_x = -1.0 * np.ones(n-1)
    diag_y = -1.0 * np.ones(n-nx)
    diag_z = -1.0 * np.ones(n-nx*ny)
    
    # Fix boundaries (no wrap-around)
    for i in range(1, ny*nz):
        diag_x[i*nx - 1] = 0.0
    
    for i in range(1, nz):
        for j in range(nx):
            diag_y[i*nx*ny - nx + j] = 0.0
    
    A = sp.diags([diag_main, diag_x, diag_x, diag_y, diag_y, diag_z, diag_z],
                  [0, -1, 1, -nx, nx, -nx*ny, nx*ny],
                  shape=(n, n), format='csr')
    
    # RHS with 3D pattern
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    b = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel()) * np.sin(np.pi * Z.ravel())
    b = b * 1e-3
    
    if save_path:
        save_matrix(A, b, save_path)
    
    return A, b

def save_matrix(A, b, base_name):
    """Save matrix in the format expected by cleanpipeline_neurofem.py"""
    
    # Ensure directory exists
    os.makedirs("data/matrix", exist_ok=True)
    
    mtx_file = f"data/matrix/{base_name}_mtx.txt"
    rhs_file = f"data/matrix/{base_name}_rhs.txt"
    
    # Save matrix in COO format with header
    A_coo = A.tocoo()
    with open(mtx_file, 'w') as f:
        # Header: rows cols nnz
        f.write(f"{A.shape[0]} {A.shape[1]} {A.nnz}\n")
        # Data: row col value (1-indexed!)
        for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
            f.write(f"{i+1} {j+1} {v:20.16e}\n")
    
    # Save RHS
    with open(rhs_file, 'w') as f:
        f.write(f"{len(b)}\n")
        for val in b:
            f.write(f"{val:20.16e}\n")
    
    print(f"Saved: {mtx_file}, {rhs_file}")
    
    # Print diagnostics
    diag = A.diagonal()
    row_sums = np.abs(A).sum(axis=1).A1
    diag_abs = np.abs(diag)
    off_diag = row_sums - diag_abs
    dominance = diag_abs - off_diag
    
    weak_rows = np.sum(dominance < 0)
    print(f"  Size: {A.shape[0]}")
    print(f"  NNZ: {A.nnz}")
    print(f"  Diagonal dominance: {100*(1-weak_rows/A.shape[0]):.1f}% rows dominant")
    print(f"  Max off-diag/diag: {(off_diag/diag_abs).max():.3f}")
    print(f"  Cores needed: ~{int(np.ceil(A.shape[0]/16))}")
    print()

if __name__ == "__main__":
    print("="*80)
    print("POISSON MATRIX GENERATOR")
    print("="*80)
    
    # Generate test suite with different sizes
    test_suite = [
        ("poisson2D_10x10", "2d", 10, 10, None),
        ("poisson2D_20x20", "2d", 20, 20, None),
        ("poisson2D_30x30", "2d", 30, 30, None),
        ("poisson2D_50x50", "2d", 50, 50, None),
        ("poisson2D_70x70", "2d", 70, 70, None),
        ("poisson2D_90x90", "2d", 90, 90, None),
        ("poisson2D_100x100", "2d", 100, 100, None),
        ("poisson3D_5x5x5", "3d", 5, 5, 5),
        ("poisson3D_8x8x8", "3d", 8, 8, 8),
        ("poisson3D_10x10x10", "3d", 10, 10, 10),
        ("poisson3D_15x15x15", "3d", 15, 15, 15),
        ("poisson3D_20x20x20", "3d", 20, 20, 20),
    ]
    
    print("\nGenerating test matrices...")
    print("="*80)
    
    for suite in test_suite:
        name = suite[0]
        dim = suite[1]
        
        if dim == "2d":
            nx, ny = suite[2], suite[3]
            print(f"\n{name}: 2D Poisson {nx}×{ny} grid (n={nx*ny})")
            generate_poisson_2d(nx, ny, name)
        else:
            nx, ny, nz = suite[2], suite[3], suite[4]
            print(f"\n{name}: 3D Poisson {nx}×{ny}×{nz} grid (n={nx*ny*nz})")
            generate_poisson_3d(nx, ny, nz, name)
    
    print("="*80)
    print("DONE! Test these matrices with:")
    print("  python3 cleanpipeline_neurofem.py poisson2D_10x10")
    print("  python3 cleanpipeline_neurofem.py poisson3D_8x8x8")
    print("="*80)
