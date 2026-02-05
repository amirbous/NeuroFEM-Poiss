
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


##############################################################################
# Important: From HLRS neurofem.ipynb notebook!!!!!!!
# source: hlrs neurofem tutorial
##############################################################################
# spinnaker2 imports
from spinnaker2 import snn, hardware
from spinnaker2.experiment_backends import BackendSettings, ExperimentBackendType
from spinnaker2.experiment_backends.backend_settings import ROUTING
import scipy as sp
from spinnaker2.experiment_backends.backend_settings import ROUTING, BackendSettings
import scipy.sparse.linalg as spla




def enable_async_start():
    """Disable synchronous start to avoid blocking on failed cores."""
    import spinnaker2.mapper as s2_mapper

    if getattr(s2_mapper, "_async_start_patched", False):
        return

    original = s2_mapper.Mapper.map_and_generate_experiment_config

    def _map_and_generate_experiment_config_async(self, *args, **kwargs):
        exp_config, sim_cfg = original(self, *args, **kwargs)
        exp_config.synchronous_start = False
        return exp_config, sim_cfg

    s2_mapper.Mapper.map_and_generate_experiment_config = _map_and_generate_experiment_config_async
    if hasattr(s2_mapper, "IterativeMapper"):
        s2_mapper.IterativeMapper.map_and_generate_experiment_config = _map_and_generate_experiment_config_async
    s2_mapper._async_start_patched = True



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
def LoadCSRMatrix(model_name="", data_dir="../../data/"):
    
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


def BuilSNNConnections(A_csr=None, b=None, nmesh=0, npm=8, gamma=5):

    print("Building SNN connections...")                  

    row_indices, col_indices = A_csr.nonzero()
    nnz = len(row_indices)  
    conns = np.zeros((nnz, 4), dtype=int)

    for idx, (i_mesh, j_mesh) in enumerate(zip(row_indices, col_indices)):
        value = A_csr[i_mesh, j_mesh]
        conns[idx] = [
            j_mesh * npm,  # pre_neuron_id
            i_mesh * npm,  # post_neuron_id
            value,
            0
        ]
    nb_neurons = nmesh * npm
    print(f"Total number of neurons: {nb_neurons}")
    return conns.tolist(), nb_neurons



def InitializeSNNInstance(conns, A_quant, n_mesh, nb_neurons, npm, neurons_per_core, neuron_params):

    nb_cores = nb_neurons // neurons_per_core if nb_neurons % neurons_per_core == 0 else (nb_neurons // neurons_per_core) + 1

    pop = snn.Population(size=nb_neurons, neuron_model="neurofem_2048", params=neuron_params, name="pop", record=["x_mean"])

    pop.set_max_atoms_per_core(neurons_per_core)

    A_density = A_quant.nnz / A_quant.size
    print(f"Sparse matrix A density: {A_density*100:.4f}%")

    conns_density = len(conns) / ((n_mesh * npm)**2)
    print(f"Connection density: {conns_density*100:.6f}%")

    proj = snn.Projection(pre=pop, post=pop, connections=conns, name="proj")

    net = snn.Network("Ax=b Network")
    net.add(pop, proj)

    return net, pop, proj



def RunSNNInstance(net, num_timesteps, sys_tick_in_s):

 # 0. Set SCP timeout FIRST before anything else (only safe change)
    try:
        from spinnman import constants as spinnman_constants
        original_timeout = spinnman_constants.SCP_TIMEOUT
        spinnman_constants.SCP_TIMEOUT = 5.0
        print(f"[DEBUG] SCP_TIMEOUT changed from {original_timeout}s to {spinnman_constants.SCP_TIMEOUT}s", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to set SCP timeout: {e}", flush=True)

 # 1. Access the backend settings
    settings = BackendSettings()
    
    # 2. Force the routing type to C2C and disable single-chip mode
    # This ensures the AppBuilder passes 'ROUTING_SETTINGS=C2C_ROUTING SINGLE_CHIP=0'
    settings.routing_type = ROUTING.C2C
    settings._single_chip = False 
    
    # Existing IP logic
    STM_IP = os.environ.get("STM_IP", "192.168.1.2")
    ETH_IP = os.environ.get("S2_IP", "192.168.1.17")
    
    hw = hardware.SpiNNcloud48NodeBoard(eth_ip=ETH_IP, stm_ip=STM_IP)
    # if the chip does not exit when extracting results. 
    #enable_async_start()
    # This call will now trigger the compiler with the correct flags
    hw.run(net, num_timesteps, debug=False, mapping_only=False, reset_board=True, sys_tick_in_s=sys_tick_in_s)
    
    return hw

def get_solution(net, pop, nmesh, meshes_per_core, npm, num_timesteps, steady_state):

    def get_solution_inner(x_means):
        solution = []
        for i in range(nmesh):
            index = i % meshes_per_core + (i // meshes_per_core) * npm * meshes_per_core
            r = x_means[index] / (num_timesteps * steady_state + 1)
            solution.append(r)
        return np.array(solution)

    x_mean = pop.get_x_mean()
    solution = get_solution_inner(x_mean)

    return solution

def main():

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Sphere_00"


    print("Starting CSR to SNN pipeline...")

    #1. Load CSR Matrix and RHS
    A_csr, b = LoadCSRMatrix(model_name)
    n_mesh = A_csr.shape[0]

    print("1. Loaded matrix and RHS.")


    A_quant, A_scale = float_to_signed_sparse(A_csr, x_bits=21)


    npm = 8                            # neurons per node
    r = npm // 2                       # num neurons with positive sign

    num_timesteps = 10000              # timesteps
    sys_tick_in_s = 1e-3              # (default is 1 ms)

    gamma = 10.0                      # neuron gain
    
    
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


    #2. Build SNN Connections
    conns, nb_neurons = BuilSNNConnections(A_quant, b, n_mesh, npm, gamma=gamma)
    print("2. Built SNN connections.")

    # Use fixed target to avoid issues with large problems
    max_neurons_per_core = 2048
    target_meshes_per_core = 16
    neurons_per_core = min(max_neurons_per_core, target_meshes_per_core * npm)
    meshes_per_core = neurons_per_core // npm

    net, pop, proj = InitializeSNNInstance(conns, A_quant, n_mesh, nb_neurons, npm, neurons_per_core, neuron_params)
    print("3. Initialized SNN instance.")


    hw = RunSNNInstance(net, num_timesteps, sys_tick_in_s)
    print("4. Ran SNN instance.")


    sol = get_solution(net, pop, n_mesh, meshes_per_core, npm, num_timesteps, steady_state)
    
    
    sol_exact = spla.spsolve(A_csr, b)
    error = np.linalg.norm(sol - sol_exact) / np.linalg.norm(sol_exact)
    print(f"Relative error compared to exact solution: {error:.6e}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sol_exact, label='Exact Solution', marker='o')
    plt.plot(sol, label='SNN Solution', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Solution Value')
    plt.title('Comparison of Exact and SNN Solutions')
    plt.legend()
    plt.grid()
    plt.savefig(f"{model_name}_solution_comparison.png")
    
    print("5. Retrieved solution from SNN.")




if __name__ == "__main__":      
    main()


