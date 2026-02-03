import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

import tqdm

import sys
import os




#########################################################
################@Copyright from sandilabs################
#########################################################
def generate_gamma_sparse(n_sys, neurons_per_mesh_point, gamma_norm):   
    '''
    Generates the Gamma matrix mapping neurons to readout variables.  
    n_sys: number of readout variables
    neurons_per_mesh_point: number of neurons to place at each mesh node
    gamma_norm:  Magnitude of the readout kernel for each neuron
    
    '''
    
    n_neurons = neurons_per_mesh_point * n_sys

    gamma_data = []
    gamma_row = []
    gamma_col = []
    for pt in range(n_sys):
      
        start_idx = pt * neurons_per_mesh_point
        half_npm = neurons_per_mesh_point//2
        
        gamma_row.extend(neurons_per_mesh_point*[pt])
        gamma_col.extend(list(range(start_idx, start_idx + neurons_per_mesh_point)))
        gamma_data.extend(half_npm*[-gamma_norm])
        gamma_data.extend(half_npm*[gamma_norm])
         
    return sp.sparse.csr_array((np.array(gamma_data), (np.array(gamma_row), np.array(gamma_col))), shape=(n_sys, n_neurons))

#########################################################
################@Copyright from sandilabs################
#########################################################
def create_spiking_fem_network_sparse(A_sys, gamma, gamma_norm, lambda_d=10, lambda_v=20, mu=1e-6, nu=1e-5, tau_A=0.1):
    '''
    Generates sparse weight matrices, thresholds, and reset voltages for a NeuroFEM network implementing the
    linear system A_sys with readout kernel gamma
    '''
    n_sys, n_neurons = gamma.shape
    gTg = (gamma.T) @ gamma
    omega_f = gTg + mu*(lambda_d**2) * sp.sparse.eye_array(n_neurons)
    
    GTAG = (gamma.T) @ (A_sys / tau_A) @ gamma  # should be sparse
    
    threshs = 0.5 * (nu*lambda_d**2 + (gamma_norm**2) * np.ones(n_neurons))
    v_reset = threshs - (gamma_norm**2 + mu*(lambda_d**2)) * np.ones(n_neurons)
    omega_f = omega_f - (gamma_norm**2 + mu*(lambda_d**2)) * sp.sparse.eye_array(n_neurons)

    return omega_f, GTAG, threshs, v_reset



def read_csr_matrx(mtx_filename):
    try:
    
        raw_data = np.loadtxt(mtx_filename, skiprows=1, dtype=float)
        
        
        with open(mtx_filename, 'r') as f:
            header = f.readline().split()
            n_rows = int(header[0])
            n_cols = int(header[1])


        rows = raw_data[:, 0].astype(int) - 1
        cols = raw_data[:, 1].astype(int) - 1
        values = raw_data[:, 2]


        A = sp.sparse.csr_matrix((values, (rows, cols)), shape=(n_rows, n_cols))


        initial_nnz = A.nnz
        A.eliminate_zeros()

        return A
    except Exception as e:
        print(f"Failed to load matrix: {e}")
        sys.exit(1)


def write_vec(filename, vec):
    with open(filename, "w") as f:
        f.write(f"{vec.size}\n")          # first line: number of entries
        for x in vec:
            f.write(f"{x:.16e}\n")        # one value per line

def read_vec(filename):
    try:
        b = np.loadtxt(filename, dtype=float, skiprows=1)
        return b
    except Exception as e:
        print(f"Failed to load RHS: {e}")
        sys.exit(1)

def write_vec(filename, vec):
    with open(filename, "w") as f:
        f.write(f"{vec.size}\n")          # first line: number of entries
        for x in vec:
            f.write(f"{x:.16e}\n")        # one value per line

def read_vec(filename):
    try:
        b = np.loadtxt(filename, dtype=float, skiprows=1)
        return b
    except Exception as e:
        print(f"Failed to load RHS: {e}")
        sys.exit(1)

def complete_run_neurofem(A, b, npm, timesteps):
    #### Converting the matrix

    #### The power of two defining the magnitude of gamma

    ############################################################
    ############# @Copyright from sandilabs ####################
    ############################################################
    gamma_pow2 = -6 # In the paper, we used gamma = 2^-6 or gamma = 2^-8
    s_gamma = 7 - gamma_pow2
    gamma_norm = 15

    #### The power of two to which we rescale the slow weight matrix
    omega_s_pow2 = 1
    omega_max = 2**omega_s_pow2 - 2**-(7-omega_s_pow2)


    #### network parameters
    lambda_d = 8              # Hz; slow variable time constant.  Default = 10 Hz
    lambda_v = 16             # Hz; membrane potential time constant. Default = 20 Hz

    ki = 16                   # integral control gain
    kp = 4                    # proportional control gain


    N_interior = A.shape[0]
    neurons_per_mesh_point = npm
    n_neurons = neurons_per_mesh_point * N_interior

    sigma_v = 0.00225 # We kept this constant for all meshes
    tau_A = (gamma_norm**2) * np.amax(np.abs(A))/omega_max
    mu = 0  # spike L2 norm penalization; set to 0
    nu = 0  # spike L1 norm penalization; set to 0

    # create system
    gamma = generate_gamma_sparse(N_interior, neurons_per_mesh_point, gamma_norm)
    # Added gamma_norm in the correct position
    omega_f, GTAG, threshs, v_reset = create_spiking_fem_network_sparse(
        -A,
        gamma,
        gamma_norm,
        lambda_d=lambda_d,
        lambda_v=lambda_v,
        mu=mu,
        nu=nu,
        tau_A=tau_A
    )

    dt = 2**(-12)             # power of two
    n_timesteps = timesteps

    # neuron biases
    bias_f1 = gamma.T @ (b / tau_A)
    n_sys, n_neurons = gamma.shape

    # Initialize variables
    V = np.zeros(n_neurons)
    spikes = np.zeros((n_neurons, 1), dtype=np.int8)
    output = np.zeros((n_sys, n_timesteps))

    # Feedback current (u1) and Integral error (u_int)
    u1 = np.zeros(n_neurons)
    u_int = np.zeros(n_neurons)

    # This input is constant for a static Ax=b problem
    c_in_fixed_gamma = np.copy(bias_f1)

    for step in (range(1, n_timesteps)):


        du1 = dt * (-lambda_d * u1) + GTAG @ spikes[:, 0]
        u1 += du1


        u_err = u1 + c_in_fixed_gamma


        u_int += dt * u_err

        dv = dt * (-lambda_v * V + kp * u_err + ki * u_int) \
            - omega_f.dot(spikes[:, 0]) \
            + sigma_v * np.random.randn(n_neurons)
        V += dv


        spikes[:, 0] = np.greater(V, threshs).astype(np.int8)


        has_spiked = spikes[:, 0] > 0
        V[has_spiked] = v_reset[has_spiked]


        output[:, step] = (1 - dt * lambda_d) * output[:, step-1] + gamma.dot(spikes[:, 0])

    x_estimated = output[:, -1]


    err_2_norm = np.linalg.norm(x_estimated - x0, ord=2)

    return err_2_norm
def complete_run_neurofem_history(A, b, npm, timesteps):
    #### Converting the matrix
    #### The power of two defining the magnitude of gamma
    gamma_pow2 = -6 
    s_gamma = 7 - gamma_pow2
    gamma_norm = (2**gamma_pow2 - 2**-s_gamma)

    #### The power of two to which we rescale the slow weight matrix
    omega_s_pow2 = 1 
    omega_max = 2**omega_s_pow2 - 2**-(7-omega_s_pow2)

    #### network parameters
    lambda_d = 8              
    lambda_v = 16             

    ki = 16                   
    kp = 4                    

    # Assumes A is in global scope
    N_interior = A.shape[0]
    neurons_per_mesh_point = npm
    n_neurons = neurons_per_mesh_point * N_interior
            
    sigma_v = 0.00225 
    tau_A = (gamma_norm**2) * np.amax(np.abs(A))/omega_max
    mu = 0  
    nu = 0  

    # create system
    gamma = generate_gamma_sparse(N_interior, neurons_per_mesh_point, gamma_norm)
    
    omega_f, GTAG, threshs, v_reset = create_spiking_fem_network_sparse(
        -A, 
        gamma, 
        gamma_norm, 
        lambda_d=lambda_d, 
        lambda_v=lambda_v, 
        mu=mu, 
        nu=nu, 
        tau_A=tau_A
    )

    dt = 2**(-12)             
    n_timesteps = timesteps

    # neuron biases (Assumes b is in global scope)
    bias_f1 = gamma.T @ (b / tau_A)
    n_sys, n_neurons = gamma.shape

    # Initialize variables
    V = np.zeros(n_neurons)
    spikes = np.zeros((n_neurons, 1), dtype=np.int8)
    
    # 'output' already stores the history: shape (System Size x Time Steps)
    output = np.zeros((n_sys, n_timesteps))
    
    # NEW: Array to store the L2 error at every time step
    error_history = np.zeros(n_timesteps)

    # Feedback current (u1) and Integral error (u_int)
    u1 = np.zeros(n_neurons)
    u_int = np.zeros(n_neurons)

    # This input is constant for a static Ax=b problem
    c_in_fixed_gamma = np.copy(bias_f1)
    
    # Calculate error for the initial state (step 0, where output is 0)
    # Assumes x0 (ground truth) is in global scope
    if 'x0' in globals():
        error_history[0] = np.linalg.norm(output[:, 0] - x0, ord=2)

    for step in (range(1, n_timesteps)):
        
        du1 = dt * (-lambda_d * u1) + GTAG @ spikes[:, 0]
        u1 += du1

        u_err = u1 + c_in_fixed_gamma
        
        u_int += dt * u_err
            
        dv = dt * (-lambda_v * V + kp * u_err + ki * u_int) \
            - omega_f.dot(spikes[:, 0]) \
            + sigma_v * np.random.randn(n_neurons)
        V += dv
            
        spikes[:, 0] = np.greater(V, threshs).astype(np.int8)

        has_spiked = spikes[:, 0] > 0
        V[has_spiked] = v_reset[has_spiked]

        # Update the estimated solution for the current step
        output[:, step] = (1 - dt * lambda_d) * output[:, step-1] + gamma.dot(spikes[:, 0])

        # NEW: Calculate and store the error for the current step
        # x_estimated at this step is simply output[:, step]
        if 'x0' in globals():
            error_history[step] = np.linalg.norm(output[:, step] - x0, ord=2)

    # Returns:
    # 1. output: The full matrix of size (N_interior, timesteps). output[:, i] is x_estimated at step i.
    # 2. error_history: Array of size (timesteps,) containing the L2 norm error at each step.
    return output, error_history


problem_name = "Sphere_02"

mtx_filename = f"data/matrix/{problem_name}_mtx.txt"
rhs_filename = f"data/matrix/{problem_name}_rhs.txt"
sol_filename = f"data/sol/{problem_name}_x0.txt"


print(os.getcwd())

A = read_csr_matrx(mtx_filename)
b = read_vec(rhs_filename)

x0 = read_vec(sol_filename)


npms = 8
timesteps = 10000

err_hist = np.zeros((len(npms), timesteps))
for i, npm in enumerate(npms):
    print(npm)
    _, err_hist[i, :] = complete_run_neurofem_history(A, b, npm, timesteps)
print("Finished computations, writing output")
with open("output_3.csv", "w") as f:
    print("npm;timesteps;error_norm", file=f)
    for i, npm in enumerate(npms):
        for j in range(timesteps):
            print(f"{npm};{j};{err_hist[i, j]}", file=f)

