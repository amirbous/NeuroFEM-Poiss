import os
import json

import numpy as np

from qubo_visualization import *


def get_adjacency_matrix_from_txt(file_path, symmetric=True):
    """
    Returns the symmetric A matrix from a txt graph file of the format: ```i j w``` representing a
    connection from node `i` to node `j` with weight `w`
    """

    with open(file_path) as f:
        # read first line and extract matrix dimension
        firstline = f.readline()
        n_nodes, n_vertices, _ = str.split(firstline, ' ')
        n_nodes, n_vertices = int(n_nodes), int(n_vertices)

        # create empty adjacent matrix
        A = np.zeros((n_nodes, n_nodes))

        # iterate over lines and fill matrix
        for line in f.readlines():
            a, b, w = str.split(line, ' ')
            a = int(a) - 1
            b = int(b) - 1
            w = int(w)
            A[a, b] = w
            if symmetric:
                A[b, a] = w

    return A


def get_Q_from_A(A):
    """
    Returns the Q-matrix given an adjacency matrix
    """
    Q = A.copy()
    for i, row in enumerate(A):
        sum = 0
        for j, w in enumerate(row):
            sum += w
        Q[i,i] = -sum

    return Q


def get_number_of_used_cores(n_neurons, max_atoms_per_core):
    """
    Returns the number of cores needed to run the simulation

    :param n_neurons: Number of neurons
    :param max_atoms_per_core: Maximum number of atoms per core
    """
    n_cores = n_neurons // max_atoms_per_core
    if n_neurons % max_atoms_per_core != 0:
        n_cores += 1

    return n_cores


def neuron_states_dict_to_matrix(neuron_states_dict):
    # Number of neurons
    n_neurons = len(neuron_states_dict)

    # Number of time steps, assuming it's the max length of the lists in the dict values
    n_time_steps = max(len(s) for s in neuron_states_dict.values())

    # Initialize the spike matrix with zeros
    neuron_states_matrix = []

    # Fill the spike matrix
    for neuron_idx in neuron_states_dict.keys():
            neuron_states_matrix.append(neuron_states_dict[neuron_idx])

    return np.array(neuron_states_matrix).T


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        qubo = json.load(f)

    return qubo


def get_Q_from_JO_json(file_path):
    json_file = read_json_file(file_path)
    Q = np.array(json_file['coefficients'])

    return Q


def get_connections_stim_qubo(input_spikes_dict, n_inputs, n_neurons, weight=30):
    conns_stim_qubo = []
    #qubo_neuron_indices = random.sample(range(int(n_neurons)),  n_inputs)
    qubo_neuron_indices = [7]

    for i in input_spikes_dict.keys():
        conns_stim_qubo.append([i, qubo_neuron_indices[i], weight, 0])
        #conns_stim_qubo.append([i, qubo_neuron_indices[i] + int(n_neurons/2), -weight, 0])

    return conns_stim_qubo


def spike_matrix_timestep_pooling(spike_matrix, pool_size, stride):
    """
    Perform overlapping pooling over spike data

    :param spike_matrix: Spike data matrix
    :param pool_size: Pool size
    :param stride: Stride
    """

    n_neurons, n_timesteps = spike_matrix.shape

    # Calculate output width
    pooled_width = (n_timesteps - pool_size) // stride + 1
    pooled_matrix = np.zeros((n_neurons, pooled_width))

    for i in range(0, pooled_width):
        start_idx = i * stride
        end_idx = start_idx + pool_size
        pooled_matrix[:, i] = np.sum(spike_matrix[:, start_idx:end_idx], axis=1)

    # Convert values > 1 to 1 since it's binary data
    pooled_matrix = np.where(pooled_matrix > 0, 1, 0)

    return pooled_matrix


def write_results(experiment_dir, params_dict, results_dict):
    json.dump(params_dict, experiment_dir.joinpath('params.json').open('w'), indent=4)
    json.dump(results_dict, experiment_dir.joinpath('results.json').open('w'), indent=4)


def extract_energies(root_dir):
    max_energies = []
    all_energies = []

    # Walk through each folder
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if results.json exists in the folder
        if 'results.json' in filenames:
            with open(os.path.join(dirpath, 'results.json'), 'r') as file:
                data = json.load(file)

                # Extract max_energy_per_time if it exists
                if 'max_energy_per_time' in data:
                    max_energies.append(data['max_energy_per_time'])

                # Extract energy_per_time if it exists
                if 'energy_per_time' in data:
                    all_energies.append(data['energy_per_time'])

    return max_energies, all_energies


def get_JO_qubo_annealing_results(txt_file):
    """
    Reads a text file containing JSON data and extracts bitstrings and energies.

    Args:
        txt_file (str): The path to the text file.

    Returns:
        tuple: A tuple containing two lists. The first list contains the extracted bitstrings,
               where each bitstring is represented as a list of integers. The second list contains
               the extracted energies as strings.
    """
    with open(txt_file, 'r') as file:
        data = file.read()

    bitstrings = []
    energies = []
    json_data = json.loads(data)
    l = json_data['payload']['blob']['rawLines']
    strs = l[0].replace('[','').split('],')
    for i in range(len(strs)):
        if len(strs[i]) > 4000:
            b = list(strs[i].split(','))
            b = [int(s) for s in b]
            bitstrings.append(b)
        else:
            energies.append(strs[i].split(',')[-1])

    return bitstrings, energies


def quantize_to_precision(array, bits=8):
    """
    Quantize a 2D numpy array according to the specified bit precision so that the maximum positive value
    and the maximum negative value map to the maximum and minimum values allowed by the given bit precision,
    resembling a signed integer representation.

    :param array: 2D numpy array to be quantized
    :param bits: Number of bits for quantization
    :return: Quantized 2D numpy array with values adjusted to the specified bit precision
    """
    # Calculate the maximum positive and negative values allowed by the bit precision
    max_pos_value = 2**(bits - 1) - 1
    max_neg_value = -2**(bits - 1)

    # Find the maximum absolute value in the array to determine the scale
    max_abs_value = np.max(np.abs(array))

    # Calculate the scale factor
    scale = max_abs_value / max_pos_value

    # Quantization process
    quantized_array = np.round(array / scale)

    # Ensure values are within the range allowed by the bit precision
    quantized_array = np.clip(quantized_array, max_neg_value, max_pos_value).astype(np.int8)

    return quantized_array


def analyze_multi_run_results(results_dir, optimization='minimize'):
    best_energies = []

    best_energy = -np.inf if optimization == 'minimize' else np.inf

    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        json_file_path = os.path.join(item_path, 'results.json')
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            if optimization == 'minimize':
                best_energies.append(data['min_energy_per_time'])
            else:
                best_energies.append(data['max_energy_per_time'])


def validate_state_flips(spike_matrix, neuron_states):
    spikes = []
    states = []
    total_state_flips = 0

    for i in range(len(neuron_states) - 1):
        state_flip_index = np.where(neuron_states[i] != neuron_states[i + 1])[0]
        if state_flip_index.size == 0:
            states.append(99999)
        else:
            total_state_flips += state_flip_index.size
            states.append(state_flip_index.item())

        spike_index = np.where(spike_matrix[i] == 1)[0]
        if spike_index.size == 0:
            spikes.append(99999)
        else:
            spikes.append(spike_index.item())

    for i in range(1, len(spikes)):
        if spikes[i] != states[i - 1]:
            print("ANOMALY DETECTED")

    return spikes, states, total_state_flips


def get_adjacency_matrix_from_Q(Q):
    # Create an adjacency matrix where non-zero values are replaced with ones
    adjacency_matrix = np.where(Q != 0, 1, 0)

    # Set the diagonal elements to zero
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix


def process_multirun_results(experiment_dir, mode='maximize', plot=False):
    files = os.listdir(experiment_dir)
    neuron_files = sorted([file for file in files if file.startswith('neuron_states_') and file.endswith('.npy')])
    voltage_files = sorted([file for file in files if file.startswith('voltages_') and file.endswith('.npy')])
    energy_files = sorted([file for file in files if file.startswith('energy_') and file.endswith('.npy')])

    voltage_files = voltage_files[-100:]

    states = []
    for file in neuron_files:
        array_path = os.path.join(experiment_dir, file)
        states.append(np.load(array_path))

    energies = []
    for file in energy_files:
        array_path = os.path.join(experiment_dir, file)
        energies.append(np.load(array_path))

    concatenated_states = np.concatenate(states, axis=0)
    concatenated_energies = np.concatenate(energies, axis=0)


    if mode == 'maximize':
        best_energy = max(concatenated_energies)
        max_index = np.where(concatenated_energies == max(concatenated_energies))[0]
        best_solution = concatenated_states[max_index]
    else:
        best_energy = min(concatenated_energies)
        min_index = np.where(concatenated_energies == min(concatenated_energies))[0][0]
        best_solution = concatenated_states[min_index]

    voltages = []
    for file in voltage_files:
         array_path = os.path.join(experiment_dir, file)
         voltages.append(np.load(array_path))

    #concatenated_voltages = np.concatenate(voltages, axis=1)

    #noisethlds = []
    #for file in noisethld_files:
    #    array_path = os.path.join(experiment_dir, file)
    #    noisethlds.append(np.load(array_path))

    #concatenated_noisethlds = np.concatenate(noisethlds, axis=1)

    if plot:
        plot_spikes_energy(concatenated_states.T, concatenated_energies, experiment_dir=experiment_dir, cmap='black',
                           mode=mode)

    print("############")
    print("MAX ENERGY: {}".format(max(concatenated_energies)))
    print("############")

    return best_energy, best_solution


def validate_solution(Q, states, voltages):
    total_state_flips = 0
    state_changes = []
    for i in range(len(states) - 1):
        state_flip_index = np.where(states[i] != states[i + 1])[0]
        if state_flip_index.size == 0:
            state_changes.append(99999)
        else:
            total_state_flips += state_flip_index.size
            state_changes.append(state_flip_index.item())
    valid = []
    for i in range(len(state_changes)):
        flip_index = state_changes[i]
        receiving_indices = np.where(Q[flip_index] == 1)
        flag = True
        for j in range(len(receiving_indices)):
            if voltages[receiving_indices[0][j]][i] != voltages[receiving_indices[0][j]][i + 1]:
                continue
            else:
                flag = False

        if flag == False:
            print(state_changes[i])
            print('---')
            print(receiving_indices)
            print('---')
            print(voltages[receiving_indices[0][j]])
            print('---')

        valid.append(flag)

    return valid


def validate(Q, states, voltages):
    spike_indices = []
    voltage_change_indices = []
    post_synaptic_indices = []
    negative_spike = []

    for i in range(len(states) - 2):
        s = states[i] == states[i + 1]
        spike_index = np.where(s == False)[0]
        if (len(spike_index) == 0):
            spike_indices.append(-1)
        elif(len(spike_index) > 1):
            spike_indices.append(9999)
        else:
            spike_indices.append(spike_index[0])
            if states[i + 1][spike_index] == 1:
                negative_spike.append(False)
            elif states[i + 1][spike_index] == -1:
                negative_spike.append(True)

        voltage_change_indices.append(np.where(voltages[:,i+1] != voltages[:,i+2])[0].tolist())

        if spike_indices[-1] != -1:
            post_synaptic_indices.append(np.where(Q[spike_indices[-1]] == 1)[0])
        else:
            post_synaptic_indices.append([])

        if np.array_equal(voltage_change_indices[-1], post_synaptic_indices[-1]):
            print('TRUE')
        else:
            print('FALSE')

    return spike_indices, voltage_change_indices, post_synaptic_indices, negative_spike
