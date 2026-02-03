import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qubo_utils import *

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from matplotlib.patches import Patch, Circle


def draw_snn_clustered(weight_matrix, node_signs=None, clusters=None, 
                      title="SNN with Functional Clusters", 
                      figsize=(14, 10), dpi=300):
    """
    Draw SNN with neurons grouped by mesh point/cluster.
    
    Parameters:
    -----------
    weight_matrix : numpy.ndarray
        SNN weight matrix
    node_signs : numpy.ndarray, optional
        Array of +1 for excitatory, -1 for inhibitory neurons
    clusters : list of lists
        List where each element is a list of neuron indices belonging to one mesh point
    title : str
        Plot title
    """
    
    if hasattr(weight_matrix, 'toarray'):
        weight_matrix = weight_matrix.toarray()
    
    n_neurons = weight_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))
    
    # Add edges
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            weight = weight_matrix[i, j]
            if abs(weight) > 1e-10:
                G.add_edge(i, j, weight=weight)
    
    # Determine node signs if not provided
    if node_signs is None:
        node_signs = np.ones(n_neurons)
        for i in range(n_neurons):
            # Excitatory if net outgoing positive weight > negative
            pos_weights = weight_matrix[i, weight_matrix[i] > 0]
            neg_weights = weight_matrix[i, weight_matrix[i] < 0]
            if len(pos_weights) > 0 and len(neg_weights) > 0:
                if np.mean(pos_weights) < np.mean(np.abs(neg_weights)):
                    node_signs[i] = -1
    
    # If clusters not provided, create default ones
    if clusters is None:
        # Assuming 4 neurons per mesh point
        n_clusters = n_neurons // 4
        clusters = []
        for i in range(n_clusters):
            clusters.append(list(range(i*4, (i+1)*4)))
    
    # Create layout with clusters
    pos = {}
    
    # Arrange in a grid of clusters
    n_cols = int(np.ceil(np.sqrt(len(clusters))))
    n_rows = int(np.ceil(len(clusters) / n_cols))
    
    rng = np.random.default_rng(1)   # reproducible layout

    cluster_jitter = 0.2           # jitter magnitude (relative to spacing=2)

    for cluster_idx, neuron_indices in enumerate(clusters):
        row = cluster_idx // n_cols
        col = cluster_idx % n_cols
        
        # Position for this cluster
        dx, dy = rng.normal(0, cluster_jitter, size=2)

        base_x = col * 2 + dx
        base_y = row * 2 + dy

        # Arrange neurons in a circle within cluster
        n_in_cluster = len(neuron_indices)
        for i, neuron in enumerate(neuron_indices):
            angle = 2 * np.pi * i / n_in_cluster
            radius = 0.3
            pos[neuron] = (base_x + radius * np.cos(angle), 
                        base_y + radius * np.sin(angle))

    
    plt.figure(figsize=(12, 12), dpi=dpi)
    
    # Draw edges first (behind nodes) - ALL BLACK
    edges = list(G.edges())
    edge_widths = []
    
    # Calculate edge widths based on weight magnitude
    for u, v in edges:
        weight = G[u][v]['weight']
        edge_widths.append(0.5 + 2 * abs(weight) / (np.max(np.abs(weight_matrix)) if np.max(np.abs(weight_matrix)) > 0 else 1))
    
    # All edges are black
    nx.draw_networkx_edges(G, pos, edgelist=edges, 
                          edge_color='black', alpha=0.4, 
                          width=edge_widths, style='solid')
    
    # Draw nodes with +/- signs and colors based on node_signs

    
    for cluster_idx, neuron_indices in enumerate(clusters):
        # Get node colors for this cluster
        node_colors = []
        for neuron in neuron_indices:
            if node_signs[neuron] > 0:
                node_colors.append('green')
            else:
                node_colors.append('red')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=neuron_indices,
                              node_color=node_colors,
                              node_size=1500, edgecolors='black', 
                              linewidths=2.0)
        
        # Add +/- labels inside nodes
        for neuron in neuron_indices:
            x, y = pos[neuron]
            sign = '+' if node_signs[neuron] > 0 else '−'
            plt.text(x, y, sign, 
                    ha='center', va='center',
                    fontsize=15, fontweight='bold',
                    color='black')
    
    # Draw cluster boundaries with continuous black lines
    for cluster_idx, neuron_indices in enumerate(clusters):
        if neuron_indices:
            # Get center of cluster
            cluster_pos = [pos[n] for n in neuron_indices]
            center_x = np.mean([p[0] for p in cluster_pos])
            center_y = np.mean([p[1] for p in cluster_pos])
            
            # Find radius that encloses all nodes in cluster
            radius = max([np.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) 
                         for p in cluster_pos]) + 0.2
            
            # Draw cluster boundary (continuous black line)
            # make the circle discontinuous by using a FancyBboxPatch
            
            circle = plt.Circle((center_x, center_y), radius + 0.2, 
                              fill=False, linestyle='--', 
                              color='black', linewidth=2.0, alpha=0.8)
            
            if cluster_idx == len(clusters) - 3:
                print("Haw zabbi")
                plt.gca().add_patch(circle)
            # Add cluster label

                plt.text(center_x, center_y + radius + 0.4, f'Neurons correspoding \nto one mesh point', 
                        ha='center', va='center', fontsize=32,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', 
                                edgecolor='black',
                                alpha=0.9))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Add legend (simplified since edges are all black now)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', linewidth=2, label='Neurons projecting \nwith positive weights (+)'),
        Patch(facecolor='red', edgecolor='black', linewidth=2, label='Neurons projecting \nwith negative weights (−)'),
    ]
    plt.legend(handles=legend_elements, loc='lower left', 
              frameon=True, fancybox=True, framealpha=0.9,
              edgecolor='gray', fontsize=36)
    
    plt.savefig("snn_clustered.svg", dpi=dpi, bbox_inches='tight')
    plt.show() 

def plot_spikes_energy(spike_matrix, energy_per_time, experiment_dir=None, cmap='viridis', ceil=False, mode='maximize'):
    """
    """

    fig, ax = plt.subplots(nrows=2, sharex=True)

    spike_matrix = np.where(spike_matrix==-1, 0, spike_matrix)
    spike_times = [np.where(row)[0] for row in spike_matrix]
    if cmap == 'black':
        cmap = mcolors.ListedColormap(['black'])
    else:
        cmap = getattr(plt.cm, cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, len(spike_times))]

    ax[0].eventplot(spike_times, colors=colors)
    ax[1].plot(range(spike_matrix.shape[1]), energy_per_time, c='k')

    if mode == 'maximize':
        max_index = np.argmax(energy_per_time)
        ax[1].text(max_index, max(energy_per_time), 'Max: {}'.format(max(energy_per_time)), fontsize=8, color='red',
                   bbox=dict(facecolor='white', alpha=0.5))
    else:
        min_index = np.argmin(energy_per_time)
        ax[1].text(min_index, min(energy_per_time), 'Min: {}'.format(min(energy_per_time)), fontsize=8, color='red',
                   bbox=dict(facecolor='white', alpha=0.5))

    if ceil:
        ax[1].set_ylim(0, math.ceil(max(energy_per_time)/100)*100)

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)

    ax[0].set_ylabel('QUBO')
    ax[1].set_ylabel('Energy')
    ax[1].set_xlabel('Timesteps x 100')

    if experiment_dir is not None:
        plt.savefig(experiment_dir.joinpath('spikes_energy'), dpi=300)
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()


def plot_multirun_energy(results_dir):
    """
    Plot mean and standard deviation of energy_per_time from multiple runs
    """

    _, all_energy = np.array(extract_energies(results_dir))
    mean_energies = np.mean(all_energy, axis=0)
    std_energies = np.std(all_energy, axis=0)

    plt.plot(mean_energies, label='Mean Energy')
    plt.fill_between(range(len(mean_energies)), mean_energies - std_energies,
                     mean_energies + std_energies, color='gray', alpha=0.5,
                     label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Mean and Std Dev of Energy over Time')
    plt.legend()
    plt.show()


def plot_average_maximum_energy(results_dir):
    max_energies, _ = np.array(extract_energies(results_dir))

    # Calculate mean and standard deviation
    mean_value = np.mean(max_energies)
    std_value = np.std(max_energies)

    # Create bar plot
    plt.bar(['Mean'], [mean_value], yerr=[std_value], color='blue', capsize=10, width=0.02)

    print('Mean value: {}'.format(mean_value))
    print('Standard Deviation: {}'.format(std_value))
    # Set title and labels
    plt.title('Average maximum energy over {} runs'.format(len(max_energies)))
    plt.ylabel('Threshold: 0.1')
    plt.show()


def plot_JO_bitstrings_as_events(bitstrings):
    num_variables = len(bitstrings[0])

    # Convert bitstrings to a list of event times
    events = [[] for _ in range(num_variables)]
    for index, bitstring in enumerate(bitstrings):
        for bit_index, bit in enumerate(bitstring):
            if bit == 1:
                events[bit_index].append(index)  # Event time is the index of the bitstring

    # Create an event plot
    plt.eventplot(events, orientation='horizontal')
    plt.show()


def draw_partitioned_graph(A, solution):
    G = nx.Graph()
    num_nodes = len(A)

    # Add nodes to the graph
    for node in range(1, num_nodes + 1):
        G.add_node(node)

    # Add edges to the graph based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if A[i, j] == 1:
                G.add_edge(i + 1, j + 1)  # Adjust indices for 1-based node numbering

    pos = {}
    left_y = 0
    right_y = 0

    for i, group in enumerate(solution, start=1):
        if group == 0:
            # Place this node on the left
            pos[i] = (-1, left_y)
            left_y += 1
        else:
            # Place this node on the right
            pos[i] = (1, right_y)
            right_y += 1

    # Prepare node colors
    color_map = ['red' if group == 0 else 'blue' for group in solution]

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, font_size=16)
    plt.savefig("partitioned_graph.png", dpi=300)

