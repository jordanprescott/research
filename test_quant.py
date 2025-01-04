import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import unitary_group
from itertools import product
from functools import reduce


def generate_hierarchical_structure(num_qubits):
    """
    Generate a hierarchical unitary structure for a given number of qubits.

    Args:
        num_qubits (int): The number of qubits in the system (must be a power of 2).

    Returns:
        dict: A dictionary containing:
            - "final_state": The normalized final state as a flattened 1D array.
            - "projections": A dictionary mapping product state labels to probabilities.
            - "sum_probabilities": The sum of all probabilities (should be 1.0).
    """
    assert num_qubits > 1 and (num_qubits & (num_qubits - 1)) == 0, \
        "num_qubits must be a power of 2 and greater than 1"

    # Define down state
    down_state = np.array([1, 0])
    d = len(down_state)
    interaction_dim = 2 * d

    # Generate all product labels and kets
    all_prod_labels = list(product(range(d), repeat=num_qubits))
    all_prod_kets = {label: reduce(np.kron, [np.eye(1, d, i)[0] for i in label]) for label in all_prod_labels}

    # Generate random unitaries
    num_layers = num_qubits - 1  # One fewer than the number of qubits
    unitaries = [unitary_group.rvs(interaction_dim).reshape([d] * 4) for _ in range(num_layers)]

    # Recursive function to combine qubit groups
    def combine_groups(states, unitary_idx):
        if len(states) == 1:
            return states[0]  # Base case: only one state remains
        next_states = []
        for i in range(0, len(states), 2):
            # Reshape tensors explicitly for proper alignment
            state_1 = states[i].reshape(2, 2, -1)
            state_2 = states[i + 1].reshape(2, 2, -1)
            unitary = unitaries[unitary_idx]

            # Combine two groups with a unitary
            combined_state = np.einsum(
                'abc,def,mnij->mnabcf', state_1, state_2, unitary
            )
            # Flatten and normalize
            combined_state = combined_state.reshape(-1)
            combined_state /= np.linalg.norm(combined_state)
            # Expand back for next layer
            new_dim = int(np.sqrt(combined_state.size))
            next_states.append(combined_state.reshape(new_dim, new_dim))
        return combine_groups(next_states, unitary_idx + len(states) // 2)

    # Step 1: Initialize states for individual qubits
    initial_states = [np.einsum('i,j->ij', down_state, down_state) for _ in range(num_qubits // 2)]

    # Step 2: Generate final state through hierarchical combination
    final_state_tensor = combine_groups(initial_states, 0)
    final_state = final_state_tensor.reshape(-1)  # Flatten the final state

    # Step 3: Compute projections and probabilities
    projections = {
        label: abs(np.dot(all_prod_kets[label].conj(), final_state)) ** 2
        for label in all_prod_labels
    }
    sum_probabilities = np.sum(list(projections.values()))

    return {
        "final_state": final_state,
        "projections": projections,
        "sum_probabilities": sum_probabilities,
    }


def generate_hierarchical_circuit_diagram(num_qubits):
    """
    Generate a hierarchical circuit diagram for the given number of qubits.

    Args:
        num_qubits (int): The number of qubits (must be a power of 2).

    Returns:
        None. Displays a diagram of the hierarchical circuit.
    """
    assert num_qubits > 1 and (num_qubits & (num_qubits - 1)) == 0, \
        "num_qubits must be a power of 2 and greater than 1"

    # Create a directed graph to represent the hierarchical structure
    G = nx.DiGraph()

    # Step 1: Add nodes for the initial qubits
    for i in range(num_qubits):
        G.add_node(f"Q{i}", layer=0)

    # Step 2: Add layers of unitary operations
    layer = 1
    current_qubits = [f"Q{i}" for i in range(num_qubits)]
    while len(current_qubits) > 1:
        next_qubits = []
        for i in range(0, len(current_qubits), 2):
            unitary_name = f"U{layer}-{i // 2}"
            G.add_node(unitary_name, layer=layer)
            G.add_edge(current_qubits[i], unitary_name)
            G.add_edge(current_qubits[i + 1], unitary_name)
            next_qubit = f"Q{len(G.nodes)}"
            G.add_node(next_qubit, layer=layer + 1)
            G.add_edge(unitary_name, next_qubit)
            next_qubits.append(next_qubit)
        current_qubits = next_qubits
        layer += 2

    # Step 3: Create the circuit layout
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(12, 6))

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Add title and display
    plt.title(f"Hierarchical Circuit Diagram for {num_qubits} Qubits", fontsize=14)
    plt.axis("off")
    plt.show()


# Example usage
num_qubits = 8
result = generate_hierarchical_structure(num_qubits)

# Display results
print("Final state shape:", result["final_state"].shape)
print("Sum of probabilities:", result["sum_probabilities"])
print("Projections:" + "".join(f"\n  {label}: {prob}" for label, prob in result["projections"].items()))
print("Projection shape:", len(result["projections"]))

# Generate and display the circuit diagram
generate_hierarchical_circuit_diagram(num_qubits)
