import quimb as qu
import quimb.tensor as qtn
from scipy.stats import unitary_group
import numpy as np


def generate_tensor_network(num_qubits):
    """
    Dynamically generate a tensor network based on the number of qubits.

    Args:
        num_qubits (int): The number of qubits (must be a power of 2).

    Returns:
        qtn.TensorNetwork: The dynamically generated tensor network.
    """
    assert num_qubits > 1 and (num_qubits & (num_qubits - 1)) == 0, \
        "num_qubits must be a power of 2 and greater than 1"

    # Define the down state tensor
    down_state = np.array([1, 0])

    # Initialize tensors for qubits
    qubit_tensors = [
        qtn.Tensor(data=down_state, inds=(f"i{q}",), tags=(f"b{q}",))
        for q in range(num_qubits)
    ]

    # Generate random unitaries for tensor network connections
    unitaries = []
    num_layers = num_qubits - 1
    for _ in range(num_layers):
        unitaries.append(unitary_group.rvs(2 * len(down_state)).reshape([2] * 4))

    # Build the tensor network dynamically
    tensors = []
    current_indices = [f"i{q}" for q in range(num_qubits)]
    next_index = num_qubits

    for layer, unitary in enumerate(unitaries):
        new_indices = []
        for i in range(0, len(current_indices) - 1, 2):  # Process pairs of indices
            # Assign indices and tags
            inds = (f"i{next_index}", f"i{next_index + 1}", current_indices[i], current_indices[i + 1])
            tags = (f"U{layer}-{i // 2}",)
            tensors.append(qtn.Tensor(data=unitary, inds=inds, tags=tags))
            new_indices.append(f"i{next_index}")
            next_index += 2
        current_indices = new_indices

    # Combine all tensors into the tensor network
    TN = qtn.TensorNetwork(tensors + qubit_tensors)

    return TN


# Example usage
num_qubits = 4
tensor_network = generate_tensor_network(num_qubits)

# Draw the tensor network
tensor_network.draw(show_tags=True, show_inds="all")

# (tensor_network ^ ...).draw(show_inds=True, show_tags=True)
