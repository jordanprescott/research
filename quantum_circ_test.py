import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev
from scipy.stats import unitary_group
from functools import reduce
from itertools import product

class SlicedPackage:
    def __init__(self, edge_ind, ind_dim, connected_nodes, sliced_tns):
        self.edge_ind = edge_ind
        self.ind_dim = ind_dim
        self.connected_nodes = connected_nodes
        self.sliced_tns = sliced_tns

def slice_at_ind(tn, edge_inds):
    package_list = []
    for edge_ind in edge_inds:
        sliced_tns = []
        connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
        ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
        for i in range(ind_dim):
            stn = tn.copy()
            for cn in connected_nodes:
                indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
                index_tuple = tuple(indices)
                new_data = stn[cn].data[index_tuple]
                new_inds = [ind for ind in stn[cn].inds if ind != edge_ind]
                stn[cn].modify(inds=new_inds, data=new_data)
            sliced_tns.append(stn)
        package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))
    return package_list

def obtain_all_prod_labels(d, N=4):
    return list(product(range(d), repeat=N))

def ket(label, d):
    def basis(d, i):
        return np.eye(1, d, i)
    return reduce(np.kron, [basis(d, i) for i in label])[0]

def encode(package_list, x):
    for sliced_package in package_list:
        e_tn = sliced_package.sliced_tns[0].copy()
        for cn in sliced_package.connected_nodes:
            encoded = 0
            for i, stn in enumerate(sliced_package.sliced_tns):
                data_shape = stn[cn].data.shape
                encoded += np.array([c * Chebyshev.basis(i + 1)(x) for c in stn[cn].data.ravel()], dtype=object).reshape(data_shape)
            e_tn[cn].modify(data=encoded)
    return e_tn ^ ...

# Define down and up states
down_state = np.array([1, 0])
d = len(down_state)
interaction_dim = 2 * d

# Qubits
N=16

# Generate all product labels and kets
all_prod_labels = obtain_all_prod_labels(d=d, N=N)
all_prod_kets = {label: ket(label, d) for label in all_prod_labels}




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











TN = generate_tensor_network(num_qubits=N)


TN.draw(show_tags=True, show_inds='all')
# (TN ^ ...).draw(show_tags=True, show_inds='all')


# Slice and encode
to_slice = ['i6', 'i0']
slices_packed = slice_at_ind(TN, to_slice)

# Gauss-Chebyshev quadrature
n_points = 150
x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))
weights = 2 / n_points

y = [weights * encode(slices_packed, x).data for x in x_values]
summed_values = sum(y)


# Output results
print("Original contracted value:")
print((TN ^ ...).data)

print("Sum of contracted values:")
print(summed_values)

error = np.abs(summed_values - (TN ^ ...).data)
print("Error:")
print(error)
print("Sum of error:")
print(np.sum(error))

state_final = summed_values.reshape(-1)

# print(all_prod_kets)


print()

projections_all_prod = {label: abs(np.dot(all_prod_kets[label].conj(), state_final))**2 for label in all_prod_labels}
print("This sum should be 1:", np.sum(list(projections_all_prod.values())))
