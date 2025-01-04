import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev
from scipy.stats import unitary_group
from functools import reduce
from itertools import product
import time
import matplotlib.pyplot as plt

# Define helper classes and functions
class SlicedPackage():
	def __init__(self, edge_ind, ind_dim, connected_nodes, sliced_tns):
		self.edge_ind = edge_ind
		self.ind_dim = ind_dim
		self.connected_nodes = connected_nodes
		self.sliced_tns = sliced_tns


class SlicedHelper():
	def __init__(self, edge_ind, ind_dim, connected_nodes):
		self.edge_ind = edge_ind
		self.ind_dim = ind_dim
		self.connected_nodes = connected_nodes



def slice_helper(tn, edge_ind):
    # print(edge_ind)
    sliced_tns = []
    connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
    ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
    for i in range(ind_dim):
        # print(ind_dim)
        stn = tn.copy()
        for cn in connected_nodes:
            # print(stn[cn])
            indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
            index_tuple = tuple(indices)
            new_data = stn[cn].data[index_tuple]
            new_inds = [ind for ind in stn[cn].inds if ind != edge_ind]
            stn[cn].modify(inds=new_inds, data=new_data)
        sliced_tns.append(stn)

            
    # package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))

    return sliced_tns

def slice_at_ind(tn, edge_inds):
    # Initialize the slices with the initial tensor network
    current_slices = [tn]

    # Iterate through each edge index
    for edge_ind in edge_inds:
        next_slices = []
        # Slice each current tensor at the given edge index
        for slice_part in current_slices:
            sliced = slice_helper(slice_part, edge_ind)
            next_slices.extend(sliced)
        # Update current slices to be processed in the next round
        current_slices = next_slices

    # The final result will be the fully sliced tensors
    return current_slices


def obtain_all_prod_labels(d, N):
    return list(product(range(d), repeat=N))

def ket(label, d):
    def basis(d, i):
        return np.eye(1, d, i)
    return reduce(np.kron, [basis(d, i) for i in label])[0]


def encode(tn, edge_inds, x):
    package_list = []
    for edge_ind in edge_inds:
        e_tn = tn.copy()
        connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
        ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
        for cn in connected_nodes:
            e_data = []
            node = tn[cn]
            for i in range(ind_dim):
                indices = [slice(None) if j != node.inds.index(edge_ind) else i for j in range(len(node.inds))]
                index_tuple = tuple(indices)
                new_data = node.data[index_tuple]
                encoded = np.array([c * Chebyshev.basis(i + 1)(x) for c in new_data.ravel()], dtype=object).reshape(new_data.shape)
                e_data.append(encoded)
            new_inds = [ind for ind in node.inds if ind != edge_ind]
            e_tn[cn].modify(inds=new_inds, data=sum(e_data))
    return e_tn ^ ...


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

    down_state = np.array([1, 0])

    # Initialize tensors for qubits
    qubit_tensors = [
        qtn.Tensor(data=down_state, inds=(f"i{q}",), tags=(f"b{q}",))
        for q in range(num_qubits)
    ]

    # Generate random unitaries for tensor network connections
    unitaries = [unitary_group.rvs(4).reshape([2] * 4) for _ in range(num_qubits - 1)]

    # Build the tensor network dynamically
    tensors = []
    current_indices = [f"i{q}" for q in range(num_qubits)]
    next_index = num_qubits

    for layer, unitary in enumerate(unitaries):
        new_indices = []
        for i in range(0, len(current_indices) - 1, 2):  # Process pairs of indices
            inds = (f"i{next_index}", f"i{next_index + 1}", current_indices[i], current_indices[i + 1])
            tags = (f"U{layer}-{i // 2}",)
            tensors.append(qtn.Tensor(data=unitary, inds=inds, tags=tags))
            new_indices.append(f"i{next_index}")
            next_index += 2
        current_indices = new_indices

    return qtn.TensorNetwork(tensors + qubit_tensors)



# we need time difference between slicing and encoding
# we need error difference between slicing and encoding
# we need to inject errors and find when encoding becomes more efficient than slicing


class PerformanceTestPoint():
    def __init__(self, slice_time, encoding_time, slice_error, encoding_error):
        self.slice_time = slice_time
        self.encoding_time = encoding_time
        self.slice_error = slice_error
        self.encoding_error = encoding_error

        self.time_difference = encoding_time - slice_time


def performance_test(TN, N, down_state, d, to_slice):
    # Slice and encode
    # print("Slicing and contracting...")
    start_time = time.time()
    slices_packed = slice_at_ind(TN, to_slice)
    sliced_contracted = 0
    for tn in slices_packed:
        sliced_contracted += (tn ^ ...).data
    slicing_time = time.time() - start_time
            
    # print(sliced_contracted)
    # print((TN ^ ...).data)

    # Gauss-Chebyshev quadrature
    # print("Encoding and contracting...")
    n_points = 4
    x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))
    weights = 2 / n_points

    start_time = time.time()
    y = [weights * encode(TN, to_slice, x).data for x in x_values]
    summed_values = sum(y)
    encoding_time = time.time() - start_time
    
    slicing_error = np.sum(np.abs(sliced_contracted - (TN ^ ...).data))
    encoding_error = np.sum(np.abs(summed_values - (TN ^ ...).data))

    return PerformanceTestPoint(slicing_time, encoding_time, slicing_error, encoding_error)



def time_vs_num_of_slices(TN, N, down_state, d, to_slice):
    perf_and_slices = []
    for i in range(len(to_slice)):
        print(to_slice[:i+1])

        num_tests = 10
        performance_points = []
        for _ in range(num_tests):
            performance_points.append(performance_test(TN, N, down_state, d, to_slice[:i+1]))
        
        # print("Performance test average results:")
        # print("Slice time:", np.mean([point.slice_time for point in performance_points]))
        # print("Encoding time:", np.mean([point.encoding_time for point in performance_points]))
        # print("Slice error:", np.mean([point.slice_error for point in performance_points]))
        # print("Encoding error:", np.mean([point.encoding_error for point in performance_points]))
        # print("Time difference:", np.mean([point.time_difference for point in performance_points]))

        avg_performance = PerformanceTestPoint(
            np.mean([point.slice_time for point in performance_points]),
            np.mean([point.encoding_time for point in performance_points]),
            np.mean([point.slice_error for point in performance_points]),
            np.mean([point.encoding_error for point in performance_points])
        )

        perf_and_slices.append((avg_performance, len(to_slice[:i+1])))

    
    print("Performance test results:")
    for point, num_slices in perf_and_slices:
        print(f"Number of sliced indices: {num_slices}")
        print("Slice time:", point.slice_time)
        print("Encoding time:", point.encoding_time)
        print("Slice error:", point.slice_error)
        print("Encoding error:", point.encoding_error)
        print("Time difference:", point.time_difference)
        print("")


    x = [num_slices for _, num_slices in perf_and_slices]
    y_slice = [point.slice_time for point, _ in perf_and_slices]
    y_encoding = [point.encoding_time for point, _ in perf_and_slices]
    plt.plot(x, y_slice, label="Slicing")
    plt.plot(x, y_encoding, label="Encoding")
    plt.xlabel("Number of slices")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()




def time_vs_num_of_injected_errors(TN, N, down_state, d, to_slice):
    num_tests = 10
    performance_points = []
    for _ in range(num_tests):
        performance_points.append(performance_test(TN, N, down_state, d, to_slice))
    
    

# Main script
if __name__ == "__main__":
    # Define constants
    N = 16
    down_state = np.array([1, 0])
    d = len(down_state)
    to_slice = ['i0', 'i30', 'i36', 'i40']
    # to_slice = ['i0', 'i4', 'i8', 'i12']
    # to_slice = ['i0', 'i4', 'i8', 'i12', 'i16', 'i20', 'i24', 'i28', 'i32', 'i36', 'i40']


    # Generate the tensor network
    print("Generating tensor network...")
    TN = generate_tensor_network(num_qubits=N)
    TN.draw(show_tags=True, show_inds='all')


    # Performance test
    print("Running performance test...")
    time_vs_num_of_slices(TN, N, down_state, d, to_slice)
    





    # # Slice and encode
    # start_time = time.time()
    # to_slice = ['i40', 'i36']
    # slices_packed = slice_at_ind(TN, to_slice)

    # sliced_contracted = 0
    # for tn in slices_packed:
    #     sliced_contracted += (tn ^ ...).data
    
    # slicing_time = time.time() - start_time
    # print(f"Slicing and contracting took {slicing_time:.4f} seconds.")
            
    # # print(sliced_contracted)
    # # print((TN ^ ...).data)

    # # Gauss-Chebyshev quadrature
    # print("Encoding and contracting...")
    # n_points = 3
    # x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))
    # weights = 2 / n_points

    # start_time = time.time()
    # y = [weights * encode(TN, to_slice, x).data for x in x_values]
    # summed_values = sum(y)
    # encoding_time = time.time() - start_time
    # print(f"Encoding and contracting took {encoding_time:.4f} seconds.")

    # # Output results
    # # print("Original contracted value:")
    # # print((TN ^ ...).data)

    # # print("Sum of contracted values:")
    # # print(summed_values)

    # error = np.abs(summed_values - (TN ^ ...).data)
    # # print("Error:")
    # # print(error)
    # print("Sum of error for encoding and total tn:")
    # print(np.sum(error))

    # print("Sum of error for encoding and sliced tn:")
    # print(np.sum(np.abs(summed_values - sliced_contracted)))

    # print("Error for sliced:")
    # print(np.sum(np.abs(sliced_contracted - (TN ^ ...).data)))

    # print("Time difference:")
    # print(encoding_time - slicing_time)

    # # # Generate all product labels and kets
    # # print("Generating all product labels and kets...")
    # # all_prod_labels = obtain_all_prod_labels(d=d, N=N)
    # # all_prod_kets = {label: ket(label, d) for label in all_prod_labels}

    # # # Compute final state projections
    # # state_final = summed_values.reshape(-1)
    # # projections_all_prod = {label: abs(np.dot(all_prod_kets[label].conj(), state_final))**2 for label in all_prod_labels}

    # # print("This sum should be 1:", np.sum(list(projections_all_prod.values())))
