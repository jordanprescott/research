import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev

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

class EncodedPackage:
    def __init__(self, tag, poly):
        self.tag = tag
        self.poly = poly

def encode(packed):
    encoded_list = []
    i = 0
    for pack in packed:
        for tn in pack.sliced_tns:
            i += 1
            for tensor in tn:
                poly = np.array([c * Chebyshev.basis(i) for c in tensor.data.ravel()], dtype=object).reshape(tensor.data.shape)
                encoded_list.append(EncodedPackage(tensor.tags, poly))
    return encoded_list

def master_node(encoded_list):
    polynomial_sum_by_tag = {}
    for package in encoded_list:
        tag = frozenset(package.tag)
        poly = package.poly
        if tag not in polynomial_sum_by_tag:
            polynomial_sum_by_tag[tag] = poly.copy()
        else:
            polynomial_sum_by_tag[tag] += poly.copy()
    for tag, poly in polynomial_sum_by_tag.items():
        print(f"Tag '{set(tag)}': {poly}")
    return polynomial_sum_by_tag

def f(x, product_polynomial):
    return product_polynomial(x)

# Tensor network creation
data = np.random.rand(1, 20)
a = qtn.Tensor(data=data, inds=('i', 'j'), tags=('a',))
data = np.random.rand(20, 1)
b = qtn.Tensor(data=data, inds=('j', 'k'), tags=('b',))
TN = a & b

# Slice tensor network
edges_to_slice_over = ['j']
slices_packed = slice_at_ind(TN, edges_to_slice_over)

# Encode sliced tensor networks
encoded_list = encode(slices_packed)

# Process encoded packages
polynomial_sum_by_tag = master_node(encoded_list)

# Compute polynomial product
polynomial_a = polynomial_sum_by_tag[frozenset({'a'})]
polynomial_b = polynomial_sum_by_tag[frozenset({'b'})]
product_polynomial = (polynomial_a * polynomial_b)[0]

# Numerical integration using Gauss-Chebyshev quadrature
n_points = 100
x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))
weights = 2 / n_points
integral = sum(weights * f(x, product_polynomial) for x in x_values)
print(f"Numerical integral using Gauss-Chebyshev quadrature: {integral}")

# Compare with original contraction
print("Original contracted value:")
print((TN ^ ...).data)
