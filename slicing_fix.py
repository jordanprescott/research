import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev
from scipy.stats import unitary_group
from functools import reduce
from itertools import product

# Define helper classes and functions
class SlicedPackage:
    def __init__(self, edge_ind, ind_dim, connected_nodes, sliced_tns):
        self.edge_ind = edge_ind
        self.ind_dim = ind_dim
        self.connected_nodes = connected_nodes
        self.sliced_tns = sliced_tns


def slice_helper(tn, edge_ind):
    print(edge_ind)
    sliced_tns = []
    connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
    ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
    for i in range(ind_dim):
        print(ind_dim)
        stn = tn.copy()
        for cn in connected_nodes:
            print(stn[cn])
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









# data = np.random.rand(1, 2)
data = np.array([[1, 2]])
inds = ('i', 'j')
tags = ('a',)
a = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(2, 2)
data = np.array([[1, 2], [1, 2]])
inds = ('j', 'k')
tags = ('b',)
b = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(2, 2)
data = np.array([[1, 2], [1, 2]])
inds = ('k', 'l')
tags = ('c',)
c = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(2, 1)
data = np.array([[1, 2], [1, 2]])
inds = ('l', 'm')
tags = ('d',)
d = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(2, 1)
data = np.array([[1, 2], [1, 2]])
inds = ('m', 'n')
tags = ('e',)
e = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(2, 1)
data = np.array([[1], [2]])
inds = ('n', 'o')
tags = ('f',)
f = qtn.Tensor(data=data, inds=inds, tags=tags)



# TN = a & b
TN = a & b & c & d & e & f
# TN.draw(show_tags=True, show_inds='all')


# edges_to_slice_over = ['j']
edges_to_slice_over = ['j', 'l', 'n']



# slice non-adjacent edges
slices_packed = slice_at_ind(TN, edges_to_slice_over)
# for pack in slices_packed:
# 	print(pack.sliced_tns)


slice_sum = 0
for tn in slices_packed:
    # tn.draw(show_tags=True, show_inds='all')
    slice_sum += (tn ^ ...).data
    print((tn ^ ...).data)

print("sum of sliced contracted values:")
print(slice_sum)

print("Original contracted value:")
print((TN ^ ...).data)











# def slice_at_ind(tn, edge_inds):
#     package_list = []
#     for edge_ind in edge_inds:
#         sliced_tns = []
#         connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
#         ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
#         for i in range(ind_dim):
#             stn = tn.copy()
#             for cn in connected_nodes:
#                 indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
#                 index_tuple = tuple(indices)
#                 new_data = stn[cn].data[index_tuple]
#                 new_inds = [ind for ind in stn[cn].inds if ind != edge_ind]
#                 stn[cn].modify(inds=new_inds, data=new_data)
#             sliced_tns.append(stn)
#         package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))
#     return package_list