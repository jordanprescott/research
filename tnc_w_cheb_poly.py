import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev




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



def encode(tn, edge_inds, x):
	package_list = []
	for edge_ind in edge_inds:
		sliced_tns = []
		e_tn = tn.copy()
		connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
		ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
		# for i in range(ind_dim):
		for cn in connected_nodes:
			# print(i)
			# for cn in connected_nodes:
			e_data = []
			e_poly = []
			for i in range(ind_dim):
				stn = tn.copy()
				indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
				index_tuple = tuple(indices)
				new_data = stn[cn].data[index_tuple]

				poly = np.array([c * Chebyshev.basis(i + 1) for c in new_data.ravel()], dtype=object).reshape(new_data.shape)
				e_poly.append(poly)
				# print(poly)
				
				encoded = np.array([c * Chebyshev.basis(i + 1)(x) for c in new_data.ravel()], dtype=object).reshape(new_data.shape)
				e_data.append(encoded)
				# print(encoded)


			
			sliced_tns.append(stn)

			new_inds = [ind for ind in stn[cn].inds if ind != edge_ind]
			e_tn[cn].modify(inds=new_inds, data=sum(e_data))

			

		package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))
	return e_tn ^ ...


# steps: get network, select edge, slice, encode, contract w/ diff eval points, interpolate

# get network
# data = np.random.rand(2, 4)
# inds = ('i', 'j')
# tags = ('a',)
# a = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(4, 3)
# inds = ('j', 'k')
# tags = ('b',)
# b = qtn.Tensor(data=data, inds=inds, tags=tags)




data = np.random.rand(1, 2)
inds = ('i', 'j')
tags = ('a',)
a = qtn.Tensor(data=data, inds=inds, tags=tags)

data = np.random.rand(2, 2)
inds = ('j', 'k')
tags = ('b',)
b = qtn.Tensor(data=data, inds=inds, tags=tags)

data = np.random.rand(2, 2)
inds = ('k', 'l')
tags = ('c',)
c = qtn.Tensor(data=data, inds=inds, tags=tags)

data = np.random.rand(2, 50)
inds = ('l', 'm')
tags = ('d',)
d = qtn.Tensor(data=data, inds=inds, tags=tags)


data = np.random.rand(50, 2)
inds = ('m', 'n')
tags = ('e',)
e = qtn.Tensor(data=data, inds=inds, tags=tags)

data = np.random.rand(2, 2)
inds = ('n', 'o')
tags = ('f',)
f = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.array([[1, 2], [3, 4]])
# inds = ('i', 'j')
# tags = ('a',)
# a = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.array([[3], [4]])
# inds = ('j', 'k')
# tags = ('b',)
# b = qtn.Tensor(data=data, inds=inds, tags=tags)




# TN = a & b
TN = a & b & c & d & e & f
TN.draw(show_tags=True, show_inds='all')



# edges_to_slice_over = ['k', 'm']
# edges_to_slice_over = ['j']
edges_to_slice_over = ['j', 'l', 'n']
# edges_to_slice_over = ['m']



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
		





# # slice non-adjacent edges
# for pack in slices_packed:
# 	# print(pack.sliced_tns)
# 	for tn in pack.sliced_tns:
# 		# print(tn)
# 		for tensor in tn:
# 			print(tensor.data)
# 			# sliced_package.sliced_tns[j][cn].data



# Gauss-Chebyshev quadrature points and weights
n_points = 150  # Number of points for higher accuracy
x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))  # Chebyshev nodes
weights = 2 / n_points  # Each point has the same weight in Gauss-Chebyshev quadrature


print("original contracted value:")
print((TN ^ ...).data)


y = []
# x_values = x_values[[0]]
for x in x_values:
	y.append(weights * (encode(TN, edges_to_slice_over, x)).data)
print("sum of encoded contractions:")
print(sum(y))


# method to get polynomial from tensor network and evaluate at x

# print("encoded value:")

# f = encode(slices_packed, 0)
# print(f)
# integral = sum(weights * f.data[0][0](x) for x in x_values)

# print("sum of contracted values:")
# print(integral)


print()
print("error:")
print(np.abs(sum(y) - (TN ^ ...).data))
print("sum of error:")
print(np.sum(np.abs(sum(y) - (TN ^ ...).data)))




