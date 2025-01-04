import numpy as np
import quimb.tensor as qtn
import scipy
from scipy.interpolate import CubicSpline


class SlicedPackage():
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
		# print(connected_nodes)
		ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
		for i in range(ind_dim):
			stn = tn.copy()
			for cn in connected_nodes:

				indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
				index_tuple = tuple(indices)
				new_data = stn[cn].data[index_tuple]
				# new_data = new_data[:, np.newaxis]
				new_inds = [i for i in stn[cn].inds if i != edge_ind]

				# print(new_inds)
				# print(new_data)
				# print(np.shape(new_data))
				stn[cn].modify(inds=new_inds, data=new_data)

			sliced_tns.append(stn)
		package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))

	return package_list


# steps: get network, select edge, slice, encode, contract w/ diff eval points, interpolate

# get network
# data = 5 * np.random.rand(2, 2, 4)
# inds = ('i', 'j', 'm')
# tags = ('a',)
# a = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = 4 * np.random.rand(2, 4)
# inds = ('j', 'k')
# tags = ('b',)
# b = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = 3 * np.random.rand(4,)
# inds = ('k',)
# tags = ('c',)
# c = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = 2 * np.random.rand(4,)
# inds = ('m',)
# tags = ('d',)
# d = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = 2 * np.random.rand(2,)
# inds = ('i',)
# tags = ('e',)
# e = qtn.Tensor(data=data, inds=inds, tags=tags)





data = 5 * np.random.rand(1, 2)
inds = ('i', 'j',)
tags = ('a',)
a = qtn.Tensor(data=data, inds=inds, tags=tags)

data = 4 * np.random.rand(2, 3)
inds = ('j', 'k',)
tags = ('b',)
b = qtn.Tensor(data=data, inds=inds, tags=tags)

data = 3 * np.random.rand(3,2)
inds = ('k', 'l',)
tags = ('c',)
c = qtn.Tensor(data=data, inds=inds, tags=tags)

data = 2 * np.random.rand(2,1)
inds = ('l','m',)
tags = ('d',)
d = qtn.Tensor(data=data, inds=inds, tags=tags)



TN = a & b & c & d
TN.draw(show_tags=True, show_inds='all')


edges_to_slice_over = ['j', 'l']
# edges_to_slice_over = ['j']
# edges_to_slice_over = ['m']



# slice non-adjacent edges
slices_packed = slice_at_ind(TN, edges_to_slice_over)
for pack in slices_packed:
	print(pack.sliced_tns)
print(slices_packed)

# for n in slices:
# 	# n.draw()
# 	print((n ^ ...).data)




# get coefficients
def encode(package_list, x):
	L_k = [1] + [package.ind_dim for package in package_list]
	k = 0
	deg = 0.0
	degree = []
	for sliced_package in package_list:

		e_tn = sliced_package.sliced_tns[0].copy()
		dir_indicator = 0

		for cn in sliced_package.connected_nodes:
			encoded = 0
			for j in range(sliced_package.ind_dim):
				if dir_indicator % 2 == 0:
					encoded += sliced_package.sliced_tns[j][cn].data * x**(j * np.prod(L_k[0:k+1]))
					# print((j * np.prod(L_k[0:k+1])))
				elif dir_indicator % 2 == 1:
					encoded += sliced_package.sliced_tns[j][cn].data * x ** ((sliced_package.ind_dim - 1 - j) * np.prod(L_k[0:k+1]))
					# print(((sliced_package.ind_dim - 1 - j) * np.prod(L_k[0:k+1])))

				if (j * np.prod(L_k[0:k+1])) > deg:
					deg = (j * np.prod(L_k[0:k+1]))



			# print()
			# print(cn)
			# print(encoded)
			# print()
			dir_indicator += 1
			# all_encoded.append(tuple([cn, encoded]))
			e_tn[cn].modify(data=encoded)
			# all_encoded.append(encoded)

		degree.append(deg)
		k += 1

	# print('degree:')
	# print(2 * np.sum(degree))

	poly_degree = 2.0 * np.sum(degree)

	# encoded_tn.draw()
	# return all_encoded
	return e_tn ^ ..., poly_degree


# encoded_tn_at_x = encode(slices, edge_to_slice_over, nodes_to_encode, ind_dimension, 2)
# encoded_tn_at_1 = encode(slices_packed, 2)





def chebyshev_points(n):
	"""
	Generate n Chebyshev points in the interval [-1, 1].

	Parameters:
	n (int): The number of Chebyshev points to generate.

	Returns:
	np.ndarray: Array of Chebyshev points.
	"""
	i = np.arange(1, n+1)
	chebyshev_nodes = np.cos((2*i - 1) * np.pi / (2 * n))
	return chebyshev_nodes













# x = []
y = []
print("poly degree: ")

_, poly_degree = encode(slices_packed, 1)


print(poly_degree)

x = chebyshev_points(int(poly_degree)+1)
for i in range(int(poly_degree)+1):
	encoded_tn = encode(slices_packed, x[i])
	y.append(encoded_tn[0].data)

y = np.array(y)


print()
print("contracted value for original network: ")
print((TN ^ ...).data)
print()



# Number of data points (degree of polynomial will be n-1)
n = len(x)

# Create the Vandermonde matrix
vander_matrix = np.vander(x, n, increasing=False)



# Solve the linear system to find the coefficients
coefficients = np.linalg.solve(vander_matrix, y)


print("predicted coeff from vandermonde interpolation:")
print(coefficients[18])
print()

condition_number = np.linalg.cond(vander_matrix)
print(f"Condition number: {condition_number}")





n_features = y.shape[1]
n_points = len(x)

