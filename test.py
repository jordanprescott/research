import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev

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
data = np.random.rand(1, 2)
inds = ('i', 'j')
tags = ('a',)
a = qtn.Tensor(data=data, inds=inds, tags=tags)

data = np.random.rand(2, 3)
inds = ('j', 'k')
tags = ('b',)
b = qtn.Tensor(data=data, inds=inds, tags=tags)

data = np.random.rand(3, 1)
inds = ('k', 'l')
tags = ('c',)
c = qtn.Tensor(data=data, inds=inds, tags=tags)


# data = np.random.rand(1, 2)
# inds = ('i', 'j')
# tags = ('a',)
# a = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.random.rand(2, 3)
# inds = ('j', 'k')
# tags = ('b',)
# b = qtn.Tensor(data=data, inds=inds, tags=tags)


# data = np.array([[1, 2],
# 				[1, 2]])
# inds = ('i', 'j')
# tags = ('a',)
# a = qtn.Tensor(data=data, inds=inds, tags=tags)

# data = np.array([[3], [4]])
# inds = ('j', 'k')
# tags = ('b',)
# b = qtn.Tensor(data=data, inds=inds, tags=tags)


# TN = a & b
TN = a & b & c
# TN.draw(show_tags=True, show_inds='all')


# edges_to_slice_over = ['k', 'm']
edges_to_slice_over = ['j']
# edges_to_slice_over = ['m']



# slice non-adjacent edges
slices_packed = slice_at_ind(TN, edges_to_slice_over)


inddd = 0
for pack in slices_packed:
	print("pack")
	print(pack.sliced_tns)
	print(pack.connected_nodes)
	for tn in pack.sliced_tns:
		inddd+=1
		print("tn")
		print(tn)
		# tn.draw(show_tags=True, show_inds='all')
		# (tn ^ ...).draw(show_tags=True, show_inds='all')
		for tensor in tn:
			print("tensor")
			print(tensor.tags)
			print(tensor.inds)
			print(tensor.data)
			print(tensor.data.shape)
			poly = np.array([c * Chebyshev.basis(inddd)(1) for c in tensor.data.ravel()], dtype=object).reshape(tensor.data.shape)
			print(poly)
			# print(poly(1))

			# sliced_package.sliced_tns[j][cn].data


class EncodedPackage:
    def __init__(self, tag, poly):
        self.tag = tag
        self.poly = poly


class TestThing:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def encode(packed, x):
	encoded_list = []
	test_list = []
	i = 0

	# Iterate through each package of sliced tensor networks
	for pack in packed:
		print(pack.sliced_tns)

		# For each sliced tensor network in the package
		for tn in pack.sliced_tns:
			print(tn)
			i += 1
			print(f'i: {i}')

			# For each tensor in the sliced tensor network
			for tensor in tn:
				# Create polynomials for each tensor's data
				poly = np.array([c * Chebyshev.basis(i) for c in tensor.data.ravel()], dtype=object).reshape(tensor.data.shape)
				# Create and append an EncodedPackage object
				encoded_list.append(EncodedPackage(tensor.tags, poly))

				# Evaluate the polynomial at x
				eval_node = np.array([c * Chebyshev.basis(i)(x) for c in tensor.data.ravel()], dtype=object).reshape(tensor.data.shape)
				# Modify the tensor's data in place
				tensor.modify(data=poly)
				print(tensor.data)

			# Create and append a TestThing object with the contracted tensor network
			y = tn ^ ...
			test_list.append(y)


	return encoded_list, test_list







test, test2 = encode(slices_packed, 1)
# for t in test:
# 	print(t.tag)
# 	print(t.poly)

for y in test2:
	print("test2")
	print(y)





def master_node(encoded_list):

	# Sum polynomials by tags using the EncodedPackage objects
	polynomial_sum_by_tag = {}
	for package in encoded_list:
		tag = frozenset(package.tag)  # Convert oset to frozenset for use as a dictionary key
		poly = package.poly

		if tag not in polynomial_sum_by_tag:
			polynomial_sum_by_tag[tag] = poly.copy()
		else:
			polynomial_sum_by_tag[tag] += poly.copy()

	# Display the results
	for tag, poly in polynomial_sum_by_tag.items():
		print(f"Tag '{set(tag)}': {poly}")  # Convert frozenset back to a regular set for display

	return polynomial_sum_by_tag


polynomial_sum_by_tag = master_node(test.copy())

# Extract the polynomials
polynomial_a = polynomial_sum_by_tag[frozenset({'a'})]
polynomial_b = polynomial_sum_by_tag[frozenset({'b'})]

# add new axis to fix dimensions
polynomial_a = polynomial_a[np.newaxis, :]
polynomial_b = polynomial_b[np.newaxis, :]

# Compute the product
product_polynomial = np.transpose(polynomial_a) * polynomial_b

print(f"Product of polynomials: {product_polynomial}")
print(product_polynomial.shape)


# Gauss-Chebyshev quadrature points and weights
n_points = 100  # Number of points for higher accuracy
x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))  # Chebyshev nodes
weights = 2 / n_points  # Each point has the same weight in Gauss-Chebyshev quadrature

# Calculate the integral using Gauss-Chebyshev quadrature

result = []
for idx, f in np.ndenumerate(product_polynomial):
	integral = sum(weights * f(x) for x in x_values)
	result.append(integral)
result = np.array(result).reshape(product_polynomial.shape)
print(f"Numerical integral using Gauss-Chebyshev quadrature:\n {result}")

actual = (TN ^ ...).data
print("original contracted value:")
print(actual)

print("error:")
print(np.abs(result - actual))