from math import sqrt
import numpy as np

sparse_alternative_1 = np.array([1, 0, 0, 0, 1, 0, 0, 0])
sparse_alternative_2 = np.array([0, 0, 1, 0, 0, 0, -1, 0])
sparse_alternative_3 = np.array([0, 0, 0, -1, 0, 0, 0, 0])
dictionary = np.array([(0.4, -0.6), (0.55, -0.45), (0.5, -0.5), (-0.1, 0.9), (-0.5, -0.5), (0.9, 0.1), (0.5, 0.5), (0.45, 0.55)])
data = np.array([-0.05, -0.95])

def sparse_coding_method(sparse_alternative, dictionary, data):
    product = np.dot(sparse_alternative, dictionary)
    input = data - product 
    l2_norm = sqrt(input[0]**2 + input[1]**2)
    return l2_norm

result = sparse_coding_method(sparse_alternative_2, dictionary, data)
print(result)