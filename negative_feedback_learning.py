import numpy as np
import sys 
from sklearn.preprocessing import normalize

iterations = int(sys.argv[1])
alpha = float(sys.argv[2])
epsilon1 = None
epsilon2 = None
print(len(sys.argv))
if len(sys.argv) == 5:
    epsilon1 = float(sys.argv[3])
    epsilon2 = float(sys.argv[4])

print(epsilon1)
print(epsilon2)

weights = np.array([[1, 1, 0], [1, 1, 1]])

data_input = np.array([1, 1, 0])

def learning(weights, iterations, data_input, alpha):
    y = np.array([0, 0]) # Need to update this based on the number of output nodes in the question
    for _ in range(iterations):
        e = data_input - np.dot(y, weights)
        y = y + (alpha * np.dot(weights, e))
    return y

value_returned = learning(weights, iterations, data_input, alpha)
print(value_returned)

def learning_stable(weights, iterations, data_input, epsilon1, epsilon2):
    y = np.array([0, 0])
    normalised_weights = normalize(weights, norm="l1")
    for _ in range(iterations):
        divider = max(np.linalg.norm(np.dot(y, weights)), epsilon2)
        if divider != epsilon2:
            divider = np.dot(y, weights)
        e = data_input / divider
        multiplier = max(np.linalg.norm(y), epsilon1)
        if multiplier != epsilon1:
            multiplier = y
        y = multiplier * np.dot(normalised_weights, e)
    return y 

# value_returned_2 = learning_stable(weights, iterations, data_input, epsilon1, epsilon2)
# print(value_returned_2)
