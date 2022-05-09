import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

# data = np.array([(0, 0), (1, 0), (2, 1), (0, 1), (1, 2)])
data = np.array([(0, 2), (1, 2), (2, 1), (-3, 1), (-2, -1), (-3, -2)])
# data = iris.data
# data_labels = [1, 1, 1, -1, -1]
data_labels = [1, 1, 1, -1, -1, -1]
# data_labels = iris.target
# biases = [2, 2, 2, 2, 2]
biases = [1, 1, 1, 1, 1, 1]
# biases = []
# for i in range(len(data)):
#     biases.append(1)
# weight_vector = np.array([-1.5, 5, -1])
weight_vector = np.array([1, 0, 0])
# weight_vector = np.array([0.5, -0.5, -1.5, 2.5, -1.5])
learning_rate = 0.1
# learning_rate = 0.2
number_of_iterations = 12

def augmentation(data_to_be_augmented):
    augmented_data = []
    for datapoint in data_to_be_augmented:
        augmented_datapoint = np.insert(datapoint, 0, 1)
        augmented_data.append(augmented_datapoint)
    return augmented_data

augmented_data = augmentation(data)

def normalised(data_to_be_normalised):
    for i in range(len(data_to_be_normalised)):
        if data_labels[i] == 1: # Potentially needs to be changed depending on the linear discriminant function given
            continue
        else:
            data_to_be_normalised[i] = data_to_be_normalised[i] * -1
    return data_to_be_normalised

normalised_data = normalised(augmented_data)

def Widrow_Hoff_Algo(data_to_run, weights, iterations):
    iterations_completed = 0
    current_datapoint = 0
    while iterations_completed < iterations:
        print(current_datapoint)
        updated_weight = weights - (learning_rate * (np.dot(weights, data_to_run[current_datapoint]) - biases[current_datapoint]) * data_to_run[current_datapoint])
        weights = updated_weight
        current_datapoint = current_datapoint + 1
        if current_datapoint == len(data_to_run):
            current_datapoint = 0
        iterations_completed = iterations_completed + 1
    return weights 

updated_weights = Widrow_Hoff_Algo(normalised_data, weight_vector, number_of_iterations)
print(updated_weights)
