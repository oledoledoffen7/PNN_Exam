import numpy as np
import sys

algorithm_type = sys.argv[1] # The type of algorithm (sequential or batch)
weight_vector = np.array([-1.5, 2])
data = np.array([(1, 0), (1, 1)])
data_labels = [1, 0]
learning_rate = 1

def heaviside(input):
    if 1.0e-09 > input > -1.0e-09:
        # Input is zero
        return 0.5
    elif input > 1.0e-09:
        # Input is positive
        return 1
    else:
        # Input is negative
        return 0

def delta_learning(weight, data, algorithm_type):
    weight_vector_changed = True
    while weight_vector_changed == True:
        if algorithm_type == "sequential":
            counter = 0
            for i in range(len(data)):
                prediction = heaviside(np.dot(weight, data[i]))
                if prediction != data_labels[i]:
                    weight = weight + (learning_rate * (data_labels[i] - prediction) * data[i])
                else:
                    counter = counter + 1
            if counter == len(data):
                weight_vector_changed = False
        elif algorithm_type == "batch":
            predictions = []
            for i in range(len(data)):
                prediction = heaviside(np.dot(weight, data[i]))
                predictions.append(prediction)
            change_sum = 0
            counter = 0
            for i in range(len(predictions)):
                if data_labels[i] - predictions[i] == 0:
                    counter = counter + 1
                change_sum = change_sum + ((data_labels[i] - predictions[i]) * data[i])
            if counter == len(data):
                weight_vector_changed = False
            else:
                weight = weight + learning_rate * change_sum
    return weight            

weight_to_print = delta_learning(weight_vector, data, algorithm_type)
print(weight_to_print)