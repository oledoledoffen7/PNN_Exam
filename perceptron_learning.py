import numpy as np
import sys

if len(sys.argv) == 3:
    algorithm_type = sys.argv[1] # The type of algorithm (sequential, batch or multi_class)
    number_of_classes = sys.argv[2] # The number of classes in the multi_class case
else:
    algorithm_type = sys.argv[1] # The type of algorithm (sequential, batch or multi_class)

########################################################
# These need to be changed based on the question given. Also need to add multiple weight vectors in case
# it is a multi-class perceptron learning algorithm question.

weight_vector = np.array([1, 0.5, 0.5, -0.75])
weight_vector_2 = np.array([-1, 2, 2, 1])
weight_vector_3 = np.array([2, -1, -1, 1])
weight_vectors = []
weight_vectors.append(weight_vector)
weight_vectors.append(weight_vector_2)
weight_vectors.append(weight_vector_3)

data = np.array([(0, 1, 0), (1, 0, 0), (0.5, 0.5, 0.25), (1, 1, 1), (0, 0, 0)])
data_labels = [1, 1, 2, 2, 3]
learning_rate = 1

########################################################

def augmentation(data_to_be_augmented):
    augmented_data = []
    for datapoint in data_to_be_augmented:
        augmented_datapoint = np.insert(datapoint, 0, 1)
        augmented_data.append(augmented_datapoint)
    return augmented_data

augmented_data = augmentation(data)

def normalised(data_to_be_normalised):
    for i in range(len(data_to_be_normalised)):
        if data_labels[i] == 1: # Only needs to be changed if one of the labels in the question is not 1
            continue
        else:
            data_to_be_normalised[i] = data_to_be_normalised[i] * -1
    return data_to_be_normalised

if algorithm_type == "multi_class":
    normalised_data = augmented_data
else:
    normalised_data = normalised(augmented_data)

def perceptron_learning(algorithm_type, data_to_use, weight_vector):
    if algorithm_type == "sequential":
        converged = False
        while converged == False:
            number_of_true_predictions = 0
            for datapoint in data_to_use:
                prediction = np.dot(datapoint, weight_vector)
                if prediction > 0:
                    number_of_true_predictions = number_of_true_predictions + 1
                else:
                    weight_vector = weight_vector + (learning_rate * datapoint)
            if number_of_true_predictions == len(data_to_use):
                converged = True
        return weight_vector
    elif algorithm_type == "batch":
        converged = False
        while converged == False:
            number_of_true_predictions = 0
            wrongly_predicted_datapoints = []
            for datapoint in data_to_use:
                prediction = np.dot(datapoint, weight_vector)
                if prediction > 0:
                    number_of_true_predictions = number_of_true_predictions + 1
                else:
                    wrongly_predicted_datapoints.append(datapoint)
            for wrong_prediction in wrongly_predicted_datapoints:
                weight_vector = weight_vector + (learning_rate * wrong_prediction)
            if number_of_true_predictions == len(data_to_use):
                converged = True 
        return weight_vector
    elif algorithm_type == "multi_class":
        converged = False
        while converged == False:
            number_of_true_predictions = 0
            for i in range(len(data_to_use)):
                predictions = []
                for j in range(len(weight_vectors)):
                    prediction = np.dot(data_to_use[i], weight_vectors[j])
                    predictions.append((prediction, j + 1))
                predicted_class = max(predictions, key=lambda i: i[0])[1]
                if all(x[0] == predictions[0][0] for x in predictions):
                    # In the event of ties, these lines may have to be changed
                    predicted_class = 1
                    # predicted_class = len(weight_vectors)
                target_class = data_labels[i]
                if target_class != predicted_class:
                    weight_vectors[target_class-1] = weight_vectors[target_class-1] + (learning_rate * data_to_use[i])
                    weight_vectors[predicted_class - 1] = weight_vectors[predicted_class - 1] - (learning_rate * data_to_use[i])
                else:
                    number_of_true_predictions = number_of_true_predictions + 1
            if number_of_true_predictions == len(data_to_use):
                converged = True
        return weight_vectors

resulting_weight_vector = perceptron_learning(algorithm_type, normalised_data, weight_vector)
print(resulting_weight_vector)


