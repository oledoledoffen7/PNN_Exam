import numpy as np

weights1 = np.array([-1, 5])
weights2 = np.array([2, -3])
data = np.array([(1, 2), (2, 1), (3, 3), (6, 5), (7, 8)])
data_labels = [1, 1, 1, 2, 2]

def fisher_method(weights, data, data_labels):
    first_class_sum = 0
    second_class_sum = 0
    number_first_class = 0
    number_second_class = 0
    for i in range(len(data_labels)):
        if data_labels[i] == 1:
            first_class_sum = first_class_sum + data[i] 
            number_first_class = number_first_class + 1
        else:
            second_class_sum = second_class_sum + data[i]
            number_second_class = number_second_class + 1
    m1 = (1 / number_first_class) * first_class_sum
    m2 = (1 / number_second_class) * second_class_sum
    print(m1)
    print(m2)
    sb = (np.dot(weights, (m1 - m2)))**2 
    s1_squared = 0
    s2_squared = 0
    for i in range(len(data_labels)):
        if data_labels[i] == 1:
            s1_squared = s1_squared + (np.dot(weights, (data[i] - m1)))**2
        else:
            s2_squared = s2_squared + (np.dot(weights, (data[i] - m2)))**2
    sw = s1_squared + s2_squared
    print(sb)
    print(sw)
    j = sb / sw
    return j

result1 = fisher_method(weights1, data, data_labels)
result2 = fisher_method(weights2, data, data_labels)
print(result1)
print(result2)