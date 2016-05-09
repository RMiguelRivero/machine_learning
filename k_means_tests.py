"""
Example use of the k mean algorithm
"""

__author__ = 'Miguel Rivero'


from k_means_algorithm import *
import arff   # pip install liac-arff or easy_install liac-arff
from collections import defaultdict


population_weight = {
    'description': 'Weight from 2 groups: male and female',
    'relation': 'Weight',
    'attributes': [
        ('Weight', 'REAL'),
    ],
    'data': [
        [51.0], [43.0], [62.0], [64.0], [45.0], [42.0], [46.0], [45.0], [45.0], [62.0], [47.0],
        [52.0], [64.0], [51.0], [65.0], [48.0], [49.0], [46.0], [64.0], [51.0], [52.0], [62.0],
        [49.0], [48.0], [62.0], [43.0], [40.0], [48.0], [64.0], [51.0], [63.0], [43.0], [65.0],
        [66.0], [65.0], [46.0], [39.0], [62.0], [64.0], [52.0], [63.0], [64.0], [48.0], [64.0],
        [48.0], [51.0], [48.0], [64.0], [42.0], [48.0], [41.0]
    ]
}


sample = [64.0]
centroids = [[39.0], [45.0]]
print('sample: '+  str(sample))
print('centroids: '+  str(centroids))
index = calculates_nearest_centroid(sample, centroids, abs_distance)
print("Closest centroid index: " + str(index))
print("Closest centroid value: " + str(centroids[index]))

print("=" * 80)
clas = empty_initial_classification(population_weight['data'])
centr = [[65.0], [48.0]]
assign_cluster(clas, centr, abs_distance)
print(str(clas))
print("Reassign centroids ", reassign_centroids(clas, 2))

classified_data = k_means(2, population_weight['data'], abs_distance)
classification = defaultdict(list)
for val, cl in classified_data[1]:
    classification[cl].append(val)
print('Population weight classification')
print('================================')
print(classification)



# Experimentation with data from "iris.arff"
# https://pypi.python.org/pypi/liac-arff/2.0.1

iris = arff.load(open("./iris.arff"))


def validation_iris(classification):
    possible_values = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    counters = dict()
    for val in possible_values:
        for x in range(3):
            counters[val, x] = 0
    for i in range(len(classification)):
        counters[iris['data'][i][-1], classification[i][1]] += 1
    for val in possible_values:
        print(val + "\n" + "=" * len(val))
        for x in range(3):
            print("Cluster ", x, ": ", counters[val, x])
        print("\n")


print("=" * 80)
iris_without_classification = [x[:-1] for x in iris['data']]
iri_k_means = k_means(3, iris_without_classification, euclidean_distance)[1]
validation_iris(iri_k_means)


# Experimentation with data from cardiac arrhythmias

arrhythmia = arff.load(open("./arrhythmia.arff"))


def validation(k, data, distance):
    """
    Validation for K-mean algorithm
    """
    # declassify data removing last element from data
    data_without_classification = [x[:-1] for x in data['data']]

    for index, lista in enumerate(data_without_classification):
        data_without_classification[index] = list(float(x) for x in lista)
    # apply k-means algorithm
    k_means_classification = k_means(k, data_without_classification, distance)[1]
    possible_values = [x for x in data['attributes'][-1][-1]]
    counters = dict()
    for val in possible_values:
        for x in range(len(possible_values)):
            counters[val, x] = 0
    for i in range(len(k_means_classification)):
        counters[data['data'][i][-1], k_means_classification[i][1]] += 1
    # do some printing statement so we can understand better the results
    for val in possible_values:
        print(val+"\n"+"="*len(val)+"\n")
        for x in range(len(possible_values)):
            print("Cluster ", x, ": ", counters[val, x])
        print("\n")


print("=" * 80)
print("Iris")
validation(3, iris, euclidean_distance)
print("\n" + "=" * 80)
print("Arrhythmia")
validation(16, arrhythmia, euclidean_distance)

