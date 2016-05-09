"""
k-NN: k-nearest neighbours algorithm.

A test sample is classified based on the classification of its K-nearest neighbours.
Notes:
    - The contribution of each neighbour is not weighted. 
    - The class with more ocurrences in the k neighbours classifies the test sample.
"""

__author__ = 'Miguel Rivero'


import operator
import math
import random
from distance import euclidean_distance

def kNN(k, data, instance):
    """
    Returns a list with the k instances in the data set closest to a given instance
    """
    # Extract the real data
    classification = data['data']
    # convert into float to be able to calculate euclidean distance
    for index, li in enumerate(classification):
        aux_list = list(float(x) for x in li[:-1])
        classification[index] = aux_list + [li[-1]]

    # initialize a dict with all votes to 0
    classification_dict = {val: 0 for val in data['attributes'][-1][-1]}
    # add the distance to the instance in a new field
    for i in classification:
        i.append(euclidean_distance(instance, i[:-1]))
    # Sort the classification by the last element
    sorted_classification = sorted(classification, key=operator.itemgetter(-1))

    # we get the value of the K elements with shortest distance
    for x in sorted_classification[:k]:
        classification_dict[x[-2]] += 1

    # Generate final candidates
    candidates_list = list()
    maximum_value = -1
    for key, val in classification_dict.items():
        if val > maximum_value:
            candidates_list = [key]
            maximum_value = val
        elif val == maximum_value:
            candidates_list.append(key)

    return candidates_list