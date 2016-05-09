"""
k-means algorithm.

k-Means Clustering is a partitioning method which partitions data into k mutually exclusive clusters.
It returns the index of the cluster to which it has assigned each observation.
"""

__author__ = 'Miguel Rivero'


import operator
import math
import random
from distance import *


def empty_initial_classification(datos):
    """
    Generate empty list this way [[x, None], [y, None],...]
    >>> empty_initial_classification(population_weight['data'])
    [[[51.0], None], [[43.0], None], [[62.0], None], [[64.0], None],
    [[45.0], None], [[42.0], None], [[46.0], None], [[45.0], None],
    [[45.0], None], [[62.0], None], [[47.0], None], [[52.0], None],
    .......]
    """
    return [[x, None] for x in datos]


def initial_centroids(data, cluster_numbers):
    """
    Returns as many values as cluster numbers randomly picked from the data
    >>> initial_centroids(population_weight['data'],2)
    [[65.0], [48.0]]
    """
    return random.sample(data, cluster_numbers)


def calculates_nearest_centroid(data, centroids, distance):
    """
    Assign for a element to a centroid by distance returning the index of the centroid list

    >>> calculates_nearest_centroid([41.0],[[39.0],[45.0]],abs_distance)
    0
    >>> calculates_nearest_centroid([64.0],[[39.0],[45.0]],abs_distance)
    1
    """
    minimum_distance = float("infinity")
    index = (-1)
    for i, c in enumerate(centroids):
        dist = distance(data, c)
        if dist < minimum_distance:
            minimum_distance = dist
            index = i
    return index


def assign_cluster(classification, centroids, distance):
    """
    Update the classification assigning a centroid to each element

    >>> clas = empty_initial_classification(population_weight['data'])
    >>> centr = [[65.0], [48.0]]
    >>> assign_cluster(clas,centr,abs_distance)
    >>> clas
    [[[51.0], 1], [[43.0], 1], [[62.0], 0], [[64.0], 0], [[45.0], 1],
    [[42.0], 1], [[46.0], 1], [[45.0], 1], [[45.0], 1], [[62.0], 0],
    [[47.0], 1], [[52.0], 1], [[64.0], 0], [[51.0], 1], [[65.0], 0],
    ...]
    """
    for x in classification:
        x[1] = calculates_nearest_centroid(x[0], centroids, distance)


def reassign_centroids(classification, num_cluster):
    """
    Update for each cluster its centroid
    """
    def sum_vec(v1, v2):
        return [x+y for (x, y) in zip(v1, v2)]

    def div_vec(v, n):
        if n != 0:
            return[x/n for x in v]
        else:
            print("Second parameter can not be 0")

    new_centroids = [[0] * len(classification[0][0])] * num_cluster
    #  [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],... num_cluster times]
    num = [0] * num_cluster
    for d in classification:
        new_centroids[d[1]] = sum_vec(new_centroids[d[1]], d[0])
        num[d[1]] += 1
    for i in range(num_cluster):
        new_centroids[i] = div_vec(new_centroids[i], num[i])
    return new_centroids


def k_means(k, datos, distancia):
    """
    Main algorithm
    Returns an array with the centroids and classification
    """
    centroids = initial_centroids(datos, k)
    classification = empty_initial_classification(datos)
    while True:
        assign_cluster(classification, centroids, distancia)
        new_centroids = reassign_centroids(classification, k)
        if new_centroids != centroids:
            centroids = new_centroids
        else:
            return [centroids, classification]