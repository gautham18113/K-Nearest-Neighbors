"""
Implementation of K-Nearest Neighbors algorithm in python.

Pseudo code:
- Load the data
- Initialise the value of k
- For getting the predicted class, iterate from 1 to total number of training data points
- Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it’s the most popular method. The other metrics that can be used are Chebyshev, cosine, etc.
- Sort the calculated distances in ascending order based on distance values
- Get top k rows from the sorted array
- Get the most frequent class of these rows
- Return the predicted class
"""

import pandas as pd
import numpy as np
import operator
import math
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("iris.csv")
print(data.head())


def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


# KNN model
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[1]

    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classVotes = {}

    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors)


testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

k = 3

result, neigh = knn(data, test, k)

print(result)
print(neigh)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data.iloc[:,0:4], data['species'])
print(neigh.predict(test))
print(neigh.kneighbors(test)[1])