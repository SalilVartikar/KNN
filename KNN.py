import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

acc = 0
K = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
p = []

#get data
train_data = genfromtxt('train.csv', delimiter=',')
test_data = genfromtxt('test.csv', delimiter=',')

train_set = train_data[0:6000, :]
training_labels = train_set[:, 0]
train_set = np.delete(train_set, 0, 1)

test_set = test_data[0:1000, :]
testing_labels = test_set[:, 0]
test_set = np.delete(test_set, 0, 1)


def get_dist(te, tr):
    d1 = np.sum(np.square(te - tr), axis=1)
    return d1


def get_nb(n, t):
    n = np.argsort(n, kind='mergesort')
    n = n[0:t]
    n = training_labels[n]
    n = n.astype(np.int64)
    n = np.bincount(n)
    return n


for k in K:
    i = 0
    acc = 0
    for row in test_set:
        d = get_dist(row, train_set)
        knn = get_nb(d, k)
        knn = np.argmax(knn)
        if testing_labels[i] == knn:
            acc += 1
        i += 1
    p.append(float(acc) / 10)
print(p)

plt.plot(K, p, 'b', marker='o', label='Test Error')
plt.ylim([0, 100])
plt.ylabel('Accuracy')
plt.xlabel('K Neighbors')
plt.legend()
plt.show()