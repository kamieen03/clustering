#!/usr/bin/env python3

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

def main():

    n_samples = 1500
    data = {}
    data[0] = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)[0]
    data[1] = datasets.make_moons(n_samples=n_samples, noise=.05)[0]
    data[2] = datasets.make_blobs(n_samples=n_samples)[0]
    data[3] = np.random.rand(n_samples, 2)

    aniso, _ = datasets.make_blobs(n_samples=n_samples)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    data[4] = np.dot(aniso, transformation)

    data[5] = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5])[0]

    for i in range(6):
        np.savetxt(f'data/{i+1}.csv', data[i], delimiter=' ')
        plt.subplot(2, 3, i+1)
        plt.title(i+1)
        plt.scatter(data[i][:,0], data[i][:,1])
    plt.show()

if __name__ == '__main__':
    main()
