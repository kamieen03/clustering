#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

def main():
    algs = ['kmeans']
    for alg in algs:
        for i in range(6):
            with open(f'data/out/{alg}/{i}.csv', 'r') as f:
                data = {}
                for line in f.readlines():
                    line = line.split()
                    cl = int(line[-1])
                    point = [float(x) for x in line[:-1]]
                    if cl not in data:
                        data[cl] = []
                    data[cl].append(point)
                plt.subplot(2, 3, i+1)
                plt.title(i)
                for cluster in data.values():
                    cluster = np.array(cluster)
                    plt.scatter(cluster[:,0], cluster[:,1])
        plt.show()

if __name__ == '__main__':
    main()
