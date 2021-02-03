
# K-means Clustering Python Example
# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

import time
import os
os.system("cls")

import warnings                                  
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def get_program_running(start_time):
    end_time = time.clock()
    diff_time = end_time - start_time
    result = time.strftime("%H:%M:%S", time.gmtime(diff_time)) 
    print("program runtime: {}".format(result))

def main():
    print("module03_kmeans_example.py")

    # apply any dataset for X and y!
    X, y = make_blobs(n_samples=1000, n_features=6, centers=5, cluster_std=0.7, random_state=0)
    print(X)
    print(y)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()
    y = np.reshape(y, (1000, 1))
    print(y)

    X = np.append(X, y, axis=1)
    print(X)

    df = pd.DataFrame(X)
    df.to_csv(path_or_buf="papa.csv", header=None, sep=",", index=False)  


    exit()

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()

if __name__ == '__main__':
    start_time = time.clock()
    main()
    get_program_running(start_time)
