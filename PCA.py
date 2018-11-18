import numpy as np
import pandas as pd
from numpy import *

def pca(data):
    data_mean = np.mean(data, axis=0)
    data_centralizationn = data - data_mean
    data_convariance = np.cov(data_centralizationn, rowvar=False)
    eigen_values, eigen_vectors = linalg.eig(mat(data_convariance))
    eigen_values_acsending_order = argsort(eigen_values)

    print("data_mean is:", data_mean)
    print("eigen values order is:", eigen_values_acsending_order)
    print("eigen values are:")
    print(eigen_values)
    print("eigen vectors are:")
    print(eigen_vectors)
    n_dimenntion = data.shape[1]
    eigen_values_ascending = eigen_values.copy()
    eigen_vectors_ascennding = eigen_vectors.copy()

    print("eigen vector ascennding is:")
    print(eigen_vectors_ascennding)

    for i in np.arange(n_dimenntion):
        temp_id = eigen_values_acsending_order[i]
        print("tempory values id is:", eigen_values_acsending_order[i])
        print("aim eigen value is:", eigen_values_ascending[i])
        print("source eigen value is:", eigen_values[temp_id])
        print("temp eigen vector is: ", eigen_vectors_ascennding[:, i])
        print("test vector is: ", eigen_vectors[:, temp_id])
        print("test vector2 is:", eigen_vectors_ascennding[:, temp_id])
        print("temp input eigen vector is:", eigen_vectors[:, temp_id])

        eigen_values_ascending[i] = eigen_values[temp_id]
        eigen_vectors_ascennding[:,i] = eigen_vectors[:, temp_id]

    print("after sorting:")
    print("eigen values are:")
    print(eigen_values_ascending)
    print("eigen vectors are:")
    print(eigen_vectors_ascennding)
    return eigen_values_ascending, eigen_vectors_ascennding