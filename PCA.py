import numpy as np
import pandas as pd
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# data size: n * nDimension
# n is the data quantity
# nDimension is the features' quantity
def pca(data):
    data_mean = np.mean(data, axis=0)
    data_centralizationn = data - data_mean
    data_convariance = np.cov(data_centralizationn, rowvar=False)
    eigen_values, eigen_vectors = linalg.eig(mat(data_convariance))
    eigen_values_acsending_order = argsort(eigen_values)

    n_dimenntion = data.shape[1]
    eigen_values_ascending = eigen_values.copy()
    eigen_vectors_ascending = eigen_vectors.copy()

    for i in np.arange(n_dimenntion):
        temp_id = eigen_values_acsending_order[i]
        eigen_values_ascending[i] = eigen_values[temp_id]
        eigen_vectors_ascending[:,i] = eigen_vectors[:, temp_id]

    return eigen_values_ascending, eigen_vectors_ascending


def show_pca(data, eigen_vector):
    data = np.array(data)
    # print(data[0 : 2, :])
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    x = list(x)
    y = list(y)

    fig = plt.figure(0)
    # scatters
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('points distribution')
    plt.scatter(x, y, c='green', alpha=0.1, marker='o', label="points")
    data_mean = np.mean(data, axis=0)

    # principal line
    eigen_vector = eigen_vector.reshape(1, 3)
    p_start = data_mean
    p_end = data_mean + 2.0 * eigen_vector
    # print('p_start is', p_start)
    # print('p_end is', p_end)
    p_start_list = p_start.tolist()
    p_end_list = p_end.tolist()
    p_start_list = p_start_list[0:2]
    p_end_list = p_end_list[0]
    p_end_list = p_end_list[0:2]
    line_x = [p_start_list[0], p_end_list[0]]
    line_y = [p_start_list[1], p_end_list[1]]
    # print('line_x:', line_x)
    # print('line_y', line_y)

    plt.plot(line_x, line_y, c='red', alpha=1.0, label="normal")
    plt.legend()
    plt.show(fig)
    return True


def subsample_along_principal(data, principal, sampleN):
    data = np.array(data)
    data_center = np.mean(data, axis=0)
    principal_vector = principal.reshape(3, 1)
    data_center = data_center.reshape(1, 3)
    vectors_OQ = data - data_center
    data_proj_length = np.dot(vectors_OQ, principal_vector)
    data_with_length = np.concatenate((data, data_proj_length), axis=1)

    np.sort(data_with_length)
    data_with_length = data_with_length[np.lexsort(data_with_length.T)]
    data_with_length = data_with_length[0]

    # Choice1: random selection
    #sample_ids = np.random.randint(data_with_length.shape[0], size=sampleN)

    # Choice2: ranged selection
    sample_ids = np.arange(0, data_with_length.shape[0], step=data_with_length.shape[0]/sampleN, dtype=np.int)

    last_id = np.array([data_with_length.shape[0] - 1])
    sample_ids = np.append(sample_ids, last_id)
    sampled_points = data_with_length[sample_ids, :]
    sampled_points = sampled_points[np.lexsort(sampled_points.T)]
    sampled_points = sampled_points[0]
    sampled_points = sampled_points[:, 0:3]

    return sampled_points


def show_samples(data, sampled_data):
    data = np.array(data)
    sampled_data = np.array(sampled_data)

    fig = plt.figure(1)
    x_data = data[:, 0]
    y_data = data[:, 1]
    x_sampled_data = sampled_data[:, 0]
    y_sampled_data = sampled_data[:, 1]
    plt.scatter(x_data, y_data, c='green', alpha=0.1, marker='o', label="points")
    plt.plot(x_sampled_data, y_sampled_data, c='red', alpha=1.0, label='sampled points')
    plt.axis("equal")
    plt.legend()
    plt.show(fig)

    return True




