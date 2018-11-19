import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
from PCA import pca
from PCA import show_pca
from PCA import subsample_along_principal
from PCA import show_samples

from curve_plot import plot_scatters
from curve_plot import plot_1st_order_curve
from curve_plot import plot_2nd_order_curve
from curve_plot import plot_3rd_order_curve
from curve_plot import plot_1_2_3_order_fitting_curve

# xs = range(500)
# ys = randn(500) * 1.0 + 10.0
#
# plt.plot(xs, ys)
# print('mean of the reading is: {:.3f}'.format(np.mean(ys)))

# 1. import points
points = pd.read_csv('/Users/mixiaoxin/Desktop/single_curb_points_cloud/curb_curve4.txt', dtype=np.float64)
print('points size is: ')
print(points.shape)

points = points[points.columns[0:3]]
number_points = points.shape[0]
dimension_points = points.shape[1]

print('after shape resize, points size is: ')
print(points.shape)
print(points.shape[0], points.shape[1])
print(points.ix[0:2, ])
points = pd.DataFrame(points)
# 2. points subsample
# Select N points randomly to fit 3rd Bezier Curve as 4 Control Points by minimum residual error.
# ===============================================================================================
# B(t) = (1 - t^3)*P0 + 3*t*(1 - t^2)*P1 + 3*t^2*(1 - t)*P2 + t^3 * P3
#
#                                      [ 1  3 -3  1]   [P0]
# B(t) = T(t)*CP = [t^3 t^2 t^1 t^0] * | 3 -6  3  0| * |P1|
#                                      |-3  3  0  0|   |P2|
#                                      [ 1  0  0  0]   [P3]
#
# t_j = sum(i = 1 : j){d(p_i, p_(i-1))} / sum(i = 1 : N){d(p_i, p_(i-1))}
# d(p_i, p_j) = sqrt((u_i - u_j)^2 + (v_i - v_j)^2)
# ===============================================================================================
#

N_select = 10
# calculate principal direction, eigen values are sorting by ascending
# set z_value as 0.
points = np.array(points)
points[:, 2] = 0.
eigen_values, eigen_vectors = pca(points)
principal = eigen_vectors[:, -1]
normal = eigen_vectors[:, -2]
print("principal direction is:\n", principal)
print('normal direction is\n', normal)

#show_pca(points, normal)

sampled_points = subsample_along_principal(points, principal, N_select)
show_samples(points, sampled_points)
figure_title = 'curve fitting'
# plot_scatters(points, figure_title)
# plot_1st_order_curve(points, figure_title)
# plot_2nd_order_curve(points, figure_title)
plot_3rd_order_curve(points, figure_title)

plot_1_2_3_order_fitting_curve(points, figure_title)

# 3. Control Points Calculation for 3rd order Bezier Curve


# 4. extend the Curve


# 5. show Bezier Curve and original points




