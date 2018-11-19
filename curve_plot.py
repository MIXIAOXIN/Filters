import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============= NOTE ============================
#
# if the data's value is quite large
# should make it centralized, or curve fitting unsuccessfully
#
# ==================================================

def fun_lst_order_curve(x, A, B):
    return A*x + B

def fun_2nd_order_curve(x, A, B, C):
    return A*x*x + B*x + C

def fun_3rd_order_curve(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D


def plot_scatters(points, fig_title):
    # points_center = np.mean(points, axis=0)
    # points = points - points_center
    x_0 = points[:, 0]
    y_0 = points[:, 1]
    x_0 = list(x_0)
    y_0 = list(y_0)
    plt.figure(fig_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_0, y_0, c='green', alpha=0.1, marker='o', label="points")
    plt.legend()
    plt.axis("equal")
    plt.show()

    return


def plot_1st_order_curve(points, fig_title):
    points_center = np.mean(points, axis=0)
    points = points - points_center
    x_0 = points[:, 0]
    y_0 = points[:, 1]
    x_0 = list(x_0)
    y_0 = list(y_0)

    x_min = np.nanmin(x_0)
    x_max = np.nanmax(x_0)

    # line fitting
    popt, pocv = curve_fit(fun_lst_order_curve, x_0, y_0)
    x_fitted = np.arange(x_min, x_max, step=0.01)

    # plot
    plt.figure(fig_title)
    plt.plot(x_fitted, fun_lst_order_curve(x_fitted, *popt), c="blue", label="line fitting")
    plt.legend()
    plt.axis("equal")
    plt.show()

    return


def plot_2nd_order_curve(points, fig_title):
    points_center = np.mean(points, axis=0)
    points = points - points_center
    x_0 = points[:, 0]
    y_0 = points[:, 1]
    x_0 = list(x_0)
    y_0 = list(y_0)

    x_min = np.nanmin(x_0)
    x_max = np.nanmax(x_0)

    # line fitting
    popt, pocv = curve_fit(fun_2nd_order_curve, x_0, y_0)
    x_fitted = np.arange(x_min, x_max, step=0.01)

    # plot
    plt.figure(fig_title)
    plt.plot(x_fitted, fun_2nd_order_curve(x_fitted, *popt), c="purple", label="2nd spline curve fitting")
    plt.legend()
    plt.axis("equal")
    plt.show()

    return

def plot_3rd_order_curve(points, fig_title):
    points_center = np.mean(points, axis=0)
    points = points - points_center
    x_0 = points[:, 0]
    y_0 = points[:, 1]
    x_0 = list(x_0)
    y_0 = list(y_0)

    # x_0 = np.array(x_0.flatten())
    # x_0 = x_0[0]
    # y_0 = np.array(y_0.flatten())
    # y_0 = y_0[0]

    x_min = np.nanmin(x_0)
    x_max = np.nanmax(x_0)

    # line fitting
    popt, pocv = curve_fit(fun_3rd_order_curve, x_0, y_0)
    x_fitted = np.arange(x_min, x_max, step=0.01)

    # plot
    plt.figure(fig_title)
    plt.plot(x_fitted, fun_3rd_order_curve(x_fitted, *popt), c="red", label="3rd spline curve fitting")
    plt.legend()
    plt.axis("equal")
    plt.show()

    return


def plot_1_2_3_order_fitting_curve(points, fig_title):
    points_center = np.mean(points, axis=0)
    points = points - points_center
    x_0 = points[:, 0]
    y_0 = points[:, 1]
    x_0 = list(x_0)
    y_0 = list(y_0)

    x_min = np.nanmin(x_0)
    x_max = np.nanmax(x_0)
    x_fitted = np.arange(x_min, x_max, step=0.01)

    # line
    popt1, pocv1 = curve_fit(fun_lst_order_curve, x_0, y_0)
    # 2nd curve
    popt2, pocv2 = curve_fit(fun_2nd_order_curve, x_0, y_0)
    # 3rd curve
    popt3, pocv3 = curve_fit(fun_3rd_order_curve, x_0, y_0)


    plt.figure(fig_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_0, y_0, c='green', alpha=0.1, marker='o', label="points")
    plt.plot(x_fitted, fun_lst_order_curve(x_fitted, *popt1), c="blue", label="line fitting")
    plt.plot(x_fitted, fun_2nd_order_curve(x_fitted, *popt2), c="purple", label="2nd spline curve fitting")
    plt.plot(x_fitted, fun_3rd_order_curve(x_fitted, *popt3), c="red", label="3rd spline curve fitting")

    plt.legend()
    plt.axis("equal")
    plt.show()

    return
