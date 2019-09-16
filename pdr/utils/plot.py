import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

###############################################################
# Define a function plotting the global position in real time #
###############################################################


def update_line(graph, new_data):
    xdata, ydata, zdata = graph._verts3d
    graph.set_xdata(list(np.append(xdata, new_data[0])))
    graph.set_ydata(list(np.append(ydata, new_data[1])))
    graph.set_3d_properties(list(np.append(zdata, new_data[2])))
    plt.draw()


def real_time_3d_plotting(period, ground_truth, estimation):
    map = plt.figure()
    ax = Axes3D(map)
    ax.autoscale(enable=True, axis='both', tight=True)

    # Setting the axes properties
    X = np.concatenate((ground_truth[0], estimation[0]))
    Y = np.concatenate((ground_truth[1], estimation[1]))
    Z = np.concatenate((ground_truth[2], estimation[2]))

    max_range = np.array(
        [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    mid_x = (X.max() + X.min()) * 0.5 + X.min()
    mid_y = (Y.max() + Y.min()) * 0.5 + Y.min()
    mid_z = (Z.max() + Z.min()) * 0.5 + Z.min()

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    graph1, *_ = ax.plot3D([ground_truth[0][0]], [ground_truth[1]
                                                  [0]], [ground_truth[2][0]], 'blue', label='Ground truth')
    graph2, *_ = ax.plot3D([estimation[0][0]],
                           [estimation[1][0]], [estimation[2][0]], '#ff6f42', label='Estimation')
    plt.legend()
    for i in range(1, len(ground_truth[0])):
        update_line(
            graph1, (ground_truth[0][i], ground_truth[1][i], ground_truth[2][i]))
        update_line(graph2, (estimation[0][i],
                             estimation[1][i], estimation[2][i]))
        plt.show(block=False)
        plt.pause(period)
    plt.show(block=True)
