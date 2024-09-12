import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # plt.figure()
    # im = plt.imshow(np.reshape(np.random.rand(100), newshape=(10,10)),
    #                 interpolation='none', vmin=0, vmax=1, aspect='equal')
    #
    # ax = plt.gca()
    #
    # # Major ticks
    # ax.set_xticks(np.arange(0, 10, 1))
    # ax.set_yticks(np.arange(0, 10, 1))
    #
    # # Labels for major ticks
    # ax.set_xticklabels(np.arange(1, 11, 1))
    # ax.set_yticklabels(np.arange(1, 11, 1))
    #
    # # Minor ticks
    # ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    # ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    #
    # # Gridlines based on minor ticks
    # ax.grid(which='both', color='w', linestyle='-', linewidth=2)
    #
    # # Remove minor ticks
    # ax.tick_params(which='minor', bottom=False, left=False)

    data = np.random.rand(10, 10)
    plt.pcolormesh(data, edgecolors='k', linewidth=2)
    ax = plt.gca()
    ax.set_aspect('equal')

    plt.show()


