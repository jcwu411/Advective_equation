#!/usr/bin/env python3.8
"""
Created on 15/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""


import numpy as np
import matplotlib.pyplot as plt


def plot_2d(n, x, y, color, lw, lb,
            ti="Plot", xl="X", yl="Y", legendloc=4,
            xlim=(0, 1), ylim=(0, 1), ylog=False,
            fn="plot2d.pdf", sa=False):
    """
    Plot n lines on x-y coordinate system
    :param n: The number of the plot line
    :param x:
    :param y:
    :param color: The color of each line
    :param lw: The width of each line
    :param lb: The label of each line
    :param ti: The title of plot
    :param xl: The label of x axis
    :param yl: The label of y axis
    :param legendloc: The location of legend
    :param xlim: The range of x axis
    :param ylim: The range of y axis
    :param ylog: Using logarithmic y axis or not
    :param fn:  The saved file name
    :param sa:  Saving the file or not
    :return: None
    """

    plt.figure()
    for i in range(n):
        plt.plot(x, y[i], color=color[i], linewidth=lw[i], label=lb[i])

    plt.title(ti)

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if ylog:
        plt.yscale('log')

    plt.legend(shadow=True, loc=legendloc)

    if sa:
        plt.savefig(fn)
    plt.show()
    plt.close()
