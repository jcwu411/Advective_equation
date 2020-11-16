#!/usr/bin/env python3.8
"""
Created on 15/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np


def forward1_1d(z, d=1):
    """
    1D Forward scheme (1st order)
    :param z: Data series
    :param d: Discrete grid space (default = 1.0)
    :return:
    """
    lz = len(z)      # Length of the Data size
    dzdx = np.array(range(lz), dtype=float)
    for i in range(lz):
        if i == (lz-1):
            dzdx[i] = (z[0] - z[i]) / d
        else:
            dzdx[i] = (z[i+1] - z[i]) / d
    return dzdx


def backward1_1d(z, d=1):
    """
    1D Backward scheme (1st order)
    :param z: Data series
    :param d: Discrete grid space (default = 1.0)
    :return: dzdx
    """
    lz = len(z)  # Length of the Data size
    dzdx = np.array(range(lz), dtype=float)
    for i in range(lz):
            dzdx[i] = (z[i] - z[i-1]) / d
    return dzdx


def centraldiff2_1d(z, d=1):
    """
    1D Central difference scheme (2nd order)
    :param z: Data series
    :param d: Discrete grid space (default = 1.0)
    :return: dzdx
    """
    lz = len(z)  # Length of the Data size
    dzdx = np.array(range(lz), dtype=float)
    for i in range(lz):
        if i == (lz - 1):
            dzdx[i] = (z[0] - z[i-1]) / (2*d)
        else:
            dzdx[i] = (z[i+1] - z[i-1]) / (2*d)
    return dzdx


def centraldiff4_1d(z, d=1):
    """
    1D Central difference scheme (4th order)
    :param z: Data series
    :param d: Discrete grid space (default = 1.0)
    :return: dzdx
    """
    lz = len(z)  # Length of the Data size
    dzdx = np.array(range(lz), dtype=float)
    for i in range(lz):
        if i == (lz - 1):
            dzdx[i] = (-z[1] + 8*z[0] - 8*z[i-1] + z[i-2]) / (12*d)
        elif i == (lz - 2):
            dzdx[i] = (-z[0] + 8*z[i+1] - 8*z[i-1] + z[i-2]) / (12*d)
        else:
            dzdx[i] = (-z[i+2] + 8*z[i+1] - 8*z[i-1] + z[i-2]) / (12*d)
    return dzdx


def generate_grid(xb, n):
    """
    Generate n points within 2pi
    :param xb: The boundary of x
    :param n: The number of point within a giving range
    :return: x
    """
    d = xb/n
    dhalf = d/2
    x = np.linspace(dhalf, xb-dhalf, n)
    return x


