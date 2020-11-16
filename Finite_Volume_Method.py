#!/usr/bin/env python3.8
"""
Created on 15/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np


def fv_superb(z, d, u0, co):
    """
    1D "SuperB" Finite Volume Method (Total Variation Diminished)
    :param z: Data series
    :param d: Discrete grid space (default = 1.0)
    :param u0: velocity
    :param co: Courant Number
    :return: pnpt
    """
    e = 1e-6
    f = u0 * z
    lf = len(f)
    pnpt = np.array(range(lf), dtype=float)
    r = np.array(range(lf), dtype=float)
    phi = np.array(range(lf), dtype=float)
    fphalf = np.array(range(lf), dtype=float)
    for i in range(lf):
        if i == (lf-1):
            r[i] = (f[i] - f[i - 1]) / (f[0] - f[i] + e)
            phi[i] = np.max((0, np.min((2 * r[i], 1)), np.min((r[i], 2))))
            fphalf[i] = f[i] + 1 / 2 * phi[i] * (1 - co) * (f[0] - f[i])
            pnpt[i] = -(fphalf[i] - fphalf[i - 1]) / d
        elif i == 0:
            r[i] = (f[i] - f[i - 1]) / (f[i + 1] - f[i] + e)
            phi[i] = np.max((0, np.min((2 * r[i], 1)), np.min((r[i], 2))))
            r[i-1] = (f[i-1] - f[i - 2]) / (f[i] - f[i - 1] + e)
            phi[i-1] = np.max((0, np.min((2 * r[i-1], 1)), np.min((r[i-1], 2))))

            fphalf[i] = f[i] + 1 / 2 * phi[i] * (1 - co) * (f[i + 1] - f[i])
            fphalf[i-1] = f[i-1] + 1 / 2 * phi[i-1] * (1 - co) * (f[i] - f[i-1])
            pnpt[i] = -(fphalf[i] - fphalf[i - 1]) / d
        else:
            r[i] = (f[i] - f[i - 1]) / (f[i + 1] - f[i] + e)
            phi[i] = np.max((0, np.min((2 * r[i], 1)), np.min((r[i], 2))))
            fphalf[i] = f[i] + 1 / 2 * phi[i] * (1 - co) * (f[i + 1] - f[i])
            pnpt[i] = -(fphalf[i] - fphalf[i - 1]) / d
    return pnpt
