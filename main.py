#!/usr/bin/env python3.8
"""
Created on 15/11/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np
import netCDF4 as nc
import ode_solver
import pde_solver
import plot_data
import Finite_Volume_Method as fv

def init(n=256):
    x_grid = pde_solver.generate_grid(xb, n)
    y = np.zeros(n)
    ic = 2
    """
    ic: The identifier of initial condition
    1: Square wave
    2: Sine
    3: Gaussian    
    """
    if ic == 1:
        y[int(0.4*n):int(0.6*n)] = 1.0
    elif ic == 2:
        y = np.sin(np.deg2rad(x_grid))
    elif ic == 3:
        y = np.exp(-(x_grid - xb/2)**2 / (2*(0.2*xb/2)**2))
    return x_grid, y


def init_nc():
    file_path = './ic_homework3.nc'
    file_obj = nc.Dataset(file_path)
    x_deg = file_obj.variables['x_deg'][:]
    x_rad = file_obj.variables['x_rad'][:]
    y = file_obj.variables['N'][:]
    return x_deg, x_rad, y


def tend_fd(y):
    if pde_scheme == 1:
        pypx = pde_solver.centraldiff2_1d(y, dx)
    elif pde_scheme == 2:
        pypx = pde_solver.centraldiff4_1d(y, dx)
    else:
        print("Undefined identifier: (pde_scheme) is undefined!!")
        raise SystemExit
    return -u0*pypx


def tend_psm(y):
    c = np.fft.fft(y)
    for k in range(kmax):
        c[k] = c[k] * jj * k * (2*np.pi/nx)
        c[-k] = c[-k] * jj * (-k) * (2*np.pi/nx)
    c[kmax] = c[kmax] * jj * kmax * (2*np.pi/nx)
    pypx = np.fft.ifft(c)
    return -u0*pypx.real


def tend_fv(y):
    pypt = fv.fv_superb(y, dx, u0, co)
    return pypt


if __name__ == '__main__':
    nx = 360
    xb = 360        # unit: degree
    test = False
    save = False
    if test:
        x, y0 = init(nx)
    else:
        x, x_r, y0 = init_nc()
    # print("x = ", x)
    # print("y0 = ", y0)
    dx = x[1] - x[0]
    dt = 0.02       # unit: day
    steps = 1800
    u0 = 10         # unit: degree/day
    kmax = int(nx/2)
    jj = (0+1j)     # The imaginary unit
    ode_scheme = "rk4"
    pde_scheme = 2

    co = u0*dt/dx
    try:
        if co < 1:
            print("The CFL condition is satisfied, with Courant Number = ", u0*dt/dx)
    except Exception:
        print("The CFL condition is violated!!")
        raise SystemExit

    print("\n Method 1: Finite Difference")
    FD = ode_solver.Ode(y0, tend_fd, dt, steps, debug=0)
    FD.integrate(ode_scheme)

    print("\n Method 2: Pseudo Spectral Method")
    PSM = ode_solver.Ode(y0, tend_psm, dt, steps, debug=0)
    PSM.integrate(ode_scheme)

    print("\n Method 3: Pseudo Spectral Method")
    FV = ode_solver.Ode(y0, tend_fv, dt, steps, debug=0)
    FV.integrate(scheme="forward")
    print(FV.trajectory[-1, :])

    # ---------------------------------------
    # Plotting
    # ---------------------------------------
    plot_data.plot_2d(4, x,
                      (y0, FD.trajectory[-1, :],
                       PSM.trajectory[-1, :], FV.trajectory[-1, :]),
                      color=('y', 'r', 'b', 'g'), lw=(5, 2, 2, 2),
                      lb=("Analytic", "FD", "PSM", "FV"), ti="Plot", xl="Longitude", yl="N",
                      xlim=(0, 360), ylim=(-0.25, 1.25),
                      fn="plot_compare_method.pdf", sa=save)
