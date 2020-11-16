#!/usr/bin/env python3.8
"""
Created on 28/10/2020
@author: Jiacheng Wu, jcwu@pku.edu.cn
"""

import numpy as np


# Ordinary differential equation
class Ode:
    def __init__(self, iv=None, function=None, dt=0.01, steps=1000, debug=0):
        """
        :param iv: Initial value
        :param function: Tendency function
        :param dt: Time step (sec)
        :param steps: Number of steps to integrate
        :param debug: Debug level (0=None, 1=Basic, 2=Detail)
        """
        try:
            if len(iv) == len(function(iv)):
                self.dim = len(iv)
        except Exception:
            print("The number of initial value does not match the number of Tendency function!")
            raise SystemExit

        self.F = function
        self.dt = dt
        self.steps = steps + 1    # The initial value occupies one step!
        self.debug = debug

        self.trajectory = np.ndarray(shape=(self.steps, self.dim), dtype=float)
        self.trajectory[0, :] = np.array(iv, dtype=float)

        if self.debug >= 1:
            print("\nThe initial value =", iv)

    def integrate(self, scheme="rk4"):
        if self.debug >= 1:
            print("Start integrating ... scheme =", scheme)
        i = 0
        while i < self.steps-1:
            if scheme == "forward":
                self.trajectory[i + 1, :] = self.scheme_forward(self.trajectory[i, :])

            elif scheme == "leapfrog":
                if i == 0:
                    self.trajectory[i + 1, :] = self.scheme_forward(self.trajectory[i, :])
                else:
                    self.trajectory[i + 1, :] = self.scheme_leapfrog(self.trajectory[i, :], self.trajectory[i - 1, :])

            elif scheme == "ab2":
                if i == 0:
                    self.trajectory[i + 1, :] = self.scheme_forward(self.trajectory[i, :])
                else:
                    self.trajectory[i + 1, :] = self.scheme_ab2(self.trajectory[i, :], self.trajectory[i - 1, :])

            elif scheme == "rk4":
                self.trajectory[i + 1, :] = self.scheme_rk4(self.trajectory[i, :])

            if self.debug >= 2:
                print("step: "+str(i), end=", value: ")
                print(self.trajectory[i + 1, :])

            i = i + 1

        if self.debug >= 1:
            print("Stop integrating ... total steps = ", self.steps-1)

    def scheme_forward(self, x):
        x_new = x + self.dt * self.F(x)
        return x_new

    def scheme_leapfrog(self, x, x_old):
        x_new = x_old + 2 * self.dt * self.F(x)
        return x_new

    def scheme_ab2(self, x, x_old):
        # Adams-Bashforth-2nd-order
        x_new = x + 1/2 * self.dt * (3 * self.F(x) - self.F(x_old))
        return x_new

    def scheme_rk4(self, x):
        # Runge-Kutta-4th-order
        q1 = self.dt * self.F(x)
        q2 = self.dt * self.F(x+1/2*q1)
        q3 = self.dt * self.F(x+1/2*q2)
        q4 = self.dt * self.F(x+q3)
        x_new = x + 1/6*(q1+2*q2+2*q3+q4)
        return x_new
