# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:20:15 2018

@author: user
"""

"""SPH class to find nearest neighbours..."""

from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from os import path
import csv


class SPH_main(object):
    """Primary SPH object"""

    def __init__(self):
        self.h = 0.0
        self.h_fac = 0.0
        self.dx = 0.0

        self.dt = 0.0
        self.t0 = 0.0
        self.t_max = 0.0

        self.min_x = np.zeros(2)
        self.max_x = np.zeros(2)
        self.min_x_with_boundary = np.zeros(2)
        self.max_x_with_boundary = np.zeros(2)
        self.max_list = np.zeros(2, int)

        self.particle_list = []
        self.search_grid = np.empty((0, 0), object)

        # Physical properties
        self.mu = 0.001  # in Pa s
        self.rho0 = 1000  # kg / m^3
        self.g = 9.81  # m^2 s^-2
        self.c0 = 20  # m s ^-1
        self.gamma = 7

    def set_values(self, min_x=(0.0, 0.0), max_x=(20, 10), dx=0.2, h_fac=1.3, t0=0.0, t_max=0.3):
        """Set simulation parameters."""

        self.min_x[:] = min_x
        self.max_x[:] = max_x
        self.dx = dx
        self.h_fac = h_fac
        self.h = self.dx * self.h_fac

        self.dt = 0.1 * (self.h / self.c0)
        self.t0 = t0
        self.t_max = t_max

    def initialise_grid(self):
        """Initalise simulation grid."""

        """Increases the minimum and maximum to account for the virtual particle padding that is required at boundaries"""
        self.min_x = self.min_x - 2.0 * self.h
        self.max_x = self.max_x + 2.0 * self.h

        """Calculates the size of the array required to store the search array"""
        self.max_list = np.array((self.max_x - self.min_x) / (2.0 * self.h) + 1,
                                 int)
        self.search_grid = np.empty(self.max_list, object)

    def place_points(self, xmin, xmax):
        """Place points in a rectangle with a square spacing of size dx"""
        inner_xmin = xmin + 2 * self.h  # Inner xmin is point 0,0
        inner_xmax = xmax - 2 * self.h  # Inner xmax is point 20,10

        # Add boundary particles
        for i in np.arange(inner_xmin[0] - 3*self.dx, inner_xmax[0] + 3*self.dx, self.dx):  # Maybe change to 2*dx for 3 boundary points
            for j in np.arange(inner_xmin[1] - 3*self.dx, inner_xmax[1] + 3*self.dx, self.dx):
                if not inner_xmin[0] < i < inner_xmax[0] or not inner_xmin[1] < j < inner_xmax[1]:
                    x = np.array([i, j])
                    particle = SPH_particle(self, x)
                    particle.calc_index()
                    particle.boundary = True
                    self.particle_list.append(particle)

        # Add fluid particles
        for i in np.arange(inner_xmin[0] + self.dx, inner_xmax[0], self.dx):  # X [0+dx : 20-dx]
            for j in np.arange(inner_xmin[1] + self.dx, inner_xmin[1] + 2, self.dx):  # Y [0+dx : 0+2]
                    x = np.array([i, j])
                    particle = SPH_particle(self, x)
                    particle.calc_index()
                    self.particle_list.append(particle)

        for i in np.arange(inner_xmin[0] + self.dx, inner_xmin[0] + 3, self.dx):  # X [0+dx : 0+3]
            for j in np.arange(inner_xmin[1] + 2, inner_xmin[1] + 2 + 3, self.dx):  # Y [0+2+dx : 0+2+3]
                    x = np.array([i, j])
                    particle = SPH_particle(self, x)
                    particle.calc_index()
                    self.particle_list.append(particle)

    def allocate_to_grid(self):
        """Allocate all the points to a grid in order to aid neighbour searching"""
        for i in range(self.max_list[0]):
            for j in range(self.max_list[1]):
                self.search_grid[i, j] = []

        for cnt in self.particle_list:
            # Keep in mind, list_num is bucket coordinates
            self.search_grid[cnt.list_num[0], cnt.list_num[1]].append(cnt)

    def neighbour_iterate(self, part):
        """Find all the particles within 2h of the specified particle"""
        # save neighbours (particles j) of particles i
        neighbours = []
        for i in range(max(0, part.list_num[0] - 1),
                       min(part.list_num[0] + 2, self.max_list[0])):
            for j in range(max(0, part.list_num[1] - 1),
                           min(part.list_num[1] + 2, self.max_list[1])):
                for other_part in self.search_grid[i, j]:
                    if part is not other_part:
                        dn = part.x - other_part.x
                        dist = np.sqrt(np.sum(dn ** 2))
                        if dist < 2.0 * self.h:
                            neighbours.append(other_part)
                            print("id:", other_part.id, "dn:", dn)
        return neighbours

    def diff_W(self, part, other_part):
        dn = part.x - other_part.x  # dn is r_ij (vector)
        dist = np.sqrt(np.sum(dn ** 2))  # dist is |r_ij| (scalar)
        q = dist / self.h
        dw = 0
        if 0 <= q <= 1:
            dw = (10 / (7 * np.pi * self.h ** 2)) * ((-3 * dn)/self.h**2 + (9/4) * (dist * dn) / self.h**3)
        if 1 <= q <= 2:
            dw = (10 / (7 * np.pi * self.h ** 2)) * ((-1/4) * (3 * dist * dn) / self.h**3 +
                                                     (3 * dn)/self.h**2 - (3/self.h**3) * (dn / dist))
        if 2 < q:
            dw = 0
        return dw

    def grad_W(self, part, other_part):
        dn = part.x - other_part.x  # dn is r_ij (vector)
        dist = np.sqrt(np.sum(dn ** 2))  # dist is |r_ij| (scalar)
        e_ij = dn / dist
        q = dist / self.h
        dw = 0
        if 0 <= q <= 1:
            dw = (10 / (7 * np.pi * self.h ** 2)) * ((-3 * dn)/self.h**2 + (9/4) * (dist * dn) / self.h**3)
        if 1 <= q <= 2:
            dw = (10 / (7 * np.pi * self.h ** 2)) * ((-1/4) * (3 * dist * dn) / self.h**3 +
                                                     (3 * dn)/self.h**2 - (3/self.h**3) * (dn / dist))
        if 2 < q:
            dw = 0
        return dw * e_ij

    def navier_cont(self, part, neighbours):
        # Set acceleration to 0 initially for sum
        a = 0
        # Set derivative of density to 0 initially for sum
        D = 0
        for nei in neighbours:
            # Calculate distance between 2 points
            r = part.x - nei.x
            dist = np.sqrt(np.sum(r ** 2))
            # Calculate the difference of velocity
            v = part.v - nei.v
            # Calculate acceleration of particle
            a += -(nei.m * ((part.P / part.rho ** 2) + (nei.P / nei.rho ** 2)) * self.grad_W()) + \
                 self.mu * (nei.m * ((1 / part.rho ** 2) + (nei.P / nei.rho ** 2)) * self.diff_W() * (v / dist)) + self.g
            # normal
            e = r / dist
            # Calculate the derivative of the density
            D += nei.m * np.dot(self.diff_W() * v, e)
        return a, D

    def forward_euler(self, part, t0, t_max):
        x_all = [part.x]
        v_all = [part.v]
        rho_all = [part.rho]
        t_all = [t0]
        x = part.x
        v = part.v
        rho = part.rho
        t = t0
        while t < t_max:
            x = x + self.dt * v
            v = v + self.dt * part.a
            rho = rho + self.dt * part.D
            t = t + self.dt
            x_all.append(x)
            v_all.append(v)
            rho_all.append(rho)
            t_all.append(t)
        return t, x, v, rho

    def simulate(self):
        # We are returning a list of particles per time step in a list of lists

        particles_times = []
        for particle in self.particle_list:
            # Get neighbours of particle
            neighbours = self.neighbour_iterate(particle)
            # Navier stokes equation
            a, D = self.navier_cont(particle, neighbours)
            # Set the acceleration and derivative of density
            particle.a = a
            particle.D = D
            # Forward euler step in time
            t_all, x_all, v_all, rho_all = self.forward_euler(particle, self.t0, self.t_max)

            particles = [None] * len(t_all)
            for i in range(len(t_all)):
                particle.x = x_all[i]
                particle.v = v_all[i]
                particle.rho = rho_all[i]
                dict[t_all[i]].append(particle)

        return particles_times
                        
    def output_particle(self):
        bound = []
        x_value = []
        y_value = []
        for part in self.particle_list:
            x_value.append(part.x[0])
            y_value.append(part.x[1])
            if part.boundary==True:
                bound.append(True)
            else:
                bound.append(False)
        return x_value, y_value, bound



class SPH_particle(object):
    """Object containing all the properties for a single particle"""

    _ids = count(0)

    def __init__(self, main_data=None, x=np.zeros(2)):
        self.id = next(self._ids)
        self.main_data = main_data
        self.x = np.array(x)
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        self.D = 0
        self.rho = 0.0
        self.P = 0.0
        self.m = main_data.dx ** 2 * main_data.rho0  # initial mass depends on the initial particle spacing
        self.boundary = False  # Particle by default is not on the boundary

    def calc_index(self):
        """Calculates the 2D integer index for the particle's location in the search grid"""
        # Calculates the bucket coordinates
        self.list_num = np.array((self.x - self.main_data.min_x) /
                                 (2.0 * self.main_data.h), int)

    def B(self):
        return (self.main_data.rho0 * self.main_data.c0 ** 2) / self.main_data.gamma

    def calc_P(self):
        """
        Equation of state
        System is assumed slightly compressible
        """
        rho0 = self.main_data.rho0
        gamma = self.main_data.gamma
        return self.B() * ((self.rho / rho0) ** gamma - 1)



"""Create a single object of the main SPH type"""
domain = SPH_main()

"""Calls the function that sets the simulation parameters"""
domain.set_values()
"""Initialises the search grid""" 
domain.initialise_grid()

"""Places particles in a grid over the entire domain - In your code you will need to place the fluid particles in only the appropriate locations"""
domain.place_points(domain.min_x, domain.max_x)

"""This is only for demonstration only - In your code these functions will need to be inside the simulation loop"""
"""This function needs to be called at each time step (or twice a time step if a second order time-stepping scheme is used)"""
domain.allocate_to_grid()
"""This example is only finding the neighbours for a single partle - this will need to be inside the simulation loop and will need to be called for every particle"""
domain.neighbour_iterate(domain.particle_list[100])

#state is the output of main function, which stores the particle state (x, v, a, D, rho, P, m) at each time. 
#states = [state1, state2, state3, ...]




def save_csv(name, x, y, bound):
    """
    Write out a csv file, delimited with comma (,)

    Parameters
    ----------

    name: output filename
    state: state of the particle
           (coordinate, velocity, acceleration, D, density, pressure, mass
           ie. now it only contains coordinate)
    kinetic: kinetic energy loss at a point
    time: timestamp

    """
    header = ("X, Y, bound")

    data = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), 
                      np.reshape(bound, (len(bound), 1))))
    loc = path.join('data', name)
    np.savetxt(loc+".csv", data, delimiter=",", fmt='%s', header=header)


x, y, bound = domain.output_particle()
save_csv("Coordinate", x, y, bound)


def load_csv(filename):
    """
    Load the state data from csv file

    Parameters
    ----------
    filename : the filename of csv data file

    Returns
    -------
    Array of the x and y
    ie. This is just a simple version
    we can also load other state data after we finish the main function
    """
    loc = path.join('data', filename)
    assert(path.exists(loc))
    assert(path.isfile(loc))
    data = np.loadtxt(loc, dtype=float, delimiter=',',
                      comments='#', skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


coordinate = 'Coordinate.csv'
x_load, y_load, bound = load_csv(coordinate)
fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(111)
for i in range(len(bound)):
    if bound[i] == 1:
        ax1.plot(x_load[i], y_load[i], 'b.')
    else:
        ax1.plot(x_load[i], y_load[i], 'r.')