"""SPH class to find nearest neighbours..."""
import numpy as np
import particle as particleClass
import csv
import pickle
from progressbar import Percentage, Bar, Timer, ETA, FileTransferSpeed, ProgressBar

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
        self.inner_min_x = np.zeros(2)
        self.inner_max_x = np.zeros(2)
        self.min_x_with_boundary = np.zeros(2)
        self.max_x_with_boundary = np.zeros(2)
        self.max_list = np.zeros(2, int)

        self.particle_list = []
        self.search_grid = np.empty((0, 0), object)

        # Physical properties
        self.mu = 0.001  # in Pa s
        self.rho0 = 1000  # kg / m^3
        self.g = np.array([0.0, -9.81])  # m^2 s^-2
        self.c0 = 20  # m s ^-1
        self.gamma = 7

        # For smoothing kernel and its derivative
        self.constant_kernel = 0

        # For predictor-corrector scheme
        self.C_CFL = 0.2

        # Stencil scheme
        self.stencil = True

    def set_values(self, min_x=(0.0, 0.0), max_x=(20, 10), dx=0.2, h_fac=1.3, t0=0.0, t_max=30, dt=0, C_CFL=0.2,
                   stencil=True):
        """Set simulation parameters."""

        self.min_x[:] = min_x
        self.max_x[:] = max_x
        self.dx = dx
        self.h_fac = h_fac
        self.h = self.dx * self.h_fac

        self.dt = 0.1 * (self.h / self.c0) if dt == 0 else dt
        self.t0 = t0
        self.t_max = t_max

        self.constant_kernel = 10 / (7 * np.pi * self.h ** 2)
        self.C_CFL = C_CFL

        # For repulsive force
        self.dwall = 0.9 * self.dx

        # Stencil
        self.stencil = stencil

    def initialise_grid(self):
        """Initalise simulation grid."""
        """Increases the minimum and maximum to account for the virtual particle padding that is required at boundaries"""
        self.min_x = self.min_x - 2.0 * self.h
        self.max_x = self.max_x + 2.0 * self.h

        """Calculates the size of the array required to store the search array"""
        self.max_list = np.array((self.max_x - self.min_x) / (2.0 * self.h) + 1,
                                 int)
        self.search_grid = np.empty(self.max_list, object)

    def place_points(self):
        """Place points in a rectangle with a square spacing of size dx"""
        # Domain border
        outer_xmin = self.min_x
        outer_xmax = self.max_x
        # Domain for fluid
        inner_xmin = outer_xmin + 2 * self.h  # Inner xmin is point 0,0
        inner_xmax = outer_xmax - 2 * self.h  # Inner xmax is point 20,10
        self.inner_min_x = inner_xmin
        self.inner_max_x = inner_xmax

        # Add boundary particles
        # Maybe change to 2*dx for 3 boundary points
        for i in np.arange(inner_xmin[0] - 2*self.dx, inner_xmax[0] + 3*self.dx, self.dx):
            for j in np.arange(inner_xmin[1] - 2*self.dx, inner_xmax[1] + 3*self.dx, self.dx):
                if not inner_xmin[0] < i < inner_xmax[0] or not inner_xmin[1] < j < inner_xmax[1]:
                    x = np.array([i, j])
                    particle = particleClass.Particle(self, x)
                    particle.calc_index()
                    particle.boundary = True
                    self.particle_list.append(particle)

        # Add fluid particles
        for i in np.arange(inner_xmin[0] + self.dx, inner_xmax[0], self.dx):  # X [0+dx : 20-dx]
            for j in np.arange(inner_xmin[1] + self.dx, inner_xmin[1] + 2, self.dx):  # Y [0+dx : 0+2]
                    x = np.array([i, j])
                    particle = particleClass.Particle(self, x)
                    particle.calc_index()
                    self.particle_list.append(particle)

        for i in np.arange(inner_xmin[0] + self.dx, inner_xmin[0] + 3, self.dx):  # X [0+dx : 0+3]
            for j in np.arange(inner_xmin[1] + 2, inner_xmin[1] + 2 + 3, self.dx):  # Y [0+2+dx : 0+2+3]
                    x = np.array([i, j])
                    particle = particleClass.Particle(self, x)
                    particle.calc_index()
                    self.particle_list.append(particle)

    def allocate_to_grid(self):
        """Allocate all the points to a grid in order to aid neighbour searching"""
        for i in range(self.max_list[0]):
            for j in range(self.max_list[1]):
                self.search_grid[i, j] = []

        for cnt in self.particle_list:
            # Set the particle bucket index
            cnt.calc_index()
            # Keep in mind, list_num is bucket coordinates
            self.search_grid[cnt.list_num[0], cnt.list_num[1]].append(cnt)

    def update_search_grid(self, part):
        """
        Update the particle to the right bucket
        """
        if np.all(np.greater_equal(part.x, self.min_x)) and np.all(np.less_equal(part.x, self.max_x)):
            self.search_grid[part.list_num[0], part.list_num[1]].append(part)


    def neighbour_iterate(self, part):
        """Find all the particles within 2h of the specified particle"""
        # save neighbours (particles j) of particles i
        neighbours = []

        # Start of dynamic time step variables
        dt_cfl = np.inf
        dt_f = np.inf
        dt_A = np.sqrt(self.h / np.linalg.norm(part.a)) if np.all(part.a != 0) else np.inf

        # This code only returns the neighbours for the stencil
        if self.stencil:
            i, j = part.list_num   # Bucket the particle is in
            stenc = [[i, j], [i+1, j], [i+1, j+1], [i, j+1], [i-1, j+1]]  # Stencil shape
            for bucket in stenc:
                if 0 <= bucket[0] < self.max_list[0] and 0 <= bucket[1] < self.max_list[1]:
                    for other_part in self.search_grid[bucket[0], bucket[1]]:
                        if not part.id == other_part.id:
                            dn = part.x - other_part.x
                            dist = np.sqrt(np.sum(dn ** 2))
                            if dist < 2.0 * self.h:
                                neighbours.append(other_part)

            return neighbours

        for i in range(max(0, part.list_num[0] - 1),
                       min(part.list_num[0] + 2, self.max_list[0])):
            for j in range(max(0, part.list_num[1] - 1),
                           min(part.list_num[1] + 2, self.max_list[1])):
                for other_part in self.search_grid[i, j]:
                    if not part.id == other_part.id:
                        dist = np.linalg.norm(part.x - other_part.x)
                        if dist < 2.0 * self.h:
                            # Add neighbour to list
                            neighbours.append(other_part)
                            # Calculate dynamic time step
                            if np.all(part.v - other_part.v != 0):
                                ans1 = self.h / np.linalg.norm(part.v - other_part.v)
                                if ans1 < dt_cfl:
                                    dt_cfl = ans1
                            if np.all(other_part.a != 0):
                                ans2 = np.sqrt(self.h / np.linalg.norm(other_part.a))
                                if ans2 < dt_f:
                                    dt_f = ans2
                            denum = self.c0 * np.sqrt((part.rho / self.rho0) ** (self.gamma - 1))
                            ans3 = self.h / denum
                            if ans3 < dt_A:
                                dt_A = ans3

            # Change dt with dynamic time step
            self.dt = self.C_CFL * np.min([dt_cfl, dt_f, dt_A])

        return neighbours

    def smooth(self, part, neighbours):
        """
        Smoothing density/pressure field
        :param part: particle
        :param neighbours: neighbours of particle
        """
        w = self.W(0.0)
        num = w
        denum = w/part.rho
        for nei in neighbours:
            r = np.linalg.norm(part.x-nei.x)
            w = self.W(r)
            num += w
            denum += (w / nei.rho)

        part.rho = num / denum

    def W(self, r):
        """
        The smoothing kernel
        """
        q = r / self.h
        if q <= 1:
            w = 1 - 1.5*q**2 + 0.75*q**3
        else:
            w = 0.25 * (2-q)**3
        return self.constant_kernel * w

    def diff_W(self, r):
        q = r / self.h
        if q <= 1:
            dw = -3*q + (9 / 4) * q ** 2
        else:
            dw = -0.75 * (2 - q)**2
        return (self.constant_kernel / self.h) * dw

    def navier_cont(self, part, neighbours):
        # For the stencil scheme
        if self.stencil:

            near_wall = False

            for nei in neighbours:

                r_ij = part.x - nei.x
                dist = np.sqrt(np.sum(r_ij ** 2))
                v_ij = part.v - nei.v
                e_ij = r_ij / dist
                dWdr = self.diff_W(dist)

                # Calculate navier eqs for this particle to neighbour interaction
                nav_acc = - nei.m * ((part.P / part.rho ** 2) + (nei.P / nei.rho ** 2)) * dWdr * e_ij + \
                         self.mu * (nei.m * ((1 / part.rho ** 2) + (1 / nei.rho ** 2)) * dWdr * (v_ij / dist))
                nav_dens = nei.m * dWdr * np.dot(v_ij, e_ij)

                # Add forces to particle
                part.a = part.a + nav_acc
                part.D = part.D + nav_dens

                if not np.array_equal(part.list_num, nei.list_num):  # Only if not in same bucket to avoid duplication
                    # add corresponding force onto neigh from part
                    nei.a = nei.a - nav_acc  # Negative for neighbour particle
                    nei.D = nei.D + nav_dens

                if nei.boundary and not part.boundary:
                    near_wall = True

            if near_wall:
                for i in ['left', 'right', 'top', 'bottom']:
                    normal = np.array([1, 0])
                    dist = np.dot(part.x - self.inner_min_x, normal)
                    if i == 'right':
                        normal = np.array([-1, 0])
                        dist = np.dot(part.x - self.inner_max_x, normal)
                    if i == 'top':
                        normal = np.array([0, -1])
                        dist = np.dot(part.x - self.inner_max_x, normal)
                    if i == 'bottom':
                        normal = np.array([0, 1])
                        dist = np.dot(part.x - self.inner_min_x, normal)

                    # dist is positive scalar distance from wall
                    d_0 = 0.75 * self.dx
                    q = dist/d_0

                    if 0 < q < 1:
                        if q < 0.04:
                            q = 0.04

                        rho_rho0 = 1.02
                        Pref = ((self.rho0*self.c0**2)/self.gamma)*(rho_rho0**self.gamma-1)  # Reference pressure
                        da = normal * (Pref/self.rho0) * ((1/q)**4 - (1/q)**2)/dist

                        part.a = part.a + da

        else:
            near_wall = False

            for nei in neighbours:
                # Calculate distance between 2 points
                r = part.x - nei.x
                dist = np.sqrt(np.sum(r ** 2))
                # Calculate the difference of velocity
                v = part.v - nei.v
                # normal
                e = r / dist

                # Calculate diffW
                dWdr = self.diff_W(dist)
                part.a = part.a - nei.m * ((part.P / part.rho ** 2) + (nei.P / nei.rho ** 2)) * dWdr * e + \
                         self.mu * (nei.m * ((1 / part.rho ** 2) + (1 / nei.rho ** 2)) * dWdr * (v / dist))

                # Calculate the derivative of the density
                part.D = part.D + nei.m * dWdr * np.dot(v, e)

                if nei.boundary and not part.boundary:
                    near_wall = True

            if near_wall:
                for i in ['left', 'right', 'top', 'bottom']:
                    normal = np.array([1, 0])
                    dist = np.dot(part.x - self.inner_min_x, normal)
                    if i == 'right':
                        normal = np.array([-1, 0])
                        dist = np.dot(part.x - self.inner_max_x, normal)
                    if i == 'top':
                        normal = np.array([0, -1])
                        dist = np.dot(part.x - self.inner_max_x, normal)
                    if i == 'bottom':
                        normal = np.array([0, 1])
                        dist = np.dot(part.x - self.inner_min_x, normal)

                    # dist is positive scalar distance from wall
                    d_0 = 0.75 * self.dx
                    q = dist/d_0

                    if 0 < q < 1:
                        if q < 0.04:
                            q = 0.04

                        Pref = ((self.rho0*self.c0**2)/7.)*(1.02**7-1)

                        da = normal * (Pref/self.rho0) * ((1/q)**4 - (1/q)**2)/dist

                        part.a = part.a + da

    def forward_euler(self, particles, smooth=False):
        # Smoothing function
        if smooth:
            for part in particles:
                neis = self.neighbour_iterate(part)
                self.smooth(part, neis)

        # Perform navier stokes and continuity equation
        for part in particles:
            # Get neighbours of each particle
            neis = self.neighbour_iterate(part)
            self.navier_cont(part, neis)

        # Clear grid
        for i in range(self.max_list[0]):
            for j in range(self.max_list[1]):
                self.search_grid[i, j] = []

        # Perform forward euler update
        for part in particles:
            if not part.boundary:
                part.x = part.x + self.dt * part.v
                part.v = part.v + self.dt * part.a
                part.calc_index()
            part.rho = part.rho + self.dt * part.D

            # Set acceleration to 0 initially for next loop
            part.a = self.g
            # Set derivative of density to 0 initially for next loop
            part.D = 0

            # Fill grid and update pressure
            self.update_search_grid(part)
            part.update_P()

    def predictor_corrector(self, particles, smooth=False):
        for n in range(2):
            # Smoothing function
            if smooth:
                for part in particles:
                    neis = self.neighbour_iterate(part)
                    self.smooth(part, neis)

            # Perform navier stokes and continuity equation
            for part in particles:
                # Get neighbours of each particle
                neis = self.neighbour_iterate(part)
                self.navier_cont(part, neis)

            # Clear grid
            for i in range(self.max_list[0]):
                for j in range(self.max_list[1]):
                    self.search_grid[i, j] = []

            # Perform forward euler update
            if n == 0:  # half step
                for part in particles:
                    if not part.boundary:
                        # Save particle info
                        part.prev_x = part.x
                        part.prev_v = part.v
                        part.prev_rho = part.rho

                        part.x = part.x + 0.5 * self.dt * part.v
                        part.v = part.v + 0.5 * self.dt * part.a
                        part.calc_index()
                    part.rho = part.rho + 0.5 * self.dt * part.D

                    # Fill grid and update pressure
                    self.update_search_grid(part)
                    part.update_P()

                    # Set acceleration to 0 initially for next loop
                    part.a = self.g
                    # Set derivative of density to 0 initially for next loop
                    part.D = 0

            if n == 1:  # full-step
                for part in particles:
                    if not part.boundary:
                        x_ = part.prev_x + 0.5 * self.dt * part.v
                        part.x = 2 * x_ - part.prev_x
                        v_ = part.prev_v + 0.5 * self.dt * part.a
                        part.v = 2 * v_ - part.prev_v
                        part.calc_index()
                    rho_ = part.prev_rho + 0.5 * self.dt * part.D
                    part.rho = 2 * rho_ - part.prev_rho
                    # Fill grid and update pressure
                    self.update_search_grid(part)
                    part.update_P()

                    # Set acceleration to 0 initially for sum
                    part.a = self.g
                    # Set derivative of density to 0 initially for sum
                    part.D = 0

    def simulate(self, n=10):
        """
        :param self:
        :param n: save file every n dt
        :param dt:
        :param scheme: This is a time-stepping scheme - can either be forward euler or predictor corrector method
        :param smooth_t:
        :return:
        """
        t = self.t0
        time_array = [t]
        self.allocate_to_grid()
        cnt = 0
        filename = 'datafile_3.pkl'
        file = open(filename,'wb')
        # generate a progressbar
        widgets = ['Progress: ',Percentage(), ' ', Bar('$'),' ', Timer(),
                       ' ', ETA(), ' ', FileTransferSpeed()] 
        pbar = ProgressBar(widgets=widgets, maxval=int(self.t_max/self.dt)+1).start()
        i = 0
        count = 0
        
        while t < self.t_max:
            cnt = cnt + 1
            smooth = False
            # Smooth after some time steps
            if cnt % 10 == 0:
                smooth = True
            self.forward_euler(self.particle_list, smooth=smooth)
            t = t + self.dt
            # save file every n dt
            if cnt % n == 0:
                pickle.dump(self.particle_list, file, -1)
                pickle.dump(t, file)
                count += 2
                f = open('countnum.txt','w')
                f.write(str(count))
            time_array.append(t)
            i += 1
            pbar.update( i )
        f.close()
        pbar.finish()

    def load_file(self, count):
        p_list = []
        t_list = []
        i = 0
        filename = 'datafile_3.pkl'
        file = open(filename, 'rb')
        while i < count:
            if i % 2 == 0:
                # load particles data
                a = pickle.load(file)
                p_list.append(a)
            else:
                # load times data
                b = pickle.load(file)
                t_list.append(b)
            i += 1
        file.close()
        return p_list, t_list


    def write_to_file(self):
        with open('data.csv', 'w') as csvfile:
            fieldnames = ['X', 'Y', 'Boundary', 'Pressure', 'Velocity_X', 'Velocity_Y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for part in self.particle_list:
                writer.writerow({'X': str(part.x[0]),
                                 'Y': str(part.x[1]),
                                 'Boundary': str(part.boundary),
                                 'Pressure': str(part.P),
                                 'Velocity_X': str(part.v[0]),
                                 'Velocity_Y': str(part.v[1])})