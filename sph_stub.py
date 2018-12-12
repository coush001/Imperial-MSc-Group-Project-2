"""SPH class to find nearest neighbours..."""
import numpy as np
import particle as particleClass
import copy
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
        self.g = np.array([0.0, -9.81])  # m^2 s^-2
        self.c0 = 20  # m s ^-1
        self.gamma = 7

        # For predictor-corrector scheme
        self.C_CFL = 0.2

    def set_values(self, min_x=(0.0, 0.0), max_x=(10, 7), dx=0.5, h_fac=1.3, t0=0.0, t_max=0.5, dt=0, C_CFL=0.2):
        """Set simulation parameters."""

        self.min_x[:] = min_x
        self.max_x[:] = max_x
        self.dx = dx
        self.h_fac = h_fac
        self.h = self.dx * self.h_fac

        self.dt = 0.1 * (self.h / self.c0) if dt == 0 else dt
        self.t0 = t0
        self.t_max = t_max

        self.C_CFL = C_CFL

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
            # print("bucket", cnt.list_num[0], cnt.list_num[1])
            # print("shape of search grid", self.min_x, self.max_x)
            # print("particle x", cnt.x, cnt.boundary)
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
                    if not part.id == other_part.id:
                        dn = part.x - other_part.x
                        dist = np.sqrt(np.sum(dn ** 2))
                        if dist < 2.0 * self.h:
                            neighbours.append(other_part)
                            # print("id:", other_part.id, "dn:", dn)
        return neighbours

    def diff_W(self, r):
        q = r / self.h
        c = 10 / (7 * np.pi * self.h ** 3)
        if q <= 1:
            dw = -3*q + (9 / 4) * q ** 2
        else:
            dw = -0.75 * (2 - q)**2
        return c * dw

    def grad_W(self, part, other_part):
        dn = part.x - other_part.x  # dn is r_ij (vector)
        dist = np.sqrt(np.sum(dn ** 2))  # dist is |r_ij| (scalar)
        # print("dn and dist", dn, dist)
        # print("parts id", part.id, other_part.id)
        e_ij = dn / dist
        dw = self.diff_W(part, other_part)
        return dw * e_ij

    def navier_cont(self, part, neighbours):
        # Set acceleration to 0 initially for sum
        part.a = self.g
        # Set derivative of density to 0 initially for sum
        part.D = 0
        # Repulsive force
        rf = 0
        for nei in neighbours:
            # Calculate distance between 2 points
            r = part.x - nei.x
            dist = np.sqrt(np.sum(r ** 2))

            # Calculate the difference of velocity
            v = part.v - nei.v
            v_mod = np.sqrt(np.sum(v ** 2))

            #if not part.boundary and nei.boundary:
            #    # Repulsive force calculation from paper
                #  http://www.wseas.us/e-library/conferences/2011/Corfu/CUTAFLUP/CUTAFLUP-15.pdf
            #    k = 0.01 * part.B() * self.gamma / self.rho0
            #    rf = k * self.repulsive_force_psi(part, nei) * (r / (dist ** 2))
            # Calculate acceleration of particle
            e = r / dist
            dWdr = self.diff_W(dist)

            part.a = part.a - nei.m * ((part.P / part.rho ** 2) + (nei.P / nei.rho ** 2)) * dWdr*e + \
                 self.mu * (nei.m * ((1 / part.rho ** 2) + (1 / nei.rho ** 2)) * dWdr * (v / dist))

            part.D = part.D + nei.m * dWdr * np.dot(v, e)
            # normal

            # Calculate the derivative of the density

    def repulsive_force_psi(self, part, other_part, kj=0.5, shao=False):
        psi = 0
        r = part.x - other_part.x
        dist = np.sqrt(np.sum(r ** 2))
        q = dist / self.h
        if shao:
            psi = self.f(q)
        elif 0 <= q <= kj:
            num = np.exp(-3*q**2) - np.exp(-3*kj**2)
            denum = 1 - np.exp(-3*kj**2)
            psi = num / denum
        return psi

    def f(self, q):
        f = 0
        if 0 <= q <= (2/3):
            f = 2 / 3
        if (2 / 3) < q <= 1:
            f = (2*q - 1.5*q**2)
        if 1 < q <= 2:
            f = 0.5*(2-q)**2
        return 10*f

    def forward_euler(self, particles, t, dt, smooth=False):
        #updated_particles = []

        if smooth:
            for part in particles:
                neis = self.neighbour_iterate(part)
                self.smooth(part, neis)

        for part in particles:
            # Get neighbours of each particle
            neis = self.neighbour_iterate(part)



            self.navier_cont(part, neis)



        for i in range(self.max_list[0]):
            for j in range(self.max_list[1]):
                self.search_grid[i, j] = []

        for part in particles:
            # Forward time step update
            if not part.boundary:
                part.x = part.x + self.dt * part.v
                part.v = part.v + self.dt * part.a
                part.calc_index()
            part.rho = part.rho + self.dt * part.D

            self.search_grid[part.list_num[0], part.list_num[1]].append(part)
            part.update_P()


    def W(self, r):
        """
        The smoothing kernel
        """
        factor = 10 / (7*np.pi * self.h**2)
        q = r / self.h
        w = 0
        if q <= 1:
            w = 1 - 1.5*q**2 + 0.75*q**3
        else:
            w = 0.25 * (2-q)**3
        return factor * w

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
            r=np.linalg.norm(part.x-nei.x)
            w = self.W(r)
            num += w
            denum += (w / nei.rho)
        self.rho = num / denum

    def var_dt(self, part, neis):
        dt_cfl = np.inf
        dt_f = np.inf
        dt_A = np.inf
        for nei in neis:
            v = part.v - nei.v
            v_abs = np.sqrt(np.sum(v ** 2))
            ans1 = self.h / v_abs
            if ans1 < dt_cfl:
                dt_cfl = ans1

            a = part.a - nei.a
            a_abs = np.sqrt(np.sum(a ** 2))
            ans2 = np.sqrt(self.h / a_abs)
            if ans2 < dt_f:
                dt_f = ans2

            denum = self.c0 * np.sqrt((part.rho / self.rho0) ** (self.gamma - 1))
            ans3 = self.h / denum
            if ans3 < dt_A:
                dt_A = ans3

            return self.C_CFL * np.min(dt_cfl, dt_f, dt_A)

    def predictor_corrector(self, particles, t, dt, smooth=False):
        updated_particles = []
        for part in particles:
            # Get neighbours of each particle
            neis = self.neighbour_iterate(part)

            x = part.x
            v = part.v
            rho = part.rho

            # Forward time step update
            if not part.boundary:
                # Half-step
                x_h = part.x + 0.5 * self.var_dt(part, neis) * part.v
                v_h = part.v * 0.5 * self.var_dt(part, neis) * part.a
                rho_h = part.rho + 0.5 * self.var_dt(part, neis) * part.D

                prev_x = part.x
                prev_v = part.v
                prev_rho = part.rho

                # Calculate a and D at time t + 1/2
                part.set_v(v_h)
                part.set_x(x_h)
                part.set_rho(rho_h)
                a_h, D_h = self.navier_cont(part, neis)

                # Full-step part 1
                x_ = prev_x + 0.5 * self.var_dt(part, neis) * v_h
                v_ = prev_v + 0.5 * self.var_dt(part, neis) * a_h
                rho_ = prev_rho + 0.5 * self.var_dt(part, neis) * D_h

                # Full-step part 2
                x = 2 * x_ - prev_x
                v = 2 * v_ - prev_v
                rho = 2 * rho_ - prev_rho

            if part.boundary:
                # Half-step
                rho_h = part.rho + 0.5 * self.var_dt(part, neis) * part.D

                prev_rho = part.rho

                # Calculate a and D at time t + 1/2
                part.set_rho(rho_h)
                a_h, D_h = self.navier_cont(part, neis)

                # Full-step part 1
                rho_ = prev_rho + 0.5 * self.var_dt(part, neis) * D_h

                # Full-step part 2
                rho = 2 * rho_ - prev_rho

                # Set v to zero
                v = np.zeros(2)

            # Smooth after some time steps
            if smooth:
                rho = self.smooth(part, neis)
            t = t + dt

            # Set new attributes
            part.set_v(v)
            part.set_x(x)
            part.set_rho(rho)

            # Calculate a and D at time t
            self.navier_cont(part, neis)

            # Allocate grid points
            self.allocate_to_grid()

            # append to list
            updated_particles.append(part)

        return updated_particles

    def simulate(self, dt, scheme, smooth_t=10):
        """
        :param self:
        :param dt:
        :param scheme: This is a time-stepping scheme - can either be forward euler or predictor corrector method
        :param smooth_t:
        :return:
        """
        # We are returning a list of particles per time step in a list of lists
        t = self.t0
        time_array = [t]
        self.allocate_to_grid()
        cnt = 0
        while t < self.t_max:
            cnt=cnt+1
            smooth = False
            # Smooth after some time steps
            if cnt%10 == 0:
                smooth = True
            scheme(self.particle_list, t, dt, smooth=smooth)
            print("Time", t)

            t = t + dt
            time_array.append(t)

    def output_particle(self):
        x_value_bound = []
        y_value_bound = []
        x_value = []
        y_value = []
        for part in self.particle_list:
            if part.boundary:
                x_value_bound.append(part.x[0])
                y_value_bound.append(part.x[1])
            else:
                x_value.append(part.x[0])
                y_value.append(part.x[1])
        return [x_value, y_value, x_value_bound, y_value_bound]

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


