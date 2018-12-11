"""SPH class to find nearest neighbours..."""
import numpy as np
import particle as particleClass


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

        # For predictor-corrector scheme
        self.C_CFL = 0.2

    def set_values(self, min_x=(0.0, 0.0), max_x=(20, 10), dx=0.02, h_fac=1.3, t0=0.0, t_max=0.001, dt=0, C_CFL=0.2):
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
        for i in np.arange(inner_xmin[0] - 3*self.dx, inner_xmax[0] + 3*self.dx, self.dx):
            for j in np.arange(inner_xmin[1] - 3*self.dx, inner_xmax[1] + 3*self.dx, self.dx):
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
            dw = (10 / (7 * np.pi * self.h ** 2)) * ((-2 * dist)/self.h**2 + (9/4) * dist**2 / self.h**3)
        if 1 <= q <= 2:
            dw = (10 / (7 * np.pi * self.h ** 2)) \
                 * 1/4 * (-3 * dist**2 / self.h**3 + 12 * dist/self.h**2 - 12/self.h**3)
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
            dw = (10 / (7 * np.pi * self.h ** 2)) * ((-2 * dist)/self.h**2 + (9/4) * dist**2 / self.h**3)
        if 1 <= q <= 2:
            dw = (10 / (7 * np.pi * self.h ** 2)) \
                 * 1/4 * (-3 * dist**2 / self.h**3 + 12 * dist/self.h**2 - 12/self.h**3)
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
            a += -(nei.m * ((part.P / part.rho ** 2) + (nei.P / nei.rho ** 2)) * self.grad_W(part, nei)) + \
                 self.mu * (nei.m * ((1 / part.rho ** 2) + (nei.P / nei.rho ** 2)) * self.diff_W(part, nei) * (v / dist)) + self.g
            # normal
            e = r / dist
            # Calculate the derivative of the density
            D += nei.m * np.dot(self.diff_W(part, nei) * v, e)
        return a, D

    def forward_euler(self, particles, t, dt, smooth=False):
        updated_particles = []
        for part in particles:
            # Get neighbours of each particle
            neis = self.neighbour_iterate(part)

            # Forward time step update
            x = part.x + self.dt * part.v
            v = part.v + self.dt * part.a
            rho = part.rho + self.dt * part.D

            # Smooth after some time steps
            if smooth:
                rho = self.smooth(part, neis)
            t = t + dt

            # Calculate a and D at time t
            a, D = self.navier_cont(part, neis)

            # If it is a boundary particle, then
            if part.boundary:
                x = 0
                v = 0

            # Set new attributes
            part.set_v(v)
            part.set_x(x)
            part.set_rho(rho)
            part.set_v(a)
            part.set_D(D)

            # Allocate grid points
            self.allocate_to_grid()

            # append to list
            updated_particles.append(part)

        return updated_particles


    def W(self, particle, other_particle):
        """
        The smoothing kernel
        """
        factor = 10 / (7*np.pi * self.h**2)
        r = particle.x - other_particle.x
        dist = np.sqrt(np.sum(r ** 2))
        q = dist / self.h
        w = 0
        if 0 <= q <= 1:
            w = 1 - 1.5*q**2 + 0.75*q**3
        if 1 <= q <= 2:
            w = 0.25 * (2-q)**3
        return factor * w

    def smooth(self, part, neighbours):
        """
        Smoothing density/pressure field
        :param part: particle
        :param neighbours: neighbours of particle
        """
        num = 0
        denum = 0
        for nei in neighbours:
            w = self.W(part, nei)
            num += w
            denum += (w / nei.rho)
        return num / denum

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

    def predictor_corrector(self, part, t0, t_max, neis, smooth_t=10):
        x_all = [part.x]
        v_all = [part.v]
        rho_all = [part.rho]
        a_all = [part.a]
        D_all = [part.D]
        t_all = [t0]
        x = part.x
        v = part.v
        rho = part.rho
        t = t0
        while t < t_max:
            # Allocate grid points
            self.allocate_to_grid()

            # Calculate a and D at time t
            part.set_v(v)
            part.set_x(x)
            part.set_rho(rho)
            a, D = self.navier_cont(part, neis)

            # Half-step
            x_h = x + 0.5 * self.var_dt(part, neis) * v
            v_h = v * 0.5 * self.var_dt(part, neis) * a
            rho_h = rho + 0.5 * self.var_dt(part, neis) * D

            # Calculate a and D at time t + 1/2
            part.set_v(v_h)
            part.set_x(x_h)
            part.set_rho(rho_h)
            a_h, D_h = self.navier_cont(part, neis)

            # Full-step part 1
            x_ = x + 0.5 * self.var_dt(part, neis) * v_h
            v_ = v + 0.5 * self.var_dt(part, neis) * a_h
            rho_ = rho + 0.5 * self.var_dt(part, neis) * D_h

            # Full-step part 2
            x = 2 * x_ - x
            v = 2 * v_ - v
            rho = 2 * rho_ - rho

            # Smooth after some time steps
            if t == t0 + smooth_t * self.var_dt(part, neis):
                rho = self.smooth(part, neis)
            t = t + self.dt

            # Set for boundaries
            if part.boundary:
                x = 0
                v = 0

            # Append variables to the lists
            x_all.append(x)
            v_all.append(v)
            rho_all.append(rho)
            a_all.append(a)
            D_all.append(D)
            t_all.append(t)

        return t_all, x_all, v_all, rho_all, a_all, D_all

    def simulate(self, dt, smooth_t=10):
        # We are returning a list of particles per time step in a list of lists
        t = self.t0
        particles_times = []
        time_array = [t]
        particles = self.particle_list
        while t < self.t_max:
            print("current time", t)
            smooth = False
            # Smooth after some time steps
            if t == self.t0 + smooth_t * dt:
                smooth = True
            particles = self.forward_euler(particles, t, dt, smooth=smooth)
            print("random particle")
            print("acc", particles[300].a)
            print("vel", particles[300].v)
            print("rho", particles[300].rho)
            print("D", particles[300].D)
            print("x", particles[300].x)

            t = t + dt
            particles_times.append(particles)

        # List of particles in each time step
        result = np.array(particles_times).T.tolist()

        # Return particles and time steps
        return result, time_array


"""Create a single object of the main SPH type"""
domain = SPH_main()

"""Calls the function that sets the simulation parameters"""
domain.set_values()
"""Initialises the search grid"""
domain.initialise_grid()
print("initialised grid")


"""Places particles in a grid over the entire domain - In your code you will need to place the fluid particles in only the appropriate locations"""
domain.place_points(domain.min_x, domain.max_x)
print("placed points")
"""This is only for demonstration only - In your code these functions will need to be inside the simulation loop"""
"""This function needs to be called at each time step (or twice a time step if a second order time-stepping scheme is used)"""
domain.allocate_to_grid()
print("allocated to grid")

"""This example is only finding the neighbours for a single partle - this will need to be inside the simulation loop and will need to be called for every particle"""
# domain.neighbour_iterate(domain.particle_list[100])

domain.simulate(domain.dt)

