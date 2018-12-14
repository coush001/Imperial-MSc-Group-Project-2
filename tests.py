import sph_stub as sphClass
import numpy as np
import particle as particleClass

# Initialise grid
domain = sphClass.SPH_main()
domain.set_values(max_x=(10, 7), dx=0.8, t_max=1)
domain.initialise_grid()
domain.place_points()
domain.simulate(domain.forward_euler)


def test_mass_not_zero():
    mass = np.array([particle.m for particle in domain.particle_list])
    assert(max(mass) > 0)


def test_velocity_is_less_than_c0():  # Check no particle exceeds speed of sound
    velocities = np.array([particle.v for particle in domain.particle_list])
    assert(np.all(velocities < domain.c0))


def test_density_does_not_change_much():  # Check no density exceeds 1.5 * initial
    densities = np.array([particle.rho for particle in domain.particle_list])
    factor = max(densities) / domain.rho0
    assert(factor < 1.5)


def test_particles_dont_leak():
    locations = np.array([particle.x for particle in domain.particle_list])
    assert(max(locations[:, 0]) <= domain.max_x[0])
    assert(max(locations[:, 1]) <= domain.max_x[1])
    assert(min(locations[:, 0]) >= domain.max_x[0])
    assert(min(locations[:, 1]) >= domain.min_x[1])

