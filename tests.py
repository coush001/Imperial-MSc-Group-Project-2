import sph_stub as sphClass
import numpy as np
import particle as particleClass

# Initialise grid
domain = sphClass.SPH_main()
domain.set_values()
domain.initialise_grid()
domain.place_points(domain.min_x, domain.max_x)
domain.allocate_to_grid()



def test_velocity_is_less_than_c0():
    velocities = np.array([particle.v for particle in domain.particle_list])
    domain.simulate(domain.dt, domain.forward_euler)
    assert(np.all(velocities < domain.c0))
