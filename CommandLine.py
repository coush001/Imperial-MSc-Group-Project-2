import argparse
import sph_stub as sphClass
import numpy as np
import particle as particleClass


parser = argparse.ArgumentParser()

parser.add_argument('t_max', help="Simulation end time", type=float)
parser.add_argument('dx', help="Initial Particle Spacing", type=float)
parser.add_argument('-x', '--xdomain', help="X Fluid Domain range, provide xmin xmax", nargs='+', type=int, default=[0, 20])
parser.add_argument('-y', '--ydomain', help="Y Fluid Domain range, provide ymin ymax", nargs='+', type=int, default=[0, 10])
parser.add_argument('-m', '--movie', help="Save to MP4, or just show?", default=False, action='store_true')
parser.add_argument('-f', '--frames', help="Save data every nth frame", default=25, type=int)


args = parser.parse_args()

print(args)

########
# Convert args to easy read variables
t_max = args.t_max
dx = args.dx
min_x = [args.xdomain[0], args.ydomain[0]]  # min x , min y
max_x = [args.xdomain[1], args.ydomain[1]]  # max x , max y
movie = args.movie
framerate = args.frames

#######
# ASSERT USER HAS ENTERED VALID MODEL PARAMETERS
if t_max > 50 or dx < 0.1:
    print('!Warning this simulation will take longer than 2 hours!')

assert t_max > 0, "Please use positive t_max"
assert dx > 0, "Please use positive dx"
assert min_x[0] < max_x[0], "Please review your xdomain input, Currently xmin > xmax"
assert min_x[1] < max_x[1], "Please review your ydomain input, Currently ymin > ymax"

#######


# create class instance with command line inputs
domain = sphClass.SPH_main()
domain.set_values(min_x=min_x, max_x=max_x, dx=dx, t_max=t_max)
domain.initialise_grid()
domain.place_points()
domain.simulate()

