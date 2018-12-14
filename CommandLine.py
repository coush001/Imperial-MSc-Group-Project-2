import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sph_stub as sphClass
import matplotlib
import numpy as np
import sys

# Process arguments
parser = argparse.ArgumentParser()

parser.add_argument('t_max', help="Simulation end time", type=float)
parser.add_argument('dx', help="Initial Particle Spacing", type=float)
parser.add_argument('-x', '--xdomain', help="X Fluid Domain range, input: xmin xmax", nargs='+', type=int, default=[0, 20])
parser.add_argument('-y', '--ydomain', help="Y Fluid Domain range, input: ymin ymax", nargs='+', type=int, default=[0, 10])
parser.add_argument('-m', '--movie', help="Save to MP4, or just show?", default=True, action='store_true')
parser.add_argument('-f', '--frames', help="Save data every nth frame", default=5, type=int)
parser.add_argument('-s', '--scheme', help="Time step scheme, choose 'fe' for forward euler or 'pc' for predictor corrector",
                    choices=['fe', 'pc'], default='fe', type=str)


args = parser.parse_args()

######

# Check user has expected sim params
print('\n \nYou are running the simulation with the following parameters: \n', args)
text = input(" \n Do you wish to procede? (y/n) \n")

while text not in ['y', 'yes', 'Y', 'n', 'no', 'N', 'No']:
    print('input not recognised, please try again')
    text = input("\n Do you wish to procede? (y/n) \n")

if text in ['n', 'no', 'N', 'No']:
    sys.exit()

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


# Initialise grid
domain = sphClass.SPH_main()
domain.set_values(min_x=min_x, max_x=max_x, t_max=t_max, dx=dx)
domain.initialise_grid()
domain.place_points()
domain.allocate_to_grid()

if args.scheme == 'fe':
    count = domain.simulate(domain.forward_euler, n=framerate)
else:
    count = domain.simulate(domain.predictor_corrector, n=framerate)


if movie: # Create movie
    f = open('countnum.txt', 'r')
    count = int(f.read())
    f.close()

    domain = sphClass.SPH_main()
    p_list, t_lsit = domain.load_file(count)

    x_data = []
    x_boundary = []
    x_small = []
    pressure = []
    for i in p_list:
        x_small = []
        x_bound_small = []
        pressure_small = []
        for j in i:
            if j.boundary is False:
                x_small.append([j.x[0], j.x[1]])
                pressure_small.append(j.P)
            else:
                x_bound_small.append([j.x[0], j.x[1]])
        x_data.append(x_small)
        x_boundary.append(x_bound_small)
        pressure.append(pressure_small)
    print("get data done")

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(-2, args.xdomain[1]+2)
    ax1.set_ylim(-2, args.ydomain[1]+2)
    # ax1.scatter(x_data[0], y_data[0], 'b.', )
    moving_part = ax1.scatter(x_data[0][0], x_data[0][1])
    # ax1.scatter(x_boundary[0], y_boundary[0],)
    boundary = ax1.scatter(x_boundary[0][0], x_boundary[0][1])
    # moving_part, = ax1.scatter([], [], 'b', )
    time_text = ax1.text(0.7, 0.8, '', transform=ax1.transAxes)


    def animate(i):
        pre = (np.array(pressure[i]) - min(pressure[i])) / (max(pressure[i]) - min(pressure[i]))
        color = []
        colorbound = []
        for j in range(len(pre)):
            color.append([pre[j], 0.6, 1 - pre[j]])
        for j in range(len(x_boundary[i])):
            colorbound.append([0, 0, 0])
        color = np.array(color)
        moving_part.set_color(color)
        moving_part.set_offsets(x_data[i])
        boundary.set_offsets(x_boundary[i])
        boundary.set_color(colorbound)
        time_text.set_text('time = {0:.3f}'.format(i * t_lsit[1]))
        return moving_part, time_text


    anim = animation.FuncAnimation(fig, animate, frames=len(t_lsit),
                                   interval=100, blit=True)  # init_func=init

    plt.show()

    # ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
    # matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
    # writer = animation.FFMpegWriter(fps=15)
    # anim.save("video.mp4", writer=writer)
    # print("animation output done")
