"""SPH class to find nearest neighbours..."""

import matplotlib.pyplot as plt
from matplotlib import animation
import sph_stub as sphClass
import numpy as np

# Initialise grid
domain = sphClass.SPH_main()
domain.set_values()
domain.initialise_grid()
domain.place_points()
domain.allocate_to_grid()

domain.simulate()

print("simulation done")
# x_data = []
# y_data = []
# x_small = []
# y_small = []
# for i in result_list:
#     x_small = []
#     y_small = []
#     for j in i:
#         if j.boundary is False:
#             x_small.append(j.x[0])
#             y_small.append(j.x[1])
#     x_data.append(x_small)
#     y_data.append(y_small)

print("output done")

x = []
y = []
x_bound = []
y_bound = []
for i in domain.particle_list:
    if i.boundary:
        x_bound.append(i.x[0])
        y_bound.append(i.x[1])
    else:
        x.append(i.x[0])
        y.append(i.x[1])

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax1.plot(x, y, 'b.', )
ax1.plot(x_bound, y_bound, 'r.', )

# moving_part, = ax1.plot([], [], 'b.', )
# # moving_part, = ax1.plot([], [], 'b', )
# ax1.plot(a[2], a[3], 'r.', )
# time_text = ax1.text(0.1, 0.1, '', transform=ax1.transAxes)
#
# def init():
#     moving_part.set_data([], [])
#     time_text.set_text('')
#     return moving_part, time_text
#
# def animate(i):
#     moving_part.set_data(x_data[i], y_data[i])
#     time_text.set_text('time = {0:.3f}'.format(i*domain.dt))
#     return moving_part, time_text
#
#
# m = np.array(x_data[-1]) - np.array(x_data[0])
# anim = animation.FuncAnimation(fig, animate, frames=len(t_array),
#                                interval=100, blit=True, init_func=init)
# print("animation done")
plt.show()

# ax1.set_xlim(-1, 21)
# ax1.set_ylim(-1, 11)
# domain.write_to_file()
"""
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
line, = ax1.plot([], [], lw=2)
"""