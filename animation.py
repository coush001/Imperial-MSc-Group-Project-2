# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:54:40 2018

@author: Dwyane Wade
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sph_stub as sphClass

domain = sphClass.SPH_main()
p_list, t_lsit = domain.load_file()

x_data = []
y_data = []
x_boundary = []
y_boundary = []
x_small = []
y_small = []

for i in p_list:
    x_small = []
    y_small = []
    x_bound_small = []
    y_bound_small = []
    for j in i:
        if j.boundary is False:
            x_small.append(j.x[0])
            y_small.append(j.x[1])
        else:
            x_bound_small.append(j.x[0])
            y_bound_small.append(j.x[1])
    x_data.append(x_small)
    y_data.append(y_small)
    x_boundary.append(x_bound_small)
    y_boundary.append(y_bound_small)

print("get data done")


fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(-1, 11)
#ax1.set_ylim(-1, 6)
# ax1.plot(x_data[0], y_data[0], 'b.', )
moving_part, = ax1.plot([], [], 'b.', )
ax1.plot(x_boundary[0], y_boundary[0], 'r.', )
boundary, = ax1.plot([], [], 'r.',)
# moving_part, = ax1.plot([], [], 'b', )
time_text = ax1.text(0.1, 0.1, '', transform=ax1.transAxes)


def init():
    moving_part.set_data([], [])
    time_text.set_text('')
    boundary.set_data([], [])
    return moving_part, time_text

def animate(i):
    moving_part.set_data(x_data[i], y_data[i])
    boundary.set_data(x_boundary[i], y_boundary[i])
    time_text.set_text('time = {0:.3f}'.format(i*t_lsit[1]))
    print(i, x_data[i])
    return moving_part, time_text


anim = animation.FuncAnimation(fig, animate, frames=len(t_lsit),
                               interval=100, blit=True, init_func=init)
print("animation done")
plt.show()