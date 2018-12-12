# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:54:40 2018

@author: Dwyane Wade
"""
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sph_stub as sphClass
import matplotlib
import numpy as np


domain = sphClass.SPH_main()
p_list, t_lsit = domain.load_file()

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
ax1.set_xlim(-2, 22)
ax1.set_ylim(-2, 12)
# ax1.scatter(x_data[0], y_data[0], 'b.', )
moving_part = ax1.scatter(x_data[0][0], x_data[0][1])
# ax1.scatter(x_boundary[0], y_boundary[0],)
boundary = ax1.scatter(x_boundary[0][0], x_boundary[0][1])
# moving_part, = ax1.scatter([], [], 'b', )
time_text = ax1.text(0.7, 0.8, '', transform=ax1.transAxes)


def animate(i):
    pre = (np.array(pressure[i])-min(pressure[i]))/(max(pressure[i])-min(pressure[i]))
    color = []
    for j in range(len(pre)):
        color.append([0.5*pre[j]+0.2, 0.5*pre[j]+0.2, 0.5*pre[j]+0.5])
    color = np.array(color)
    moving_part.set_color(color)
    moving_part.set_offsets(x_data[i])
    boundary.set_offsets(x_boundary[i])
    time_text.set_text('time = {0:.3f}'.format(i*t_lsit[1]))
    return moving_part, time_text


anim = animation.FuncAnimation(fig, animate, frames=len(t_lsit),
                               interval=100, blit=True) #  init_func=init

plt.show()

ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps = 15)
anim.save("video.mp4",writer = writer)
print("animation output done")
"""
print("animation done")
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subscatter(111)
line, = ax1.scatter([], [], 'b',)

def make_frame(t):
    i = int(t/0.001)
    line.set_data(x_data[i], y_data[i])
    ax1.clear()
    ax1.scatter(xf, Cex, 'k', lw=3, label='exact ss solution')
    ax1.set_title("SPH practice", fontsize=16)
    ax1.scatter(x, C[:, i])
    ax1.set_ylim(-0.1, 1.1)
    time_text = ax1.text(0.78, 0.95, '', transform=ax1.transAxes)
    time_text.set_text('time = {0:.3f}'.format(t))
    return mplfig_to_npimage(fig)

duration = 1
animation = VideoClip(make_frame, duration=duration)
print("animation done")
animation.write_gif('togif.gif', fps=20)
"""