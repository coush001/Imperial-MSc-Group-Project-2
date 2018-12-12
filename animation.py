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
ax1.set_xlim(-2, 12)
ax1.set_ylim(-2, 9)
# ax1.plot(x_data[0], y_data[0], 'b.', )
moving_part, = ax1.plot([], [], 'b.', )
ax1.plot(x_boundary[0], y_boundary[0], 'r.', )
boundary, = ax1.plot([], [], 'r.',)
# moving_part, = ax1.plot([], [], 'b', )
time_text = ax1.text(0.7, 0.8, '', transform=ax1.transAxes)


def init():
    moving_part.set_data([], [])
    time_text.set_text('')
    boundary.set_data([], [])
    return moving_part, time_text

def animate(i):
    moving_part.set_data(x_data[i], y_data[i])
    boundary.set_data(x_boundary[i], y_boundary[i])
    time_text.set_text('time = {0:.3f}'.format(i*t_lsit[1]))
    return moving_part, time_text


anim = animation.FuncAnimation(fig, animate, frames=len(t_lsit),
                               interval=32.5, blit=True, init_func=init)

plt.show()

ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps = 15)
anim.save("video.mp4",writer = writer)
print("animation output done")
"""
print("animation done")
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
line, = ax1.plot([], [], 'b',)

def make_frame(t):
    i = int(t/0.001)
    line.set_data(x_data[i], y_data[i])
    ax1.clear()
    ax1.plot(xf, Cex, 'k', lw=3, label='exact ss solution')
    ax1.set_title("SPH practice", fontsize=16)
    ax1.plot(x, C[:, i])
    ax1.set_ylim(-0.1, 1.1)
    time_text = ax1.text(0.78, 0.95, '', transform=ax1.transAxes)
    time_text.set_text('time = {0:.3f}'.format(t))
    return mplfig_to_npimage(fig)

duration = 1
animation = VideoClip(make_frame, duration=duration)
print("animation done")
animation.write_gif('togif.gif', fps=20)
"""