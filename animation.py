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


def get_data(p_list):
    x_data = []
    x_boundary = []
    pressure = []
    velocity = []
    rho = []
    acceleration = []
    for i in p_list:
        x_small = []
        x_bound_small = []
        pressure_small = []
        velocity_small = []
        rho_small = []
        acceleration_small = []
        for j in i:
            if j.boundary is False:
                x_small.append([j.x[0], j.x[1]])
                pressure_small.append(j.P)
                velocity_small.append(np.linalg.norm(j.v))
                acceleration_small.append(np.linalg.norm(j.a))
                rho_small.append(j.rho)
            else:
                x_bound_small.append([j.x[0], j.x[1]])
        x_data.append(x_small)
        x_boundary.append(x_bound_small)
        pressure.append(pressure_small)
        velocity.append(velocity_small)
        rho.append(rho_small)
        acceleration.append(acceleration_small)
    print("get data done")
    return x_data, x_boundary, pressure, velocity, rho, acceleration

def get_countnum():
    f = open('countnum.txt', 'r')
    count = int(f.read())
    f.close()
    return count


def get_color(data_type, i):
    """
    choose type of data to get the color
    ex: pressure, velocity, rho, acceleration
    iis the animation time
    """
    pre = (np.array(data_type[i])-min(data_type[i]))/(max(data_type[i])-min(data_type[i]))
    color = []
    colorbound = []
    for j in range(len(pre)):
        color.append([pre[j], 0.6, 1-pre[j]])
    for j in range(len(x_boundary[i])):
        colorbound.append([0, 0, 0])
    return np.array(color), np.array(colorbound)


domain = sphClass.SPH_main()
p_list, t_lsit = domain.load_file(get_countnum())
x_data, x_boundary, pressure, velocity, rho, acceleration = get_data(p_list)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax1.set_xlim(-2, 12)
ax1.set_ylim(-2, 9)
moving_part = ax1.scatter(x_data[0][0], x_data[0][1])
boundary = ax1.scatter(x_boundary[0][0], x_boundary[0][1])
time_text = ax1.text(0.7, 0.8, '', transform=ax1.transAxes)


def animate(i):
    color, colorbound = get_color(acceleration, i) # pressure can be change to velocity, rho ,acceleration
    moving_part.set_color(color)
    moving_part.set_offsets(x_data[i])
    boundary.set_offsets(x_boundary[i])
    boundary.set_color(colorbound)
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
