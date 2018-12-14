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
    for i in p_list:
        x_small = []
        x_bound_small = []
        pressure_small = []
        velocity_small = []
        rho_small = []
        for j in i:
            if j.boundary is False:
                x_small.append([j.x[0], j.x[1]])
                pressure_small.append(j.P)
                velocity_small.append(np.linalg.norm(j.v))
                rho_small.append(j.rho)
            else:
                x_bound_small.append([j.x[0], j.x[1]])
        x_data.append(x_small)
        x_boundary.append(x_bound_small)
        pressure.append(pressure_small)
        velocity.append(velocity_small)
        rho.append(rho_small)
    print("get data done")
    return x_data, x_boundary, pressure, velocity, rho

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
    pre = np.array(abs(np.array(data_type[i])-min(data_type[i]))/(max(np.array(data_type[i]))-min(np.array(data_type[i]))))
    color = []
    colorbound = []
    for j in range(len(pre)):
        color.append([pre[j], 0.6, 1-pre[j]])
    for j in range(len(x_boundary[i])):
        colorbound.append([0, 0, 0])
    return np.array(color), np.array(colorbound)


def animate_pressure(i):
    color, colorbound = get_color(pressure, i) # pressure can be change to velocity, rho ,acceleration
    moving_part1.set_color(color)
    moving_part1.set_offsets(x_data[i])
    boundary1.set_offsets(x_boundary[i])
    boundary1.set_color(colorbound)
    time_text1.set_text('time = {0:.3f}'.format(i*t_list[0]))
    return moving_part1, time_text1

def animate_velocity(i):
    color, colorbound = get_color(velocity, i) # pressure can be change to velocity, rho ,acceleration
    moving_part2.set_color(color)
    moving_part2.set_offsets(x_data[i])
    boundary2.set_offsets(x_boundary[i])
    boundary2.set_color(colorbound)
    time_text2.set_text('time = {0:.3f}'.format(i*t_list[0]))
    return moving_part2, time_text2

def animate_rho(i):
    color, colorbound = get_color(rho, i) # pressure can be change to velocity, rho ,acceleration
    moving_part3.set_color(color)
    moving_part3.set_offsets(x_data[i])
    boundary3.set_offsets(x_boundary[i])
    boundary3.set_color(colorbound)
    time_text3.set_text('time = {0:.3f}'.format(i*t_list[0]))
    return moving_part3, time_text3


domain = sphClass.SPH_main()
p_list, t_list = domain.load_file(get_countnum())
x_data, x_boundary, pressure, velocity, rho = get_data(p_list)


fig1 = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(111)

# animate pressure relate animation
ax1.set_xlim(-2, 22)
ax1.set_ylim(-2, 12)
ax1.set_title("Pressure")
moving_part1 = ax1.scatter(x_data[0][0], x_data[0][1])
boundary1 = ax1.scatter(x_boundary[0][0], x_boundary[0][1])
time_text1 = ax1.text(0.7, 0.8, '', transform=ax1.transAxes)
plt.show()
anim1 = animation.FuncAnimation(fig1, animate_pressure, frames=len(t_list),
                               interval=1000*t_list[0], blit=True) #  init_func=init

ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps = 15)
anim1.save("Pressure_video.mp4",writer = writer)

# animate velocity relate animation
fig2 = plt.figure(figsize=(12, 6))
ax2 = plt.subplot(111)

ax2.set_xlim(-2, 22)
ax2.set_ylim(-2, 12)
ax2.set_title("velocity")
moving_part2 = ax2.scatter(x_data[0][0], x_data[0][1])
boundary2 = ax2.scatter(x_boundary[0][0], x_boundary[0][1])
time_text2 = ax2.text(0.7, 0.8, '', transform=ax2.transAxes)
anim2 = animation.FuncAnimation(fig2, animate_velocity, frames=len(t_list),
                               interval=1000*t_list[0], blit=True) #  init_func=init
ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps = 15)
anim2.save("Velocity_video.mp4",writer = writer)

# animate rho relate animation
fig3 = plt.figure(figsize=(12, 6))
ax3 = plt.subplot(111)
ax3.set_title("rho")
ax3.set_xlim(-2, 22)
ax3.set_ylim(-2, 12)
moving_part3 = ax3.scatter(x_data[0][0], x_data[0][1])
boundary3 = ax3.scatter(x_boundary[0][0], x_boundary[0][1])
time_text3 = ax3.text(0.7, 0.8, '', transform=ax3.transAxes)
anim = animation.FuncAnimation(fig3, animate_rho, frames=len(t_list),
                               interval=1000*t_list[0], blit=True) #  init_func=init
ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps = 15)
anim.save("Rho_video.mp4",writer = writer)
