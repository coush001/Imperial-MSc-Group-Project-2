# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:06:59 2018

@author: Dwyane Wade
"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation 

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)#xlim=(0, 2), ylim=(-4, 4))
line, = ax1.plot([], [], lw=2) 


def init():  
    line.set_data([], [])
    return line