import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


def f(x, y):
    return np.exp(x) + np.sin(y)

x = np.linspace(0, 1, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

frames = []

for i in range(10):
    x       += 1
    curVals  = f(x, y)
    vmax     = np.max(curVals)
    vmin     = np.min(curVals)
    levels   = np.linspace(vmin, vmax, 200, endpoint = True)
    frame    = ax1.contourf(curVals, vmax=vmax, vmin=vmin, levels=levels)
    cbar     = fig.colorbar(frame, cax=ax2) # Colorbar does not update
    frames.append(frame.collections)

ani = animation.ArtistAnimation(fig, frames, blit=False)

plt.show()