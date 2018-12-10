import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# physical parameter
kappa = 1.

# numerical parameters (the spatial mesh)
Lx = 1
Ly = 1
Nx = 50
Ny = 50
dx = Lx / (Nx - 1) # since Nx 'points' means Nx-1 'intervals'
dy = Ly / (Ny - 1)

# read the docs to see the ordering that mgrid gives us
X, Y = np.mgrid[0:Nx:1, 0:Ny:1]
X = dx*X
Y = dy*Y
# the following is an alternative to the three lines above
#X, Y = np.mgrid[0: Lx + 1e-10: dx, 0: Ly + 1e-10: dy]
# but without the need to add a "small" increment to ensure
# the Lx and Ly end points are included 

# define an initial condition - here a "rectangle" or square top hat function
# zero in the far field, and a square region of value 1, in this example
# centred in the middle of the domain [0.5,0.5]
C = np.zeros_like(X)
C[(X > 0.4) & (X < 0.6) & (Y > 0.4) & (Y < 0.6)] = 1

# the following is how we can start to plot in 3D
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(111, projection='3d')

surf = ax1.plot_surface(X, Y, C, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly)
ax1.set_zlim(0, 1)

ax1.set_xlabel('$x$', fontsize=14)
ax1.set_ylabel('$y$', fontsize=14)
ax1.set_zlabel('$C$', fontsize=14)

ax1.set_title('Initial condition', fontsize=14);

def solve_diff_central(C, kappa, dt, tend, dx, dy):
    """ Function to evolve the solution in array C
    forward in time an amount 'tend' under diffusion, using time steps
    of size 'dt' and assuming a constant mesh size of dx and dy,
    using a central difference in space.
    """
    t = 0
    while t < tend:
        t += dt
        Cprev = C.copy()
        C[1:-1, 1:-1] = Cprev[1:-1, 1:-1] + dt * kappa * (
              np.diff(Cprev[:, 1:-1], n=2, axis=0) / (dx**2) +
              np.diff(Cprev[1:-1, :], n=2, axis=1) / (dy**2) )
        # Dirichlet BCs
        C[0, :] = 0
        C[-1, :] = 0
        C[:, 0] = 0
        C[:, -1] = 0
    return C

# fig 1 will be an example of plotting the solution in 3D
fig1 = plt.figure(figsize=(12, 12))
fig1.tight_layout(w_pad=6, h_pad=6)

# and fig 2 will show how to represent the soln as a 2D contour plot
fig2 = plt.figure(figsize=(10, 10))
fig2.tight_layout(w_pad=6, h_pad=6)

# let's compute solution and plot for 4 different end times
C = np.zeros_like(X)
C[(X > 0.4) & (X < 0.6) & (Y > 0.4) & (Y < 0.6)] = 1
dt = 0.0001
for (i, tend) in enumerate((dt*0, dt*10, dt*100, dt*1000)):
    C = solve_diff_central(C, kappa, dt, tend, dx, dy)
    ax = fig1.add_subplot(2, 2, i+1, projection='3d')
    surf = ax.plot_surface(X, Y, C, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=True)
    fig1.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, 1)
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_zlabel('$C$', fontsize=14)
    ax.set_title('unsteady diffusion - central. $t$={0:.4f}'.format(tend), fontsize=14)

    # try a contour plot as well
    ax = fig2.add_subplot(2, 2, i+1)
    ax.contour(X,Y,C,cmap=cm.coolwarm)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.axis('equal')
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title('unsteady diffusion - central. $t$={0:.4f}'.format(tend), fontsize=14)

# and this is how we could generate a movie


plot_args = {'rstride': 1, 'cstride': 1, 'cmap':
             cm.coolwarm, 'linewidth': 0., 'antialiased': True, 'color': 'w'}

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

C = np.zeros_like(X)
C[(X > 0.4) & (X < 0.6) & (Y > 0.4) & (Y < 0.6)] = 1

# first frame
plot = ax.plot_surface(X, Y, C, **plot_args)
plt.close()


def data_gen(framenumber, C, plot):
    # make sure initial condition is in plot
    if framenumber == 0:
        C = np.zeros_like(X)
        C[(X > 0.4) & (X < 0.6) & (Y > 0.4) & (Y < 0.6)] = 1
    else:
        # update solution by 2 dts per frame
        C = solve_diff_central(C, kappa, dt, 2*dt, dx, dy)
    ax.clear()
    plot = ax.plot_surface(X, Y, C, **plot_args)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_zlabel('$C$', fontsize=14)
    ax.set_title('Diffusion - central', fontsize=14)
    return plot,

anim = animation.FuncAnimation(fig, data_gen, frames=np.arange(0, 50), fargs=(C, plot),
                               interval=50, blit=True)