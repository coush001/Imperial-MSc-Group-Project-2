import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import os 

def assemble_adv_diff_disc_matrix_central(U, kappa, L, N):
    """ Function to form the spatial discretisation matrix for 
    advection-diffusion using central differences, given physical 
    parameters U and kappa and assuming a uniform mesh of N interior 
    nodes on the interval [0,L] with the ghost node approach.
    
    Returns the discretisation matrix A as well as the mesh x.
    """
    # define spatial mesh
    dx = L / N
    x = np.linspace(-dx / 2, dx / 2 + L, N + 2)
    # define first the parameters we defined above
    r_diff = kappa / dx**2
    r_adv = 0.5 * U / dx

    # and use them to create the B matrix - recall zeros on the first and last rows
    B = np.zeros((N + 2, N + 2))
    for i in range(1, N + 1):
        B[i, i - 1] = r_diff + r_adv
        B[i, i] = -2 * r_diff
        B[i, i + 1] = r_diff - r_adv

    # create M matrix - start from the identity
    M = np.eye(N + 2)
    # and fix the first and last rows
    M[0,(0,1)] = [0.5, 0.5]
    M[-1,(-2,-1)] = [0.5, 0.5]   

    # find A matrix
    A = np.linalg.inv(M) @ B
    return A, x

Pe = 5
L = 1
U = 1
CE = 1
kappa = 1/Pe

# define number of points in spatial mesh (N+2 including ghose nodes)
N = 40

# use the function we just wrote to form the spatial mesh and the discretisation matrix 
A, x = assemble_adv_diff_disc_matrix_central(U, kappa, L, N)

# define a time step size
dt = 0.001

# and compute and print some key non-dimensional parameters - we'll explain these later
dx = L / N
print('Pe_c: {0:.5f}'.format(U*dx/kappa))
print('CFL: {0:.5f}'.format(U*dt/dx))
print('r: {0:.5f}'.format(kappa*dt/(dx**2)))

# define the end time and hence some storage for all solution levels
tend = 1
# assume a constant dt and so can define all the t's in advance
t = np.arange(0, tend, dt)
# and we can also set up a matrix to store the discrete solution in space-time
# with our a priori knowledge of the size required.
C = np.empty((len(x),len(t)))

# define an initial condition - this one just linearly varies between the BC values, 
# i.e. 0 and 1 - this linear solution is also the steady state solution to the diffusion 
# only problem of course.
# Let's place this discrete solution in the first column of the C matrix which stores all solution levels
C[:,0] = CE * x / L

# now let's do the time-stepping via a for loop
# we will need the identity matrix so define it once outside the loop
I = np.eye(len(x))
for n in range(len(t)-1):
    C[:,n+1] = (I + A * dt) @ C[:, n]
 

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111, xlim=(0, 1), ylim=(-0.1, 1.1),
                      xlabel='$x$', ylabel='$C$',
                      title='Advection-Diffusion - convergence to the steady state solution')
# if we don't close the figure here we get a rogue frame under the movie
xf = np.linspace(0, L, 1000)
Cex = CE * (np.exp(Pe * xf / L) - 1) / (np.exp(Pe) - 1)
ax1.plot(xf, Cex, 'k', lw=3, label='exact ss solution')
line, = ax1.plot([], [], 'b', lw=3, label='transient numerical solution')
ax1.plot(xf, Cex/0.8, 'g', lw=3, label='line1plot')
line1, = ax1.plot([], [], 'r', lw=3, label='line1')
time_text = ax1.text(0.78, 0.95, '', transform=ax1.transAxes)
ax1.legend(loc='upper left', fontsize=14)


def init():
    line.set_data([], [])
    time_text.set_text('')
    line1.set_data([], [])
    return line, time_text, line1


def animate(i):
    line.set_data(x, C[:, i])
    line1.set_data(x, C[:, i]/0.8)
    time_text.set_text('time = {0:.3f}'.format(i*dt))
    return line, time_text, line1


number_frames = 100
frames = np.arange(0, C.shape[1], int(len(t)/number_frames))
anim = animation.FuncAnimation(fig, animate, frames,
                               interval=40, blit=True, init_func=init)

plt.show()

ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter()
anim.save("practice",writer = writer)
