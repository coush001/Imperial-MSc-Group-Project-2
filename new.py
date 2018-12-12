import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import visualization as visu
 
# Initialise grid
domain = visu.SPH_main()
domain.set_values()
domain.initialise_grid()
domain.place_points(domain.min_x, domain.max_x)
domain.allocate_to_grid()

result_list, t_array = domain.simulate(domain.dt, domain.forward_euler)

print("simulation done")
x_data = []
y_data = []
x_small = []
y_small = []
for i in result_list:
    x_small = []
    y_small = []
    for j in i:
        if j.boundary is False:
            x_small.append(j.x[0])
            y_small.append(j.x[1])
    x_data.append(x_small)
    y_data.append(y_small)

print("output done")
a = domain.output_particle()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)


def make_frame(t):
    i = int(t/domain.dt)
    ax1.clear()
    ax1.set_title("SPH practice", fontsize=16)
    ax1.plot(x_data[i], y_data[i])
 
    return mplfig_to_npimage(fig)

duration = 1
animation = VideoClip(make_frame, duration=duration)
print("animation done")
animation.write_gif('togif.gif', fps=20)