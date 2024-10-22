#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_particle_trajectories(file_name):
    with h5py.File(file_name, 'r') as f:
        # Retrieve w0 and dt from attributes
        dt = f.attrs['dt']
        
        
        N = len(f.keys())  # Number of particles
        colors = cm.viridis(np.linspace(0, 1, N))

        # Plotting all particle trajectories in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for index in range(N):
            grp = f[f'particle_{index}']
            x = grp['x'][:]
            y = grp['y'][:]
            z = grp['z'][:]
            ax.plot(x, y, z, color=colors[index])

        ax.set_xlabel('$x$', labelpad=15)
        ax.set_ylabel('$y$', labelpad=15)
        ax.set_zlabel('$z$', labelpad=2.5)
        plt.show()
        plt.close(fig)

        # Efficient plotting of velocities and gamma
        fig, axs = plt.subplots(4, 1, figsize=(8, 12))
        
        for index in range(N):
            grp = f[f'particle_{index}']
            ux = grp['ux'][:]
            uy = grp['uy'][:]
            uz = grp['uz'][:]
            gamma = grp['gamma'][:]
            Nt = len(ux)  # Number of time steps
            t = np.arange(Nt) * dt  # Create the time array using dt

            axs[0].plot(t / (2 * np.pi), ux/gamma, color=colors[index])
            axs[1].plot(t / (2 * np.pi), uy/gamma, color=colors[index])
            axs[2].plot(t / (2 * np.pi), uz/gamma, color=colors[index])
            axs[3].plot(t / (2 * np.pi), gamma, color=colors[index])

        axs[0].set_ylabel('$v_x$')
        axs[1].set_ylabel('$v_y$')
        axs[2].set_ylabel('$v_z$')
        axs[3].set_ylabel('$\\gamma$')
        axs[3].set_xlabel('$t \\omega_0 / 2\\pi$')
        plt.show()
        plt.close(fig)

#%%
file_name = 'particle_data.h5'

pathto = '/Users/michaelgrehan/Desktop/Boris_MPG/'
plot_particle_trajectories(pathto+file_name)

# %%
