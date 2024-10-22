import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
from numba import jit

# Particle class definition
class Particle:
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge
        self.r = np.zeros(3, dtype='float64')
        self.u = np.zeros(3, dtype='float64')
        self.gamma = 1.0  # Set initial gamma

    def initPos(self, x, y, z):
        self.r[:] = (x, y, z)

    def initSpeed(self, ux, uy, uz):
        self.u[:] = (ux, uy, uz)

# Electric field function
@jit(nopython=True)
def E(r, E0=1-5e-3):
    return np.array([E0, 0.0, 0.0])

# Magnetic field function
@jit(nopython=True)
def B(r, B0=1):
    return np.array([0.0, 0.0, 1])

# Push function for particle simulation
@jit(nopython=True)
def push(r, u, gamma, charge, mass, dt):
    rplus = r + u * dt / (2 * gamma)
    u += charge * E(rplus) * dt / (2 * mass)
    gamma_minus = np.sqrt(1 + np.sum(u ** 2))

    # Calculate magnetic field effect
    B_effect = charge * dt * B(rplus) / (2 * mass * gamma_minus)
    t = B_effect
    u1 = u + np.cross(u, t)
    u += np.cross(u1, t)

    u += charge * E(rplus) * dt / (2 * mass)
    gamma = np.sqrt(1 + np.sum(u ** 2))

    r += dt * u / gamma
    return r, u, gamma

def simulate_particle(particle_params):
    charge, mass, r0, u0, dt, Nt = particle_params

    part = Particle(mass, charge)
    part.initPos(r0[0], r0[1], r0[2])
    part.initSpeed(u0[0], u0[1], u0[2])

    # Prepare arrays to hold trajectory data
    x = np.zeros(Nt)
    y = np.zeros(Nt)
    z = np.zeros(Nt)
    ux = np.zeros(Nt)
    uy = np.zeros(Nt)
    uz = np.zeros(Nt)
    gamma = np.zeros(Nt)

    for i in tqdm(range(Nt)):
        part.r, part.u, part.gamma = push(part.r, part.u, part.gamma, charge, mass, dt)
        x[i], y[i], z[i] = part.r
        ux[i], uy[i], uz[i] = part.u
        gamma[i] = part.gamma

    return x, y, z, ux, uy, uz, gamma

def parallel_simulation_with_progress(particle_params_list, num_processors=4):
    results = []
    with Pool(processes=num_processors) as pool:
        with (total=len(particle_params_list), desc='Simulating Particles') as pbar:
            for params in particle_params_list:
                result = pool.apply_async(simulate_particle, args=(params,), callback=lambda _: pbar.update())
                results.append(result)
            pool.close()
            pool.join()
    return [res.get() for res in results]

if __name__ == '__main__':
    set_start_method('fork')

    N = 1  # Number of particles
    charge, mass = 1.0, 1.0

    # Random initial positions and speeds for particles
    r0_list = np.array([[0, 0, 0]] * N)
    u0_list = np.array([[0, 0, 0]] * N)

    # Calculate initial parameters
    gamma0 = np.sqrt(1 + np.sum(u0_list[0] ** 2))
    dt = 0.0005
    Np = 10.0  # Number of periods for longest period particle at t=0
    Tf = 2e4 * np.pi
    Nt = int(Tf // dt)

    num_processors = 10
    particle_params_list = [(charge, mass, r0_list[i], u0_list[i], dt, Nt) for i in range(N)]

    # Run simulations in parallel
    results = parallel_simulation_with_progress(particle_params_list, num_processors)

    # Save data to an HDF5 file
    with h5py.File('particle_data.h5', 'w') as f:
        f.attrs['dt'] = dt  # Save dt as an attribute
        for index, traj in enumerate(results):
            x, y, z, ux, uy, uz, gamma = traj
            grp = f.create_group(f'particle_{index}')
            grp.create_dataset('x', data=x)
            grp.create_dataset('y', data=y)
            grp.create_dataset('z', data=z)
            grp.create_dataset('ux', data=ux)
            grp.create_dataset('uy', data=uy)
            grp.create_dataset('uz', data=uz)
            grp.create_dataset('gamma', data=gamma)


