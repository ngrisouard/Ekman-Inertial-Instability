"""
#!/usr/bin/env python
# coding: utf-8

Our CISI front in 1D: the evolution in time of fields that don't depend on $x$
I will use to solve the 1D problem as described by Barbara in
EkmanLayer_analytics with Dedalus directly, and different boundary conditions.

This script only solves the problem with dedalus. For figure printing, see
other scripts.
"""

import time
import numpy as np
from dedalus import public as de
import logging
from dedalus.tools import post
import shutil
import pathlib
import subprocess

logger = logging.getLogger(__name__)


def global_noise(domain, seed=42, scale=None, **kwargs):
    """ Random perturbations, initialized globally for same results in parallel
    From the many_scales_youtube simulation (by Jeff Oishi?) """
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    # filter_field(noise_field, **kwargs)
    if scale is not None:
        noise_field.set_scales(scale, keep_data=True)

    return noise_field['g']


# Basic parameters -----------------------------------------------------------|
Ro = xRox  # switching sign to get unstable cases
f = 1.e-4  # [rad/s] Coriolis
nu = xnux  # [m2/s] vertical viscosity
nz = 256
buoy_YN = False  # Compute advection of lateral buoyancy gradient?
vpi = 0.1  # f/N0
Ri = 1.
betab = 1.
if Ro >= -1.:
    F = f*np.sqrt(1+Ro)
else:
    F = f*np.sqrt(-1-Ro)
TF = 2*np.pi/F
Tf = 2*np.pi/f
dtm = 1e-5/F
dtM = 1e-2/F
sigma_dt = F
t_end = 15/F


# Ancilliary dimensional numbers ---------------------------------------------|
dE = (2*nu/f)**.5
DE = (2*nu/F)**.5
H = 15*DE  # H is fixed (50*dE for nu=1e-4)

# numbers related to stratification
N02 = (f/vpi)**2
M02 = f*(N02/Ri)**.5

# Time stepping parameters
walltime_max = np.inf  # can set a value in seconds
nites_max = np.inf  # [ ] stop the calculation if it goes too far

# %% Create basis and domain -------------------------------------------------|
z_basis = de.Chebyshev('z', nz, interval=(-H, 0.), dealias=1)
domain = de.Domain([z_basis], grid_dtype=np.float64)


# %% 2D Boussinesq hydrodynamics ---------------------------------------------|
if buoy_YN:
    pb = de.IVP(domain, variables=['u', 'uz', 'v', 'vz', 'b', 'bz'], time='t')
else:
    pb = de.IVP(domain, variables=['u', 'uz', 'v', 'vz'], time='t')

pb.meta['u']['z']['dirichlet'] = False  # marginal speed-up of the simulations

pb.parameters['f'] = f
pb.parameters['nu'] = nu
pb.parameters['Ro'] = Ro
pb.parameters['M02'] = M02
if buoy_YN:
    pb.parameters['N02'] = N02
    pb.parameters['betab'] = betab

pb.add_equation("dt(u) - nu*dz(uz) - f*v = 0")
pb.add_equation("dt(v) - nu*dz(vz) + f*(1+Ro)*u = 0")
pb.add_equation("uz - dz(u) = 0")
pb.add_equation("vz - dz(v) = 0")
if buoy_YN:
    pb.add_equation("dt(b) - nu*dz(bz) + M02*u = 0")
    pb.add_equation("bz - dz(b) = 0")


# boundary conditions in z
pb.add_bc("left(uz) = 0")  # always free-slip
pb.add_bc("right(uz) = 0")  # always free-slip
pb.add_bc("left(vz) = 0")  # restores mean stress
# the -1 is because I define v differently in the code (fluctuations).
pb.add_bc("right(vz) = -M02/f")
if buoy_YN:
    pb.add_bc("right(bz) = (betab - 1.)*N02")
    pb.add_bc("left(bz) = 0")  # restores mean strat


# Build solver ---------------------------------------------------------------|
solver = pb.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions for u
u = solver.state['u']
uz = solver.state['uz']
noise_u = global_noise(domain, scale=1, frac=0.25)  # below is original
pert = 1e-6 * noise_u  # * (-z) * (z + H)
u['g'] = pert
u.differentiate('z', out=uz)

# Initial conditions for v
v = solver.state['v']  # initial condition for v
vz = solver.state['vz']  # maybe we can get rid of this
noise_v = global_noise(domain, scale=1, frac=0.25)  # below is original
pert = 1e-6 * noise_v  # * (-z) * (z + H)
v['g'] = pert
v.differentiate('z', out=vz)

if buoy_YN:
    # Initial conditions for b
    b = solver.state['b']  # initial condition for b
    bz = solver.state['bz']  # maybe we can get rid of this
    b.differentiate('z', out=vz)

# Integration parameters
solver.stop_sim_time = t_end
solver.stop_wall_time = walltime_max
solver.stop_iteration = nites_max


# Analysis -------------------------------------------------------------------|
# first, let's delete the old stuff
try:
    shutil.rmtree('snapshots')
    shutil.rmtree('anim')
    shutil.rmtree('spectra')
except:
    print('No folders to delete.')  # a bit quick but hey

snapshots = solver.evaluator.add_file_handler(  # everything
    'snapshots', sim_dt=Tf/10, mode='append', max_writes=32)
snapshots.add_system(solver.state)  # maybe not useful to print out everything

anim = solver.evaluator.add_file_handler(
    'anim', sim_dt=Tf/40, mode='append')
anim.add_task("u", scales=1, layout='g', name='u')
anim.add_task("v", scales=1, layout='g', name='v')
anim.add_task("uz", scales=1, layout='g', name='uz')
anim.add_task("vz", scales=1, layout='g', name='vz')
anim.add_task("nu*dz(uz) + f*v", scales=1, layout='g', name='dudt')
anim.add_task("nu*dz(vz) - f*(1+Ro)*u", scales=1, layout='g', name='dvdt')
anim.add_task("nu*dz(uz)", scales=1, layout='g', name="MxFX")
anim.add_task("nu*dz(vz)", scales=1, layout='g', name="MyFX")
anim.add_task("nu*u*dz(uz) + v*(nu*dz(vz) - f*Ro*u)", scales=1, name='dKdt',
              layout='g')
anim.add_task("f*Ro*u*v", scales=1, layout='g', name='LSP')
anim.add_task("nu*(uz**2 + vz**2)", scales=1, layout='g', name="KED")
anim.add_task("-nu*dz(u*uz + v*vz)", scales=1, layout='g', name="KFX")
if buoy_YN:
    anim.add_task("b", scales=1, layout='g', name='b')
    anim.add_task("nu*dz(bz) - M02*u", scales=1, layout='g', name='dbdt')
    anim.add_task("nu*dz(bz)", scales=1, layout='g', name="bFX")
    anim.add_task("b*(nu*dz(bz) - M02*u)/N02", scales=1,
                  layout='g', name="dPdt")
    anim.add_task("M02*u*b/N02", scales=1, layout='g', name='GBP')
    anim.add_task("-nu*bz**2/N02", scales=1, layout='g', name="PED")
    anim.add_task("-nu*dz(b*bz)/N02", scales=1, layout='g', name="PFX")


# Main loop ------------------------------------------------------------------|
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        # We start with a very small dt and increase it.
        dt = dtm + (dtM-dtm)*(1 - np.exp(-sigma_dt*solver.sim_time))
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Ite #{0:d}, Time: {1:.2e} IPs, dt: {2:e}'.format(
                solver.iteration, solver.sim_time/TF, dt))
except NameError:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: {0:d}'.format(solver.iteration))
    logger.info('Sim end time: {0:f}'.format(solver.sim_time))
    logger.info('Run time: {0:.2f} sec'.format(end_time - start_time))
    logger.info('Run time: {0:f} cpu-min'.format((end_time-start_time)/30
                                                 * domain.dist.comm_cart.size))


# Post-processing ------------------------------------------------------------|
print(subprocess.check_output("find anim", shell=True).decode())
post.merge_process_files("anim", cleanup=True)  # merges the sub-domains if any
set_paths = list(pathlib.Path("anim").glob("anim_s*.h5")
                 )  # finds all of the time series
# merges the time series
post.merge_sets("anim/anim.h5", set_paths, cleanup=True)
print(subprocess.check_output("find anim", shell=True).decode())
