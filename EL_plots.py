
"""
#!/usr/bin/env python
# coding: utf-8

Nicolas Grisouard, University of Toronto, Department of Physics.
June 2020

Ekman layer post-processing.
"""

import os
import pathlib
import numpy as np
from scipy.special import erf, erfc, erfi, fresnel
import h5py
import matplotlib.pylab as plt
from dedalus.extras import plot_tools
import logging
from matplotlib import rc
import matplotlib.ticker as tckr

logger = logging.getLogger(__name__)


def retrieve_2D(dset, transpose=False):
    xmesh, ymesh, data = plot_tools.get_plane(
        dset, xaxis=0, yaxis=1, slices=(slice(None), slice(None)),
        xscale=0, yscale=0)

    xmesh = np.delete(xmesh, (0), axis=0)
    xmesh = np.delete(xmesh, (0), axis=1)
    ymesh = np.delete(ymesh, (0), axis=0)
    ymesh = np.delete(ymesh, (0), axis=1)

    return xmesh, ymesh, data


def mVt_closed(T, Z):
    """ $V$ for an impulse forcing """
    sq2i = 2j**.5
    A = (np.exp(Z*sq2i)*erfc(-Z/(2*T)**.5 - (1j*T)**.5) -
         np.exp(-Z*sq2i)*erfc(-Z/(2*T)**.5 + (1j*T)**.5))*np.sqrt(1j/8)
    return A


def mVt_surf(T):
    """ surface $V^\ddagger$ for an impulse forcing"""
    S, C = fresnel((np.pi*T/2)**.5)
    A = S + 1j*C
    # A = erf((1j*T)**.5)*(1j/2)**.5
    return A


def EkmanSpiral(Z):
    """ The expression for the T->oo solution. """
    A = np.exp(Z + 1j*(np.pi/4 + Z))/2**.5
    return A


def find_nearest(arr, val):
    """ Given a physical value, find the index whose array value is closest """
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx, arr[idx]


plt.close('all')


# Basic parameters -----------------------------------------------------------|
Ro = xRox  # Rossby
f = 1.e-4  # [rad/s] Coriolis
nu = xnux  # [m2/s]  vertical viscosity
nz = 256

tskp = 10  # when plotting crosses of numerical simulations, skip every tskp
zskp = 8  # when plotting crosses of numerical simulations, skip every zskp

depths = [0., -1.5, -4.8]  # in units of DE
instants = [0.27, 1., 4.3]  # in units of 1/F

ftsz = 12
saveYN = True
dpi = 150

mf = tckr.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-2, 2))  # to not have crazy numbers of zeroes

rc('font',  # never really worked
   **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': ftsz})
rc('text', usetex=True)
# clrs = ['C{}'.format(str(i)) for i in range(10)]
clrs = ['0.', '0.6', '0.8']  # grey scales
# ['{}'.format(str(i)) for i in np.linspace(0., 0.9, 4)]


# Ancilliary dimensional numbers ---------------------------------------------|
beta = np.sqrt(1+Ro)
F = f*beta
TF = 2*np.pi/F
Tf = 2*np.pi/f
dE = (2*nu/f)**.5
DE = (2*nu/F)**.5


# OPEN PARENTHESIS ----------
# This procedure is a remnant of an old configuration. Particular values don't
# matter in a linear framework.
# numbers related to stratification
vpi = 0.1  # f/N0
N02 = (f/vpi)**2
Ri = 1.
M02 = f*(N02/Ri)**.5
vz0 = -M02/f
A0 = vz0*DE/beta  # This value has to match what is in Dedalus
# CLOSE PARENTHESIS ---------

# where we will save the figures
hmpth = os.path.expanduser("~")
figpth = pathlib.Path(__file__).parent.absolute()  # EL pics stay local
if not os.path.isdir(figpth):
    os.mkdir(figpth)


# Warm-up --------------------------------------------------------------------|
# Loading
fid = h5py.File("anim/anim.h5", mode='r')
_, _, u = retrieve_2D(fid['tasks']['u'])
tz, zt, v = retrieve_2D(fid['tasks']['v'])
fid.close()

mV = u + 1j*v/beta

z, t = zt[:, 0], tz[0, :]
Z, T = z/DE, t*F
ZT, TZ = zt/DE, tz*F

TZp = TZ*0.5/np.pi  # better when plotting
Tp = T*0.5/np.pi  # better when plotting


# theoretical fields ---------------------------------------------------------|
mVt_th = mVt_closed(TZ, ZT)
mVt_th[:, 0] = 0.   # Fixes NaN
u_th = mVt_th.real
v_th = beta * mVt_th.imag

mVt0_th = mVt_surf(T)
u0_th = mVt0_th.real
v0_th = beta * mVt0_th.imag
u0_th[0], v0_th[0] = 0., 0.


# Fig. 1: comparison theory/numerics, Z-profiles -----------------------------|
fg1, ax1 = plt.subplots(1, 1, figsize=(5, 4), dpi=dpi, sharey=True)

for ii, instant in enumerate(instants):
    idt, tt = find_nearest(Tp, instant)
    ax1.plot(u_th[:, idt], Z, color=clrs[ii % 10],
             label=r'$t/T_F = {0:.2f}$'.format(tt))
    ax1.plot(u[2*ii::zskp, idt]/A0, Z[2*ii::zskp], '+', color=clrs[ii % 10])
ax1.plot(EkmanSpiral(Z[2*ii::zskp]).real, Z[2*ii::zskp], 'r:')
ax1.set_xlabel('$u/A_0$')
ax1.set_ylabel(r'$z/\delta$')
ax1.legend()
ax1.grid()

plt.tight_layout()


# Fig. 2: comparison theory/numerics, T-series -------------------------------|
fg2, ax2 = plt.subplots(1, 1, figsize=(5, 4), dpi=dpi, sharex=True)

for jj, depth in enumerate(depths):
    itz, zz = find_nearest(Z, depth)
    tstrt = 2 + 2*jj
    ax2.plot(Tp, u_th[itz, :], color=clrs[jj % 10],
             label='$z/\delta = {0:.1f}$'.format(zz))
    ax2.plot(Tp[tstrt::tskp], u[itz, tstrt::tskp]/A0, '+', color=clrs[jj % 10])
ax2.legend(loc='right')
ax2.set_xlabel(r'$t/T_F$')
ax2.set_ylabel('$u/A_0$')
ax2.grid()

plt.tight_layout()


# t-evolution of \vec{v} -----------------------------------------------------|
fg4, ax4 = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)

ax4.plot(u_th[-1, :], v_th[-1, :]/beta)
ax4.plot(u[-1, ::4]/A0, v[-1, ::4]/beta/A0, '+')
ax4.plot(EkmanSpiral(0.).real, EkmanSpiral(0.).imag, 'r+')
ax4.set_aspect('equal')
ax4.set_xlabel('$u/A_0$')
ax4.set_ylabel(r'$v/(\beta A_0)$')
ax4.set_xticks([0.1*ii for ii in range(9)])
ax4.set_title(r'Up to $t/T_F = {0:.1f}$'.format(Tp[-1]))
ax4.grid()

plt.tight_layout()


# Saving figures -------------------------------------------------------------|
if saveYN:
    fg1.savefig(os.path.join(figpth, 'V_of_z_EL.pdf'), bbox_inches='tight')
    fg2.savefig(os.path.join(figpth, 'V_of_t_EL.pdf'), bbox_inches='tight')
    fg4.savefig(os.path.join(figpth, 'hodograph_EL.pdf'), bbox_inches='tight')
else:
    plt.show()

plt.close('all')
