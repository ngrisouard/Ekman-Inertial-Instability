"""
#!/usr/bin/env python
# coding: utf-8

Nicolas Grisouard, University of Toronto, Department of Physics.
June 2020

Ekman-Inertial Instability
"""

import os
import numpy as np
from scipy.special import erf, erfc, wofz, dawsn, erfi
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


def mUd_closed(T, Z):
    """ $U^\dagger$ for an impulse forcing """
    sq2 = np.sqrt(2)
    A = +np.exp(-Z**2/(2*T))*wofz(-1j*Z/np.sqrt(2*T) + np.sqrt(T))
    # A = +np.exp(-T+1j*Z*sq2)*erfc(-Z/np.sqrt(2*T) - 1j*np.sqrt(T))
    return A.imag/sq2


def mV_closed(T, Z):
    """ $V$ for an impulse forcing """
    sq2 = np.sqrt(2)
    A = (np.exp(Z*sq2)*erfc(-Z/(2*T)**.5 - T**.5) -
         np.exp(-Z*sq2)*erfc(-Z/(2*T)**.5 + T**.5))/(2*sq2)
    return A


def mUd_surf(T):
    """ surface $U^\dagger$ for an impulse forcing  """
    A = (2/np.pi)**.5*dawsn(T**.5)
    return A


def mV_surf(T):
    """ surface $V$ for an impulse forcing """
    A = erf(T**.5)/2**.5
    return A


def sigma_mU_closed(T, Z):
    """ growth rate of $U^\dagger$ for an impulse forcing """
    dmUdt = (2*np.pi*T)**(-0.5) * np.exp(- Z**2/(2*T))  # omitting exp(T)
    mU = mUd_closed(T, Z)  # omitting exp(T)
    return dmUdt/mU


def sigma_mU_surf(T):
    """ surface growth rate of $U^\dagger$ for an impulse forcing"""
    dmUdt = (2*np.pi*T)**(-0.5)  # we omit multiplication by exp(T)
    mU = mUd_surf(T)  # we omit multiplication by exp(T)
    return dmUdt/mU


def find_nearest(arr, val):
    """ Given a physical value, find the index whose array value is closest """
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx, arr[idx]


plt.close('all')


# Basic parameters -----------------------------------------------------------|
Ro = xRox  # Rossby number
f = 1.e-4  # [rad/s] Coriolis
nu = xnux  # [m2/s]  vertical viscosity
nz = 256

tskp = 10  # when plotting crosses of numerical simulations, skip every tskp
zskp = 8  # when plotting crosses of numerical simulations, skip every zskp

depths = [0., -1., -2., -5.]  # in units of DE
instants = [0.1, .854, 2., 15.]  # in units of 1/F
ite = 30  # iteration to display the early stage of instability

ftsz = 12
saveYN = False  # set True to print pics (will have to change the figpath)
dpi = 150

mf = tckr.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-2, 2))  # to not have crazy numbers of zeroes

rc('font',  # never really worked
   **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': ftsz})
rc('text', usetex=True)
# clrs = ['C{}'.format(str(i)) for i in range(10)]
clrs = ['0.', '0.5', '0.65', '0.8']  # grey scales
# ['{}'.format(str(i)) for i in np.linspace(0., 0.9, 4)]


# Ancilliary dimensional numbers ---------------------------------------------|
alpha = np.sqrt(-1-Ro)
F = f*alpha
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
A0 = vz0*DE/alpha  # This value has to match what is in Dedalus
# CLOSE PARENTHESIS ---------

hmpth = os.path.expanduser("~")
figpth = os.path.join(  # EII pics go somewhere
    hmpth, "Dropbox/Applications/Overleaf/Ekman Inertial Instability/figures")
if not os.path.isdir(figpth):
    os.mkdir(figpth)


# Warm-up --------------------------------------------------------------------|
# Loading
fid = h5py.File("anim/anim.h5", mode='r')
_, _, u = retrieve_2D(fid['tasks']['u'])
_, _, v = retrieve_2D(fid['tasks']['v'])
_, _, LSP = retrieve_2D(fid['tasks']['LSP'])
_, _, Phiz = retrieve_2D(fid['tasks']['KFX'])
_, _, eps = retrieve_2D(fid['tasks']['KED'])
_, _, ut = retrieve_2D(fid['tasks']['dudt'])
_, _, vt = retrieve_2D(fid['tasks']['dvdt'])
tz, zt, Kt = retrieve_2D(fid['tasks']['dKdt'])
fid.close()

mU = u + v/alpha
mUd = mU*np.exp(-F*tz)
mUt = ut + vt/alpha
mV = - u + v/alpha

z, t = zt[:, 0], tz[0, :]
Z, T = z/DE, t*F
ZT, TZ = zt/DE, tz*F


# theoretical fields ---------------------------------------------------------|
mUd_th = mUd_closed(TZ, ZT)
mV_th = mV_closed(TZ, ZT)
mUd_th[:, 0] = 0.  # Fixes NaN
mV_th[:, 0] = 0.  # Fixes NaN

mUd0_th = mUd_surf(T)
mV0_th = mV_surf(T)
mUd0_th[0] = 0.  # Fixes NaN
mV0_th[0] = 0.  # Fixes NaN

sigma_mU_th = sigma_mU_closed(TZ, ZT)
sigma_mU0_th = sigma_mU_surf(T)

mU_th = mUd_th*np.exp(TZ)

u_th = 0.5*(mU_th - mV_th)
v_th = 0.5*(mU_th + mV_th)*alpha


# Fig. 1: comparison theory/numerics, Z-profiles -----------------------------|
fg1, ax1 = plt.subplots(1, 2, figsize=(5, 4), dpi=dpi, sharey=True)

for ii, instant in enumerate(instants):
    idt, tt = find_nearest(T, instant)
    ax1[0].plot(mUd[2*ii::zskp, idt]/A0, Z[2*ii::zskp],
                '+', color=clrs[ii % 10])
    ax1[0].plot(mUd_th[:, idt], Z, color=clrs[ii % 10])
    ax1[1].plot(mV[2*ii::zskp, idt]/A0, Z[2*ii::zskp],
                '+', color=clrs[ii % 10])
    ax1[1].plot(mV_th[:, idt], Z, color=clrs[ii % 10],
                label='$Ft = {0:.2f}$'.format(tt))
ax1[1].set_xlabel('$V/A_0$')
ax1[0].set_ylabel('$z/\delta$')
ax1[1].legend()
ax1[0].grid()
ax1[0].set_ylim([-10., 0.2])
ax1[0].set_xlabel('$U^\dagger/A_0$')
ax1[1].grid()

plt.tight_layout()
fg1.subplots_adjust(wspace=0.05)


# Fig. 2: comparison theory/numerics, T-series -------------------------------|
fg2, ax2 = plt.subplots(2, 1, figsize=(5, 4), dpi=dpi, sharex=True)

for jj, depth in enumerate(depths):
    itz, zz = find_nearest(Z, depth)
    tstrt = 2 + 2*jj
    ax2[0].plot(T[tstrt::tskp], mUd[itz, tstrt::tskp]/A0, '+',
                color=clrs[jj % 10])
    ax2[0].plot(T, mUd_th[itz, :], color=clrs[jj % 10])
    ax2[1].plot(T[tstrt::tskp], mV[itz, tstrt::tskp]/A0, '+',
                color=clrs[jj % 10])
    ax2[1].plot(T, mV_th[itz, :], color=clrs[jj % 10],
                label='$z/\delta = {0:.1f}$'.format(zz))
    ax2[1].set_aspect(5.5)
ax2[0].set_ylabel('$U^\dagger/A_0$')
ax2[1].legend(loc='right')
ax2[0].grid()
ax2[1].set_xlabel('$Ft$')
ax2[1].set_ylabel('$V/A_0$')
ax2[1].grid()

# # Un-comment for verification purposes
# ax2[0].plot(T, erfi(T**.5)*np.exp(-T)/np.sqrt(2), '.')
# ax2[0].plot(T, mUd_surf(T), ':')
# ax2[1].plot(T, mV_surf(T))
# # END Un-comment for verification purposes

plt.tight_layout()
fg2.subplots_adjust(hspace=-0.15)


# Fig.3: growth rate ---------------------------------------------------------|
fg3, ax3 = plt.subplots(2, 1, figsize=(5, 3.5), dpi=dpi, sharex=True)

for jj, depth in enumerate(depths):
    itz, zz = find_nearest(Z, depth)
    ax3[1].semilogy(T, sigma_mU_th[itz, :], color=clrs[jj % 10],
                    label='$z/\delta = {0:.1f}$'.format(zz))
    ax3[0].semilogy(T, sigma_mU_th[itz, :], color=clrs[jj % 10],
                    label='$z/\delta = {0:.1f}$'.format(zz))
    tstrt = 2 + 2*jj
    s0 = mUt[itz, tstrt::tskp]/mU[itz, tstrt::tskp]/F
    ax3[1].semilogy(T[tstrt::tskp], s0, '+', color=clrs[jj % 10])
    ax3[0].semilogy(T[tstrt::tskp], s0, '+', color=clrs[jj % 10])
# ax3[0].set_ylabel('$\sigma_U/(A_0F)$')
ax3[0].legend(loc='upper right')
ax3[0].grid()
ax3[1].set_xlabel('$Ft$')
ax3[1].set_ylabel('$\sigma_U/F$')
ax3[0].set_ylim([2., 1e4])
ax3[1].set_ylim([0.7, 2.])
ax3[1].grid()

plt.tight_layout()
fg3.subplots_adjust(hspace=0.0)  # I think 0 is the minimum


# t-evolution of \vec{v} -----------------------------------------------------|
# Messy

fg4 = plt.figure(figsize=(5.8, 3.5), dpi=dpi)
gs = fg4.add_gridspec(nrows=1, ncols=3)
ax4 = {}
ax4[0] = fg4.add_subplot(gs[0])
ax4[1] = fg4.add_subplot(gs[1:])

for ii, this_it in enumerate([ite, 4*ite]):
    ax4[ii].plot(u[-1, :this_it]/A0,
                 v[-1, :this_it]/A0/alpha, 'x', color=clrs[0])
    ax4[ii].plot(u_th[-1, :this_it],
                 v_th[-1, :this_it]/alpha, color=clrs[0],
                 label='$z/\delta = 0$')
    ax4[ii].set_aspect('equal')
    ax4[ii].set_xlabel('$u/A_0$')


this_it = ite-1
ax4[0].annotate(
    '$Ft = {:.1f}$'.format(T[this_it]),
    xy=(u_th[-1, this_it], v_th[-1, this_it]/alpha),
    xytext=(u_th[-1, this_it]-0.2, (v_th[-1, this_it]-0.02)/alpha),
    verticalalignment='center', horizontalalignment='right', color=clrs[0],
    arrowprops=dict(color=clrs[0], shrink=0.1, width=.5, headwidth=3,
                    headlength=4))

this_it = 4*ite-1
ax4[1].annotate(
    '$Ft = {:.1f}$'.format(T[this_it]),
    xy=(u_th[-1, this_it], v_th[-1, this_it]/alpha),
    xytext=(u_th[-1, this_it]-0.27*(1+np.exp(4)),
            (v_th[-1, this_it]-0.02*(1+np.exp(4)))/alpha),
    verticalalignment='center', horizontalalignment='right', color=clrs[0],
    arrowprops=dict(color=clrs[0], shrink=0.05, width=.5, headwidth=3,
                    headlength=4))

idz, zz = find_nearest(Z, -5.)  # a little below
for ii, this_it in enumerate([ite+68, 4*ite+40]):
    ax4[ii].plot(u[idz, :this_it]/A0,
                 v[idz, :this_it]/A0/alpha, 'x', color=clrs[2])
    ax4[ii].plot(u_th[idz, :this_it],
                 v_th[idz, :this_it]/alpha, color=clrs[2],
                 label='$z/\delta = {:.1f}$'.format(zz))
    ax4[ii].set_aspect('equal')
    ax4[ii].set_xlabel('$u/A_0$')
    # ax4[ii].set_title('Up to $Ft={0:.1f}$'.format(T[this_it]))
    ax4[ii].grid()
    ax4[ii].set_ylabel(r'$v/(\alpha A_0)$')

this_it = ite + 68 - 1
ax4[0].annotate(
    '$Ft = {:.1f}$'.format(T[this_it]),
    xy=(u_th[idz, this_it], v_th[idz, this_it]/alpha),
    xytext=(u_th[idz, this_it]-0.25, (v_th[idz, this_it]-0.15)/alpha),
    verticalalignment='center', horizontalalignment='center', color=clrs[2],
    arrowprops=dict(color=clrs[2], shrink=0.1,
                    width=.5, headwidth=3, headlength=4))

this_it = 4*ite + 40 - 1
ax4[1].annotate(
    '$Ft = {:.1f}$'.format(T[this_it]),
    xy=(u_th[idz, this_it], v_th[idz, this_it]/alpha),
    xytext=(u_th[idz, this_it]-0.1*(1+np.exp(4)),
            (v_th[idz, this_it]-0.075*(1+np.exp(4)))/alpha),
    verticalalignment='center', horizontalalignment='center', color=clrs[2],
    arrowprops=dict(color=clrs[2], shrink=0.05,
                    width=.5, headwidth=3, headlength=4))
ax4[1].legend(loc='lower right')

plt.tight_layout()
fg4.subplots_adjust(wspace=0.05)


# Energetics -----------------------------------------------------------------|
fg5, ax5 = plt.subplots(1, 2, figsize=(5, 4), dpi=dpi)
idt, tt = find_nearest(T, instants[2])
norm = F * abs(Ro) * A0**2 * np.exp(2*tt)
for ii in range(2):
    ax5[ii].plot(Kt[:, idt]/norm, Z, '-.', color=clrs[2], label='$K_t$')
    ax5[ii].plot(-LSP[:, idt]/norm, Z, 'k', label='$-LSP$')
    ax5[ii].plot(-Phiz[:, idt]/norm, Z, 'k--', label='$-\Phi_z$')
    ax5[ii].plot(-eps[:, idt]/norm, Z, 'k:', label='$-\epsilon$')
    ax5[ii].set_xlabel(r'Units of $|\textrm{Ro}|F A_0^2 e^{-2F t}$')
    ax5[ii].set_ylabel('$z/\delta$')
    ax5[ii].grid()
    # ax5[ii].xaxis.set_major_formatter(mf)
ax5[0].legend(loc='lower right')
ax5[1].set_xlim([-1.8e-3, 3.8e-3])
ax5[0].set_ylim([-4, 0])
ax5[1].set_ylim([-3., -1.5])

plt.tight_layout()
fg5.subplots_adjust(hspace=0.05)


# Saving figures -------------------------------------------------------------|
if saveYN:
    fg1.savefig(os.path.join(figpth, 'UV_of_z.pdf'), bbox_inches='tight')
    fg2.savefig(os.path.join(figpth, 'UV_of_t.pdf'), bbox_inches='tight')
    fg3.savefig(os.path.join(figpth, 'sigmaU_of_t.pdf'), bbox_inches='tight')
    fg4.savefig(os.path.join(figpth, 'hodograph.pdf'), bbox_inches='tight')
    fg5.savefig(os.path.join(figpth, 'energetics.pdf'), bbox_inches='tight')
else:
    plt.show()

plt.close('all')
