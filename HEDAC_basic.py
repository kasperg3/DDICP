# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 17:25:16 2015

@author: stefan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.interpolate as intrp
import os
import shutil
import scipy.ndimage.filters as filters
import scipy.sparse as sprs
from scipy.sparse.linalg import splu
from matplotlib import rc
from matplotlib import colors
import matplotlib as mpl
import subprocess
import datetime
from scipy.fftpack import dct, idct

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)

CM = np.loadtxt('cmap.txt').reshape(256, 3)


# print(np.shape(CM))


def DCT(field):
    return dct(dct(field.T, norm='ortho').T, norm='ortho')


def IDCT(coefficients):
    return idct(idct(coefficients.T, norm='ortho').T, norm='ortho')


class HEDAC_basic():

    def __init__(self):

        self.start_time = datetime.datetime.now()
        self.X = np.linspace(0, 1, 200)
        self.Y = np.linspace(0, 1, 200)
        self.T = np.linspace(0, 1, 101)

        self.results_dir = None
        self.alpha = 1.0
        self.beta = 1e6
        self.gamma = 1.0
        self.kappa = 0.
        self.nu = 0.01
        self.va = None
        self.sigma_m = 0.01
        self.sigma_c = 0.01
        self.sigma_ac = 0.01
        self.outputStep = 2
        # self.number_of_agents = 0
        self.agents = 0
        self.samples = None
        self.method = 'hedac'
        self.sourcefun = logsource

        self.FDT = [0.25, 0.5, 0.75, 1.0]

    def write_info(self):

        info = open(self.results_dir + '/info.txt', 'w')

        info.write('*' * 50 + '\n')
        info.write('*' + 'H E D A C   s e a r c h'.center(48) + '*\n')
        info.write('*' * 50 + '\n')

        info.write('Started on: ' + str(self.start_time) + '\n')

        info.write('\n')
        info.write('Parameters:\n')

        info.write('%15s: %s\n' % ('method', self.method))
        info.write('%15s: %10.3f\n' % ('x_min', self.X[0]))
        info.write('%15s: %10.3f\n' % ('x_max', self.X[-1]))
        info.write('%15s: %10.3f\n' % ('dx', self.dx))
        info.write('%15s: %10.3f\n' % ('y_min', self.Y[0]))
        info.write('%15s: %10.3f\n' % ('y_max', self.Y[-1]))
        info.write('%15s: %10.3f\n' % ('dy', self.dy))

        info.write('\n')
        info.write('%15s: %10.3f\n' % ('alpha', self.alpha))
        info.write('%15s: %10.3f\n' % ('beta', self.beta))
        info.write('%15s: %10.3f\n' % ('gamma', self.gamma))
        info.write('%15s: %10.3f\n' % ('kappa', self.kappa))
        info.write('%15s: %10.3f\n' % ('nu', self.nu))
        info.write('%15s: %10.3f\n' % ('sigma_m', self.sigma_m))
        info.write('%15s: %10.3f\n' % ('sigma_c', self.sigma_c))
        info.write('%15s: %10.3f\n' % ('sigma_ac', self.sigma_ac))
        info.write('%15s:          %s\n' % ('source', self.sourcefun.__name__))

        info.write('\n')
        info.write('%15s: %10d\n' % ('agents', len(self.agents)))
        for a in self.agents:
            info.write(' ' * 15 + '%10.3f, %10.3f\n' % tuple(a))

        info.close()

    def init(self):
        print('Load and precompute')

        """
        Delete and recreate resuluts directory
        """
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

        """
        Discretization
        """
        self.dx = self.X[1] - self.X[0]
        self.dy = self.Y[1] - self.Y[0]
        self.nx = np.size(self.X)
        self.ny = np.size(self.Y)
        self.Xc = np.linspace(self.X[0] - 0.5 * self.dx, self.X[-1] + 0.5 * self.dx, self.nx + 1)
        self.Yc = np.linspace(self.Y[0] - 0.5 * self.dy, self.Y[-1] + 0.5 * self.dy, self.ny + 1)
        self.dt = self.T[1] - self.T[0]

        self.Xmg, self.Ymg = np.meshgrid(self.X, self.Y)

        # print self.dx
        # print self.Xc

        """
        Probability initialization
        """
        self.m = np.zeros([self.ny, self.nx])
        if hasattr(self.samples, '__call__'):
            # Defined as probability function
            for iY, y in enumerate(self.Y):
                for iX, x in enumerate(self.X):
                    self.m[iY, iX] = self.samples(x, y)

        elif isinstance(self.samples, np.ndarray):
            # Defined as file containing samples locations
            self.m = self.samples

        elif isinstance(self.samples, basestring):
            # Defined as file containing samples locations

            XM, YM = np.loadtxt(self.samples, delimiter=',', unpack=True)
            """
            # scale and center
            scale = 1.1 * np.max([(np.max(XM) - np.min(XM)), (np.max(YM) - np.min(YM))])
            self.XM = (XM - np.min(XM)) / scale
            self.YM = (YM - np.min(YM)) / scale
            xc = 0.5 * (np.min(self.XM) + np.max(self.XM))
            yc = 0.5 * (np.min(self.YM) + np.max(self.YM))
            self.XM += (0.5 - xc)
            self.YM += (0.5 - yc)
            """
            self.m, _, _ = np.histogram2d(self.YM, self.XM, bins=[self.Yc, self.Xc])

        else:
            raise Exception('Probability initialization failed!')

        self.m /= np.mean(self.m)

        self.ms = filters.gaussian_filter(self.m, self.sigma_m / self.dx)

        self.mk = DCT(self.ms)

        """
        Agent initialization
        """
        self.XA = [[a[0]] for a in self.agents]
        self.YA = [[a[1]] for a in self.agents]

        self.ky, self.kx = np.meshgrid(np.arange(self.ny), np.arange(self.nx))
        self.E = []

        self.U = np.zeros([self.ny, self.nx])

        # Solve stacionary heat equation!
        A = sprs.lil_matrix((self.nx * self.ny, self.nx * self.ny))
        self.B = np.zeros(self.nx * self.ny)

        i = 0
        for iY in range(self.ny):
            for iX in range(self.nx):
                #
                k = [-2., -2., 1., 1., 1., 1.]
                if iY == 0:
                    k[1] -= 2. * self.kappa * self.dy
                    k[4] = 2.0
                    k[5] = 0.0
                if iY == self.ny - 1:
                    k[1] -= 2. * self.kappa * self.dy
                    k[4] = 0.0
                    k[5] = 2.0
                if iX == 0:
                    k[0] -= 2. * self.kappa * self.dx
                    k[2] = 2.0
                    k[3] = 0.
                if iX == self.nx - 1:
                    k[0] -= 2. * self.kappa * self.dx
                    k[2] = 0.
                    k[3] = 2.0

                A[i, i] = self.alpha * (k[0] / self.dx ** 2 + k[1] / self.dy ** 2) - self.gamma

                if iX != self.nx - 1:
                    A[i, i + 1] = k[2] / self.dx ** 2
                if iX != 0:
                    A[i, i - 1] = k[3] / self.dx ** 2
                if iY != self.ny - 1:
                    A[i, i + self.nx] = k[4] / self.dy ** 2
                if iY != 0:
                    A[i, i - self.nx] = k[5] / self.dy ** 2

                i += 1
        A = A.tocsc()
        self.lu = splu(A)

    def smoothing(self, points, sigma):
        A = 1.0
        c = np.zeros([self.ny, self.nx])
        for p in points:
            c += A * np.exp(- ((self.Xmg - p[0]) ** 2.0 + (self.Ymg - p[1]) ** 2.0) / (2.0 * sigma ** 2))

        return c

    def convergence(self):

        E1 = np.sqrt(np.sum(((self.ms - self.cs) / (self.nx * self.ny)) ** 2))
        E2 = np.sum(((self.ms - self.cs) / (self.nx * self.ny)) ** 2)
        # self.E.append( np.sum( ( (self.m - self.c)/(self.nx*self.ny) ) ** 2  ) )
        E4 = np.max(np.abs(self.ms - self.cs))

        L2 = 1.0 / (1 + self.kx ** 2 + self.ky ** 2) ** 1.5 * ((self.ck - self.mk).T ** 2)
        E3 = np.sum(L2)

        self.E.append(E3)

        fajl = open(self.results_dir + '/convergence.txt', 'a')
        fajl.write('%12.6e %12.6e %12.6e %12.6e %12.6e\n' % (self.T[self.it], E1, E2, E3, E4))
        fajl.close()

        """
        CONV = np.zeros([self.it+1, 2])
        CONV[:,0] = self.T[0:self.it+1]
        CONV[:,1] = self.E2
        np.savetxt(self.results_dir + '/convergence.txt',CONV)
        """

    def search(self):

        self.init()
        self.write_info()

        fajl = open(self.results_dir + '/convergence.txt', 'w')
        fajl.write('%12s %12s %12s %12s %12s\n' % ('t', 'E1', 'E2', 'E3', 'E4'))
        fajl.close()

        if self.outputStep > 0:
            fajl = open(self.results_dir + '/agents.txt', 'w')
            fajl.write('%d\n' % len(self.XA))
            fajl.close()

        if self.method == 'hedac':
            self.hedac_search()
        elif self.method == 'smc':
            self.smc_search()
        else:
            print('ERROR\n' * 5)
            print('\n' * 3)
            print('Invalid coverage method')
            print('\n' * 3)

        self.save_fields()
        print('the end!')

    def hedac_search(self):

        print('Search')
        self.sit = 0
        c_cum = np.zeros([self.ny, self.nx])

        for self.it, self.t in enumerate(self.T):
            # old
            xa_all, ya_all = [], []
            for xa, ya in zip(self.XA, self.YA):
                xa_all.extend(xa)
                ya_all.extend(ya)

            apoints = []
            for xa, ya in zip(self.XA, self.YA):
                for xxa, yya in zip(xa[-len(self.FDT):], ya[-len(self.FDT):]):
                    apoints.append([xxa, yya])

            self.c, _, _ = np.histogram2d(ya_all, xa_all, bins=[self.Yc, self.Xc])

            c_cum += self.smoothing(apoints, self.sigma_c)

            if np.mean(c_cum) > 0.0:
                self.cs = c_cum / np.mean(c_cum)
                self.c /= np.mean(self.c)
            else:
                self.cs = np.zeros([self.ny, self.nx])
            # self.c = filters.gaussian_filter(self.c, self.sigma_c/self.dx, truncate=50)

            self.ck = DCT(self.cs)

            self.ac = self.smoothing(apoints, self.sigma_ac)

            self.ac /= np.mean(self.ac)
            self.ac *= np.mean(self.cs) * self.nu
            # self.ac *= np.mean(self.m)

            self.s = self.sourcefun(self.ms, self.cs)

            self.s -= self.ac

            self.convergence()

            i = 0
            self.B = -self.beta * self.s.flatten()
            # for iY in range(self.ny):
            #     for iX in range(self.nx):
            #         self.B[i] = - self.beta * self.s[iY, iX]
            #         i += 1

            u = self.lu.solve(self.B)

            i = 0
            self.U = u.reshape(self.ny, self.nx)
            i += self.ny * self.nx
            # for iY in range(self.ny):
            #     for iX in range(self.nx):
            #         self.U[iY, iX] = u[i]
            #         i += 1

            #
            # Gradient
            #
            self.uy, self.ux = np.gradient(self.U)

            if self.outputStep > 0:
                if self.it % self.outputStep == 0:
                    self.plot_solution()

            vax = intrp.interp2d(self.X, self.Y, self.ux, kind='linear')
            vay = intrp.interp2d(self.X, self.Y, self.uy, kind='linear')

            for xa, ya in zip(self.XA, self.YA):
                va_x = vax(xa[-1], ya[-1])[0]
                va_y = vay(xa[-1], ya[-1])[0]
                van = np.sqrt(va_x ** 2 + va_y ** 2)
                if van < 1e-10:
                    va_x = xa[-1] - xa[-2]
                    va_y = ya[-1] - ya[-2]
                    van = np.sqrt(va_x ** 2 + va_y ** 2)
                    print('zero gradient')
                va_x /= van
                va_y /= van

                pos_eps = 0.0  # 1e-30
                last = len(xa)
                for fdt in self.FDT:  # fraction od dt
                    _xa = xa[last - 1] + fdt * self.dt * self.va * va_x
                    _ya = ya[last - 1] + fdt * self.dt * self.va * va_y
                    if _xa < self.X[0]: _xa = self.X[0] + pos_eps
                    if _xa > self.X[-1]: _xa = self.X[-1] - pos_eps
                    if _ya < self.Y[0]: _ya = self.Y[0] + pos_eps
                    if _ya > self.Y[-1]: _ya = self.Y[-1] - pos_eps
                    xa.append(_xa)
                    ya.append(_ya)
            print('%.5f %8.4e' % (self.t, self.E[-1]))

    def smc_search(self):

        self.s = None

        # lmbd = 0.5
        # self.m = np.maximum( np.log( np.maximum(self.m, 0.0) / lmbd ), 0.0 )
        # self.m /= np.mean(self.m)
        # self.mk = DCT(self.ms)

        print('Search')
        c_cum = np.zeros([self.ny, self.nx])

        for self.it, self.t in enumerate(self.T):
            xa_all, ya_all = [], []
            for xa, ya in zip(self.XA, self.YA):
                xa_all.extend(xa)
                ya_all.extend(ya)

            apoints = []
            for xa, ya in zip(self.XA, self.YA):
                for xxa, yya in zip(xa[-len(self.FDT):], ya[-len(self.FDT):]):
                    apoints.append([xxa, yya])

            self.c, _, _ = np.histogram2d(ya_all, xa_all, bins=[self.Yc, self.Xc])

            c_cum += self.smoothing(apoints, self.sigma_c)

            if np.mean(c_cum) > 0.0:
                self.cs = c_cum / np.mean(c_cum)
                self.c /= np.mean(self.c)
            else:
                self.cs = np.zeros([self.ny, self.nx])
            # self.c = filters.gaussian_filter(self.c, self.sigma_c/self.dx, truncate=50)

            self.ck = DCT(self.cs)

            """
            SMC
            """
            ##Las[iK1][iK2]=1./pow(1.+iK1*iK1+iK2*iK2,1.5) * (_fft_ck1[iK2 * _fft_n + iK1] - _fft_muk1[iK2 * _fft_n + iK1]);
            ky, kx = np.meshgrid(np.arange(self.ny), np.arange(self.nx))

            self.Las = 1.0 / (1 + kx ** 2 + ky ** 2) ** 1.5 * (self.ck - self.mk).T

            self.convergence()

            if self.outputStep > 0:
                if self.it % self.outputStep == 0:
                    self.plot_solution()

            hk = np.ones_like(self.Las) * (self.X[-1] - self.X[0]) * (self.Y[-1] - self.Y[0])
            hk[0, :] *= np.sqrt(2)
            hk[:, 0] *= np.sqrt(2)

            for xa, ya in zip(self.XA, self.YA):
                # print '.',
                kkx = kx * np.pi / (self.X[-1] - self.X[0]) * self.nx / (self.nx + 1);
                kky = ky * np.pi / (self.Y[-1] - self.Y[0]) * self.ny / (self.ny + 1);

                valx = kkx * (xa[-1] - self.X[0] + 0.5 * self.dx)
                valy = kky * (ya[-1] - self.Y[0] + 0.5 * self.dy)

                ax = self.Las * kkx * (-np.sin(valx) * np.cos(valy)) / hk
                ay = self.Las * kky * (np.cos(valx) * -np.sin(valy)) / hk

                va_x = -np.sum(ax)
                va_y = -np.sum(ay)

                van = np.sqrt(va_x ** 2 + va_y ** 2)
                if van < 1e-30:
                    va_x = xa[-1] - xa[-2]
                    va_y = ya[-1] - ya[-2]
                    van = np.sqrt(va_x ** 2 + va_y ** 2)
                    print('zero gradient')
                va_x /= van
                va_y /= van

                pos_eps = 0.0  # 1e-30
                xlast, ylast = xa[-1], ya[-1]
                for fdt in self.FDT:  # fraction od dt
                    _xa = xlast + fdt * self.dt * self.va * va_x
                    _ya = ylast + fdt * self.dt * self.va * va_y
                    if _xa < self.X[0]: _xa = self.X[0] + pos_eps
                    if _xa > self.X[-1]: _xa = self.X[-1] - pos_eps
                    if _ya < self.Y[0]: _ya = self.Y[0] + pos_eps
                    if _ya > self.Y[-1]: _ya = self.Y[-1] - pos_eps
                    xa.append(_xa)
                    ya.append(_ya)

            if self.outputStep > 0:
                fajl = open(self.results_dir + '/agents.txt', 'a')
                fajl.write('%.5f\n' % self.t)
                for xa, ya in zip(self.XA, self.YA):
                    linex = ' '.join(['%.5f' % xx for xx in xa])
                    liney = ' '.join(['%.5f' % yy for yy in ya])
                    fajl.write(linex + '\n')
                    fajl.write(liney + '\n')
                fajl.close()

            print('%.5f %8.4e' % (self.t, self.E[-1]))

    def plot_solution(self):

        plt.figure(figsize=(20, 12))

        gs1 = gridspec.GridSpec(2, 3)
        gs1.update(left=0.02, right=0.98, wspace=0.17, hspace=0.1, top=0.97, bottom=0.05)
        ax1 = plt.subplot(gs1[0, 0])
        ax2 = plt.subplot(gs1[0, 1])
        ax3 = plt.subplot(gs1[0, 2])
        ax4 = plt.subplot(gs1[1, 0])
        ax5 = plt.subplot(gs1[1, 1])
        ax6 = plt.subplot(gs1[1, 2])
        plt.title('t=%.4f' % self.T[self.it])

        # ax1.set_title('Probability density')
        ax1.set_title('Interest density')
        if self.method == 'hedac':
            cntr1 = ax1.contourf(self.X, self.Y, self.ms, 50, cmap=plt.cm.RdBu_r)
        elif self.method == 'smc':
            cntr1 = ax1.contourf(self.X, self.Y, self.ms, 50, cmap=plt.cm.RdBu_r)
        plt.colorbar(cntr1, ax=ax1, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)

        ax2.set_title('Coverage')
        if self.method == 'hedac':
            cntr2 = ax2.contourf(self.X, self.Y, self.cs, 50, cmap=plt.cm.RdBu_r)
        elif self.method == 'smc':
            cntr2 = ax2.contourf(self.X, self.Y, self.cs, 50, cmap=plt.cm.RdBu_r)
        plt.colorbar(cntr2, ax=ax2, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)

        ax3.set_title('Tracking')
        cntr3 = ax3.contourf(self.X, self.Y, self.m, 50, cmap=plt.cm.Purples)
        plt.colorbar(cntr3, ax=ax3, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)
        # ax3.plot(self.XM, self.YM, '.', color='k', alpha=0.5, ms=0.1)
        for xa, ya in zip(self.XA, self.YA):
            ax3.plot(xa, ya, c='orange', lw=1, alpha=0.5)
        for xa, ya in zip(self.XA, self.YA):
            ax3.plot(xa[-1], ya[-1], 'o', c='orange', ms=8)
        ax3.set_xlim(self.X[0], self.X[-1])
        ax3.set_ylim(self.Y[0], self.Y[-1])

        qstep = int(self.nx / 50.0)  # quiver step

        if self.method == 'hedac':
            pass
            levels = np.arange(np.min(self.s) - 0.05, np.max(self.s) + 0.05, 0.025)
            ax4.set_title('Source')
            vmin, vmax = np.min(self.s), np.max(self.s)
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            if vmin > 0:
                vmin = -vmin
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cntr4 = ax4.contourf(self.X, self.Y, self.s, levels, cmap=mpl.colors.ListedColormap(CM / 255), norm=divnorm)
            plt.colorbar(cntr4, ax=ax4, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)
            # sss = self.s
            # sss[sss<0]=0
            # ax4.contourf(self.X, self.Y, sss, 50, cmap = plt.cm.RdBu_r)

            ax5.set_title('Attraction field')
            cntr5 = ax5.contourf(self.X, self.Y, self.U, 50, cmap=plt.cm.coolwarm, alpha=0.95)
            plt.colorbar(cntr5, ax=ax5, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)
            un = np.sqrt(self.uy ** 2 + self.ux ** 2)
            # ax5.contour(self.X, self.Y, self.U, 50, linewidths=0.2, colors='black')
            ax5.quiver(self.X[::qstep], self.Y[::qstep], self.ux[::qstep, ::qstep] / un[::qstep, ::qstep],
                       self.uy[::qstep, ::qstep] / un[::qstep, ::qstep])

        elif self.method == 'smc':
            sss = -IDCT(self.Las).T
            ax4.set_title(r'$m_k - c_k$')
            sk = self.mk[:201, :201] - self.ck[:201, :201]
            divnorm = colors.TwoSlopeNorm(vmin=np.min(sk), vcenter=0., vmax=np.max(sk))
            cntr4 = ax4.imshow(sk, cmap=mpl.cm.seismic, norm=divnorm)
            plt.colorbar(cntr4, ax=ax4, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)

            self.uy, self.ux = np.gradient(sss)
            un = np.sqrt(self.uy ** 2 + self.ux ** 2)
            ax5.set_title('Attraction field')
            cntr5 = ax5.contourf(self.X, self.Y, sss, 150, cmap=plt.cm.coolwarm, alpha=0.95)
            ax5.quiver(self.X[::qstep], self.Y[::qstep], self.ux[::qstep, ::qstep] / un[::qstep, ::qstep],
                       self.uy[::qstep, ::qstep] / un[::qstep, ::qstep])
            plt.colorbar(cntr5, ax=ax5, shrink=0.98, pad=0.02, aspect=50, fraction=0.02)

        ax6.set_title('Convergence')
        ax6.plot(self.T[1:self.it + 1], np.array(self.E[1:]), 'b-', lw=2)
        # ax6.set_xlim(self.T[0], self.T[-1])
        ax6.set_xlim(self.T[1], self.T[-1])
        # print(np.max(self.E))
        ax6.set_ylim(np.min(self.E), np.max(self.E))
        ax6.set_ylabel(r'$\phi^2$')
        ax6.set_xlabel(r't')
        ax6.set_yscale('log')
        # ax6.set_xscale('log')
        ax6.grid(which='minor')
        ax6.grid(which='major')

        plt.savefig(self.results_dir + '/it_%06d.png' % self.it)
        plt.close()

        if self.sit % 1 == 0:
            try:
                print('Creating video ....')
                cmd = f'ffmpeg -y -r 25 -i it_%06d.png -c:v libx264 -r 25 -pix_fmt yuv420p {self.method}_ergodic_coverage.mp4'
                # print(os.getcwd() + '/' + self.results_dir)
                p = subprocess.Popen(cmd.split(), cwd=os.getcwd() + '/' + self.results_dir,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, err = p.communicate()
            except:
                print('Calling ffmpeg for creating video failed')
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(_x, _y)

        surf = ax.plot_surface(X, Y, _u, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0.1, antialiased=True)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        plt.savefig(results_dir + '/surf_%06d.png' % _sit)
        plt.close()
        """

    def save_fields(self):

        if self.method == 'hedac':
            np.savez(self.results_dir + '/fields_final',
                     X=self.X,
                     Y=self.Y,
                     t=self.T[self.it],
                     m=self.m,
                     ms=self.ms,
                     c=self.c,
                     cs=self.cs,
                     s=self.s,
                     u=self.U)

        elif self.method == 'smc':
            np.savez(self.results_dir + '/fields_final',
                     X=self.X,
                     Y=self.Y,
                     t=self.T[self.it],
                     m=self.m,
                     ms=self.ms,
                     c=self.c,
                     cs=self.cs,
                     mk=self.mk,
                     ck=self.ck,
                     u=-IDCT(self.Las).T)

        # Source functions


def logsource(m, c, logeps=1.e-10):
    s = np.log10((logeps + m) / (logeps + c))
    s[m <= 0.0] = 0.0
    s[s <= 0.0] = 0.0
    s /= np.mean(s)
    # s -= ac
    return s


def difsource(m, c):
    s = m - c
    s[s <= 0.0] = 0.0
    s /= np.mean(s)
    # s -= 2*ac
    return s


def difsquaredsource(m, c):
    s = m - c
    s[s <= 0.0] = 0.0
    s = s ** 2
    s /= np.mean(s)
    # s-=ac
    return s


def divsource(m, c, eps=1.e-3):
    s = (eps + m) / (eps + c)
    s /= np.mean(s)
    # s-=ac
    return s


def fullcoveragecource(m, c):
    s = m
    s[c > 0.0] = 0.0
    s /= np.mean(s)
    # s-=ac
    return s


def generate_difpowersource(power=1.0):
    def difpowersource(m, c):
        s = m - c
        s[s <= 0.0] = 0.0
        s = s ** power
        s /= np.mean(s)
        # s-=ac
        return s

    difpowersource.__name__ = '(s-c)^%f' % power
    return difpowersource


def generate_divpowersource(power=1.0, eps=1.0e-3):
    def divpowersource(m, c):
        s = (eps + m) / (eps + c)
        s = s ** power
        s /= np.mean(s)
        # s -=ac
        return s

    divpowersource.__name__ = '((s+%e)/(s+%e)^%f' % (eps, eps, power)
    return divpowersource



if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    # Load the image
    image_path = 'heatmap.png'
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = np.array(image)
    image_height, image_width = image.shape[:2]

    def m(x, y):
        # Get image dimensions
        # Map x, y to image coordinates and sample values
        x_img = int(x * (image_width-1))
        y_img = int(y * (image_height-1))
        sampled_value = image[y_img, x_img]
        return sampled_value
        
    test = HEDAC_basic()

    test.method = 'hedac'
    test.results_dir = 'data/hedac_full'
    test.sigma_m = 0.01
    test.sigma_c = 0.01

    # test.method = 'smc'
    # test.results_dir = 'test_01/smc_full'
    # test.sigma_m = 0.01
    # test.sigma_c = 0.01

    test.X = np.linspace(0, 1, image_height)
    test.Y = np.linspace(0, 1, image_width)
    test.T = np.linspace(0, 1, 5001)

    test.samples = m
    test.beta = 20.0
    test.gamma = 1000.0
    test.va = 20
    test.sigma_ac = 0.01
    # test.kappa = 0.1
    test.sourcefun = difsource
    # logsource, difsource, difsquaredsource, divsource, fullcoveragecource generate_difpowersource(0.5) generate_divpowersource(power=2.0)

    test.outputStep = 5

    # Normalize the image to create a probability distribution
    prob_dist = image / np.sum(image)

    # Flatten the probability distribution and create a list of coordinates
    flat_prob_dist = prob_dist.flatten()
    coordinates = [(i % image_width, i // image_width) for i in range(image_width * image_height)]

    # Sample agent locations based on the probability distribution
    num_agents = 20
    sampled_indices = np.random.choice(len(flat_prob_dist), size=num_agents, p=flat_prob_dist)
    sampled_coordinates = [(coordinates[i][0] / (image_width-1), coordinates[i][1] / (image_height-1)) for i in sampled_indices]

    # Assign the sampled coordinates to the agents
    test.agents = sampled_coordinates

    # test.agents = [[0.2, 0.56], [0.35, 0.53], [0.5, 0.5], [0.65, 0.47], [0.8, 0.44]]

    test.search()
    
    
    # Convert the paths back to the values corresponding in the image
    
    
    
