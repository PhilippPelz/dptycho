# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:34:48 2016

@author: philipp
"""

import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import ifft2, fft2, fftshift

def applot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
    im1, im2 = np.abs(img), np.angle(img)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 13))
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    fig.suptitle(suptitle, fontsize=20)
    imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
    cax1 = div1.append_axes("right", size="10%", pad=0.05)
    cax2 = div2.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)
    ax1.set_title(title[0])
    ax2.set_title(title[1])
    plt.tight_layout()
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)
def riplot(img, suptitle='Image', savePath=None, cmap=['hot','hot'], title=['r','i'], show=True):
    im1, im2 = np.real(img), np.imag(img)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 13))
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    fig.suptitle(suptitle, fontsize=20)
    imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
    cax1 = div1.append_axes("right", size="10%", pad=0.05)
    cax2 = div2.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)
    ax1.set_title(title[0])
    ax2.set_title(title[1])
    plt.tight_layout()
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)

def focused_probe(E, N, d, alpha_rad, defocus_nm, C3_um = 1000, C5_mm=1, tx = 0,ty =0, Nedge = 15, plot=False):
    emass = 510.99906;   # electron rest mass in keV
    hc = 12.3984244;     # h*c
    lam = hc/np.sqrt(E*1e-3*(2*emass+E*1e-3)) # in Angstrom
    alpha = alpha_rad
    tilt_x = 0
    tilt_y = 0

    phi = {
                '22':33.0,
                '33':0.0,
                '44':0.0,
                '55':0.0,
                '66':0.0,
                '31':0.0,
                '42':0.0,
                '53':0.0,
                '64':0.0,
                '51':0.0,
                '62':0.0
            }
    a_dtype = {'names':['20', '22',
                             '31','33',
                             '40', '42', '44',
                             '51', '53', '55',
                             '60', '62', '64', '66'],
                    'formats':['f8']*14}
    a0 = np.zeros(1,a_dtype)
    a0['20']=defocus_nm  # defocus: -60 nm
    a0['22']=2.3
    a0['40']=C3_um # C3/spherical aberration: 1000 um
    a0['42']=2.28
    a0['44']=0.06
    a0['60']=C5_mm    # C5/Chromatic aberration: 1 mm

    dk = 1.0/(N*d)
    qmax = np.sin(alpha)/lam
    kx, ky = np.meshgrid(dk*(-N/2.+np.arange(N))+tilt_x,dk*
                               (-N/2.+np.arange(N))+tilt_y)
    ktm = np.arcsin(qmax*lam)
    k2 = np.sqrt(kx**2+ky**2)
    #riplot(k2,'k2')
    ktheta = np.arcsin(k2*lam)
    kphi = np.arctan2(ky,kx)
    #riplot(ktheta,'ktheta')
    scaled_a = a0.copy().view(np.float64)
    scales =np.array([10, 10,           #a_2x, nm->A
                      10, 10,             #a_3x, nm->A
                      1E4, 1E4, 1E4,      #a_4x, um->A
                      1E4, 1E4, 1E4,      #a_5x, um->A
                      1E7, 1E7, 1E7, 1E7,  #a_6x, mm->A
                      ],dtype=np.float64)
    scaled_a *= scales
    a = scaled_a.view(a_dtype)

    cos = np.cos
    chi = 2.0*np.pi/lam*(1.0/2*(a['22']*cos(2*(kphi-phi['22']))+a['20'])*ktheta**2 +
            1.0/3*(a['33']*cos(3*(kphi-phi['33']))+a['31']*cos(1*(kphi-phi['31'])))*ktheta**3 +
            1.0/4*(a['44']*cos(4*(kphi-phi['44']))+a['42']*cos(2*(kphi-phi['42']))+a['40'])*ktheta**4 +
            1.0/5*(a['55']*cos(5*(kphi-phi['55']))+a['53']*cos(3*(kphi-phi['53']))+a['51']*cos(1*(kphi-phi['51'])))*ktheta**5 +
            1.0/6*(a['66']*cos(6*(kphi-phi['66']))+a['64']*cos(4*(kphi-phi['64']))+a['62']*cos(2*(kphi-phi['62']))+a['60'])*ktheta**6)
    #riplot(chi,'chi')
    arr = np.zeros((N,N),dtype=np.complex);
    arr[ktheta<ktm] = 1 + 1j
    #riplot(arr,'arr')
    dEdge = Nedge/(qmax/dk);  # fraction of aperture radius that will be smoothed
    # some fancy indexing: pull out array elements that are within
    #    our smoothing edges
    ind = np.bitwise_and((ktheta/ktm > (1-dEdge)),
                         (ktheta/ktm < (1+dEdge)))
    arr[ind] = 0.5*(1-np.sin(np.pi/(2*dEdge)*(ktheta[ind]/ktm-1)))
    # add in the complex part
    # MATLAB: probe = probe.*exp(i*chi);
    arr*=np.exp(1j*chi);
    arr = fftshift(arr)
    arr_real = fftshift(ifft2(arr))
    arr_real /= np.linalg.norm(arr_real)
    if plot:
        applot(arr,'arr')
        applot(arr_real,'arr_real')
    return np.real(arr_real).astype(np.float32), np.imag(arr_real).astype(np.float32)

#focused_probe(300e3, 1024, 1, 6e-3, 1e3, C3_um = 0, C5_mm=0, tx = 0,ty =0, Nedge = 25, plot=True)
