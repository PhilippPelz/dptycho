# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:19:35 2016

@author: philipp
"""

from pyE17.utils import imsave
import pyE17.utils.unwrap_levels
import scipy.misc
import scipy as sp
from PIL import Image
from scipy import ndimage as nd
import numpy as np
from math import *
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyE17.io.h5rw import h5read, h5write
from scipy import sparse as sp
from scipy.sparse import linalg as la
from numpy.fft import fft2, fftshift, ifft2, fft, ifft
from scipy import misc
from numpy.linalg import norm

def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

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


def riplot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['r','i'], show=True):
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
#    plt.tight_layout()
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)

def plotcx(x):
    fig, (ax1) = plt.subplots(1,1,figsize=(8,8))
    imax1 = ax1.imshow(imsave(x), interpolation='nearest')
    plt.show()

def blr_probe(N):
    rs_mask = sector_mask((N,N),(N/2,N/2),0.25*N,(0,360))
    rs_mask = rs_mask/norm(rs_mask,1)

    fs_mask1 = sector_mask((N,N),(N/2,N/2),0.25*N,(0,360))
    fs_mask2 = sector_mask((N,N),(N/2,N/2),0.20*N,(0,360))
    fs_mask3 = np.logical_not(sector_mask((N,N),(N/2,N/2),0.10*N,(0,360)))
    #riplot(fs_mask3)
    fs_mask = (fs_mask1.astype(np.int) + 2*fs_mask2.astype(np.int)) * fs_mask3.astype(np.int)
    fs_mask = fs_mask/norm(fs_mask,1)
    #riplot(rs_mask)
    #riplot(fs_mask)

    fs_mask = fftshift(fs_mask)
    phase = fftshift(nd.gaussian_filter(fftshift(np.pi*np.random.uniform(size=(N,N))),2))
    psi_f = fs_mask * np.exp(2j*phase)
    riplot(fftshift(psi_f)        )

    for i in range(100):
        # print i
        psi = ifft2(psi_f,norm='ortho')
    #    applot(psi,'psi')
        psir = psi.real
        psii = psi.imag
    #    psir = nd.gaussian_filter(psi.real,3)
    #    psii = nd.gaussian_filter(psi.imag,3)
        psi = rs_mask * (psir + 1j* psii)
        psi = psi/norm(psi)
    #    plotcx(psi)

        psi_f = fft2(psi,norm='ortho')
    #    applot(psi_f,'psi_f')
    #    plotcx(fftshift(psi_f))
    #    psi_fangle = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
        psi_fangle = np.angle(psi_f)
        psi_f = fs_mask * np.exp(1j*psi_fangle)
        psi_f = psi_f/norm(psi_f)
    #    plotcx(fftshift(psi_f))
    from skimage.restoration import unwrap_phase
    psi = ifft2(psi_f,norm='ortho')
    # plotcx(psi)
    psi_fabs = np.abs(psi_f)

    psi_fangle = unwrap_phase(np.angle(psi_f))
    psi_fu = psi_fabs * np.exp(1j*psi_fangle)
    # plotcx(fftshift(psi_fu))
    return psi.real.astype(np.float32), psi.imag.astype(np.float32)

def blr_probe2(N,rs_rad,fs_rad1,fs_rad2):
    rs_mask = sector_mask((N,N),(N/2,N/2),rs_rad*N,(0,360))
    rs_mask = rs_mask/norm(rs_mask,1)

    fs_mask1 = sector_mask((N,N),(N/2,N/2),fs_rad1*N,(0,360))
    fs_mask2 = sector_mask((N,N),(N/2,N/2),fs_rad2*N,(0,360))
    fs_mask3 = np.logical_not(sector_mask((N,N),(N/2,N/2),fs_rad2*N,(0,360)))
    #riplot(fs_mask3)   + 2*fs_mask2.astype(np.int)
    fs_mask = (fs_mask1.astype(np.int) ) * fs_mask3.astype(np.int)
    fs_mask = fs_mask/norm(fs_mask,1)
    #riplot(rs_mask)
    #riplot(fs_mask)

    fs_mask = fftshift(fs_mask)
    phase = fftshift(nd.gaussian_filter(fftshift(np.pi*np.random.uniform(size=(N,N))),2))
    psi_f = fs_mask * np.exp(2j*phase)
#    riplot(fftshift(psi_f)        )

    for i in range(50):
        # print i
        psi = ifft2(psi_f,norm='ortho')
#        plotcx(psi)
    #    applot(psi,'psi')
        psir = psi.real
        psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
        psi = rs_mask * (psir + 1j* psii)
        psi = psi/norm(psi)
#        plotcx(psi)

        psi_f = fft2(psi,norm='ortho')
#        plotcx(psi_f)
#        applot(psi_f,'psi_f')
#        riplot(fs_mask,'fs_mask')
#        plotcx(fftshift(psi_f))
    #    psi_fangle = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
        psi_fangle = np.angle(psi_f)
        psi_f = fs_mask * np.exp(1j*psi_fangle)
        psi_f = psi_f/norm(psi_f)
#        plotcx(psi_f)

    psi = ifft2(psi_f,norm='ortho')
#        plotcx(psi)
#    applot(psi,'psi')
    psir = psi.real
    psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
    psi = rs_mask * np.exp(1j*np.angle(psi))
    psi = psi/norm(psi)
#        plotcx(psi)

    psi_f = fft2(psi,norm='ortho')


#    from skimage.restoration import unwrap_phase
#    psi = ifft2(psi_f,norm='ortho')
#    plotcx(psi)
    # applot(psi)

#    psia = nd.gaussian_filter(np.abs(psi),5)
#    psip = nd.gaussian_filter(np.angle(psi),5)
#    psi2 = psia * np.exp(1j*psip)
##    plotcx(psi2)
#    angles = np.digitize(np.angle(psi2),np.arange(10)*np.pi/10) * np.pi/10
#    psi3 = np.abs(psi2) * np.exp(1j*angles)
#    plotcx(psi3)

#    plotcx(psi)
#    psi_fabs = np.abs(psi_f)
#
#    psi_fangle = unwrap_phase(np.angle(psi_f))
#    psi_fu = psi_fabs * np.exp(1j*psi_fangle)
    applot(fftshift(psi_f))
    return psi.real.astype(np.float32), psi.imag.astype(np.float32)

def blr_probe3(N,rs_rad,fs_rad1,fs_rad2):
    divs = 30
    rs_mask = sector_mask((N,N),(N/2,N/2),rs_rad*N,(0,360))
    rs_mask = rs_mask/norm(rs_mask,1)

    fs_mask = np.zeros_like(rs_mask,dtype = np.float32)

    for i in np.arange(0,divs,3):
        fs_mask += sector_mask((N,N),(N/2,N/2),fs_rad1*N,(360.0/divs*i,360.0/divs*(i+1)))

#    fs_mask1 = sector_mask((N,N),(N/2,N/2),fs_rad1*N,(0,360))
    fs_mask3 = np.logical_not(sector_mask((N,N),(N/2,N/2),fs_rad2*N,(0,360)))
    fs_mask = (fs_mask.astype(np.int) ) * fs_mask3.astype(np.int)
    fs_mask = fs_mask/norm(fs_mask,1)
    #riplot(rs_mask)
    riplot(fs_mask)

    fs_mask = fftshift(fs_mask)
    phase = fftshift(nd.gaussian_filter(fftshift(np.pi*np.random.uniform(size=(N,N))),2))
    psi_f = fs_mask * np.exp(2j*phase)
#    riplot(fftshift(psi_f)        )

    for i in range(50):
        print i
        psi = ifft2(psi_f,norm='ortho')
#        plotcx(psi)
    #    applot(psi,'psi')
        psir = psi.real
        psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
        psi = rs_mask * (psir + 1j* psii)
        psi = psi/norm(psi)
#        plotcx(psi)

        psi_f = fft2(psi,norm='ortho')
#        plotcx(psi_f)
#        applot(psi_f,'psi_f')
#        riplot(fs_mask,'fs_mask')
#        plotcx(fftshift(psi_f))
    #    psi_fangle = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
        psi_fangle = np.angle(psi_f)
        psi_f = fs_mask * np.exp(1j*psi_fangle)
        psi_f = psi_f/norm(psi_f)
#        plotcx(psi_f)

    psi = ifft2(psi_f,norm='ortho')
#        plotcx(psi)
#    applot(psi,'psi')
    psir = psi.real
    psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
    psi = rs_mask * np.exp(1j*np.angle(psi))
    psi = psi/norm(psi)
#        plotcx(psi)

    psi_f = fft2(psi,norm='ortho')


#    from skimage.restoration import unwrap_phase
#    psi = ifft2(psi_f,norm='ortho')
#    plotcx(psi)
    applot(psi)

#    psia = nd.gaussian_filter(np.abs(psi),5)
#    psip = nd.gaussian_filter(np.angle(psi),5)
#    psi2 = psia * np.exp(1j*psip)
##    plotcx(psi2)
#    angles = np.digitize(np.angle(psi2),np.arange(10)*np.pi/10) * np.pi/10
#    psi3 = np.abs(psi2) * np.exp(1j*angles)
#    plotcx(psi3)

#    plotcx(psi)
#    psi_fabs = np.abs(psi_f)
#
#    psi_fangle = unwrap_phase(np.angle(psi_f))
#    psi_fu = psi_fabs * np.exp(1j*psi_fangle)
    applot(fftshift(psi_f))
    return psi.real.astype(np.float32), psi.imag.astype(np.float32)
# blr_probe2(384)
