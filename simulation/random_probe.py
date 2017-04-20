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
from numpy.fft import fft2, ifftshift, fftshift, ifft2, fft, ifft
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
    anglemask = theta < (tmax-tmin)

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

#def plotcx(x, savePath=None):
#    fig, (ax1) = plt.subplots(1,1,figsize=(8,8))
#    imax1 = ax1.imshow(imsave(x), interpolation='nearest')
#    if savePath is not None:
#        # print 'saving'
#        fig.savefig(savePath + '.png', dpi=300)
#    plt.show()
def plotcx(x, savePath=None):
    fig, (ax1) = plt.subplots(1,1,figsize=(8,8))
    imax1 = ax1.imshow(imsave(x))
    ax1.set_xticks([])
    ax1.set_yticks([])
#    ax1.add_patch(
#        patches.Rectangle(
#            (170, 220),   # (x,y)
#            50,          # width
#            7,          # height
#            facecolor="white",
#            edgecolor="white"
#        )
#        )
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath, dpi=300)
    plt.show()
def fzp(N,a,bandlimit = None):
    rmax = 0.49*N
    r = 0
    res = np.zeros((N,N),dtype=np.float)
    lastring = 0
    for n in np.arange(0,0.75*N/2,2):
        inner = np.logical_not(sector_mask((N,N),(N/2,N/2),sqrt(a*n),(0,360)))
        if n==0: inner = np.ones_like(inner)
        outer = sector_mask((N,N),(N/2,N/2),sqrt(a*(n+1)),(0,360))

        ring = inner.astype(np.float) * outer.astype(np.float)

        res += ring
#        riplot(res)
        if sqrt(a*(n+4)) > rmax:
            lastring = sqrt(a*(n+1))
            break
#    riplot(res,'rings')
    res1 = nd.gaussian_filter(sector_mask((N,N),(N/2,N/2),int(lastring),(0,360)).astype(np.float),1)*np.exp(1j*np.pi/4*res)
    if bandlimit is not None:
        res1 *= np.logical_not(sector_mask((N,N),(N/2,N/2),int(bandlimit * N),(0,360)))
#    plotcx(fftshift(ifft2(res1)))
    plotcx(res1)
    return np.real(res1).astype(np.float32), np.imag(res1).astype(np.float32)

def random_fzp(N,a,sections):
    rmax = N/2
    res = np.zeros((N,N),dtype=np.float)
    for n in np.arange(0,0.75*N/2,2):
        for k in np.arange(0,sections+n):
            angle_step = 360/(sections+n)
            inner = np.logical_not(sector_mask((N,N),(N/2,N/2),sqrt(a*n),(angle_step*k,angle_step*(k+1))))
            if n==0: inner = np.ones_like(inner)
            outer = sector_mask((N,N),(N/2,N/2),sqrt(a*(n+1)),(angle_step*k,angle_step*(k+1)))

            ring = inner.astype(np.float) * outer.astype(np.float) * np.random.randint(1,13)

            res += ring
#        riplot(res)
        if sqrt(a*(n+4)) > rmax: break
#    riplot(res,'rings')
    res1 = sector_mask((N,N),(N/2,N/2),int(rmax),(0,360))*np.exp(1j*np.pi/12*res)
    plotcx(fftshift(ifft2(res1)))
    plotcx(res1)
    return np.real(res1).astype(np.float32), np.imag(res1).astype(np.float32)

def random_fzp2(N,a,sections):
    rmax = N/2
    res = np.zeros((N,N),dtype=np.float)
    for n in np.arange(0,0.75*N/2,1):
        for k in np.arange(0,sections+n):
            angle_step = 360/(sections+n)
            inner = np.logical_not(sector_mask((N,N),(N/2,N/2),sqrt(a*n),(angle_step*k,angle_step*(k+1))))
            if n==0: inner = np.ones_like(inner)
            outer = sector_mask((N,N),(N/2,N/2),sqrt(a*(n+1)),(angle_step*k,angle_step*(k+1)))

            ring = inner.astype(np.float) * outer.astype(np.float) * np.random.randint(1,13)

            res += ring
#        riplot(res)
        if sqrt(a*(n+4)) > rmax: break
#    riplot(res,'rings')
    res1 = sector_mask((N,N),(N/2,N/2),int(rmax),(0,360))*np.exp(1j*np.pi/12*res)
    plotcx(fftshift(ifft2(res1)))
    plotcx(res1)
    return np.real(res1).astype(np.float32), np.imag(res1).astype(np.float32)
def random_fzp21(N,a,sections):
    rmax = N/2
    res = np.zeros((N,N),dtype=np.float)
    for n in np.arange(0,0.75*N/2,1):
        for k in np.arange(0,sections+n):
            angle_step = 360/(sections+n)
            inner = np.logical_not(sector_mask((N,N),(N/2,N/2),sqrt(a*n),(angle_step*k,angle_step*(k+1))))
            if n==0: inner = np.ones_like(inner)
            outer = sector_mask((N,N),(N/2,N/2),sqrt(a*(n+1)),(angle_step*k,angle_step*(k+1)))

            ring = inner.astype(np.float) * outer.astype(np.float) * np.random.randint(1,13)

            res += ring
#        riplot(res)
        if sqrt(a*(n+4)) > rmax: break
#    riplot(res,'rings')
    res1 = sector_mask((N,N),(N/2,N/2),int(rmax),(0,360))*np.exp(1j*np.pi/12*res)
    plotcx(fftshift(ifft2(res1)))
    plotcx(res1)
    return res1
def gr_probe(N):
    xx,yy = np.mgrid[:N,:N]
    r2 = xx**2 + yy**2
    g = np.exp(-500*r2)
    riplot(g)
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
    plotcx(psi)
    psi_fabs = np.abs(psi_f)

    psi_fangle = unwrap_phase(np.angle(psi_f))
    psi_fu = psi_fabs * np.exp(1j*psi_fangle)
    # plotcx(fftshift(psi_fu))
    return np.real(psi).astype(np.float32), np.imag(psi).astype(np.float32)

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
    plotcx(psi)
    psi_fabs = np.abs(psi_f)

    psi_fangle = unwrap_phase(np.angle(psi_f))
    psi_fu = psi_fabs * np.exp(1j*psi_fangle)
    plotcx(fftshift(psi_fu))
    return np.real(psi).astype(np.float32), np.imag(psi).astype(np.float32)
def build_checkerboard(w, h) :
    re = np.r_[ w*[0,1] ]        # even-numbered rows
    ro = np.r_[ w*[1,0] ]        # odd-numbered rows
    return np.row_stack(h*(re, ro))
def blr_probe2(N,rs_rad,fs_rad1,fs_rad2,do_plot=False):
    rs_mask = sector_mask((N,N),(N/2,N/2),rs_rad*N,(0,360))
    # rs_mask = nd.gaussian_filter(rs_mask.astype(np.float),10)
    rs_mask = rs_mask/norm(rs_mask,1)

    fs_mask1 = sector_mask((N,N),(N/2,N/2),fs_rad1*N,(0,360))
    fs_mask2 = sector_mask((N,N),(N/2,N/2),fs_rad2*N,(0,360))
    fs_mask3 = np.logical_not(sector_mask((N,N),(N/
    2,N/2),fs_rad2*N,(0,360)))
    #riplot(fs_mask3)   + + 3*fs_mask2.astype(np.int)
    fs_mask = (fs_mask1.astype(np.int) ) * fs_mask3.astype(np.int)
    fs_mask = fs_mask/norm(fs_mask,1)
    sh = fs_mask.shape
#    fs_mask[sh[0]/2-1:sh[0]/2+1,:] = 0
#    fs_mask[:,sh[0]/2-1:sh[0]/2+1] = 0
    #riplot(rs_mask)
    #riplot(fs_mask)

#     h = fftshift( ifft2( ifftshift( H ) ) );
#     H = fftshift( fft2( ifftshift( h ) ) );

    fs_mask = fftshift(fs_mask)
#    phase = fftshift(np.angle(random_fzp21(N,130,5)))
    phase = 2 * np.pi * np.random.uniform(size=(N,N))
    psi_f = fftshift(fs_mask * np.exp(2j*phase))
#    rs_mask = fftshift(rs_mask)
#    fs_mask = fftshift(fs_mask)
    psi_f = ifftshift(psi_f)
    if do_plot:
        applot(psi_f,'psi_f init')
    it = 50
    for i in range(it):
#        print i
        psi = fftshift(ifft2(psi_f,norm='ortho'))


        if do_plot:
            applot(psi,'psi')
        psir = psi.real
        psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
        psi = rs_mask *(psir + 1j* psii)# np.exp(1j*np.angle(psi))#(psir + 1j* psii)
        psi = psi/norm(psi)
#        plotcx(psi)
        if do_plot:
            applot(psi,'psi masked')


        psi_f = fft2(ifftshift(psi),norm='ortho')
        if do_plot:
            applot(psi_f,'psi_f')
#        plotcx(fftshift(psi_f))
#        psi_fangle = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
#        psi_fangle = nd.zoom(nd.zoom(np.angle(psi_f),0.5),2)
        psi_fangle = np.angle(psi_f)
        # riplot(psi_fangle)
        # psi_fangle[b] += np.pi/2
        # riplot(psi_fangle)
        psi_f = fs_mask * np.exp(1j*psi_fangle)
        # phase = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
        # psi_f = np.abs(psi_f) * np.exp(1j*phase)
        psi_f = psi_f/norm(psi_f)
#        plotcx(psi_f)
        if do_plot:
            applot(psi_f,'psi_f masked')

    psi = fftshift(fft2(psi_f,norm='ortho'))
#    applot(psi,'final psi')
#    psi_f = fft2(ifftshift(psi))
#    applot(psi_f,'final psi_f')
#    psi_fangle = np.angle(psi_f)
#    psi_f = fs_mask * np.exp(1j*psi_fangle)
##    applot(psi_f,'final psi_f masked')
#    psi_f = ifftshift(psi_f)
#    applot(psi_f,'final psi_f masked shifted')
#    psi = ifft2(psi_f,norm='ortho')
#    plotcx(psi)
#    bins = 150
#    digi = np.digitize(np.angle(psi_f),np.linspace(-1,1,bins)*2*np.pi)
#    riplot(digi,'digi')
#    psi_f = np.abs(psi_f) * np.exp(1j*digi*1/bins*2.0*np.pi)
#    psi = fft2(psi_f,norm='ortho')



    # psi_f = np.abs(psi_f) * np.exp(1j*nd.zoom(nd.zoom(np.angle(psi_f),0.5),2))
    # psi = fft2(psi_f,norm='ortho')
#    psi = fftshift(psi)
    plotcx(fftshift(psi_f),'/home/philipp/drop/Public/phase_plate')
    plotcx(psi)
    applot(psi)
    plotcx(fftshift( fft2( ifftshift( psi ) ) ))

    return np.real(psi).astype(np.float32), np.imag(psi).astype(np.float32)
    
def blr_probe4(N,rs_rad,fs_rad1,fs_rad2,do_plot=False):
    rs_mask = sector_mask((N,N),(N/2,N/2),rs_rad*N,(0,360))
    # rs_mask = nd.gaussian_filter(rs_mask.astype(np.float),10)
    rs_mask = rs_mask/norm(rs_mask,1)

    fs_mask1 = sector_mask((N,N),(N/2,N/2),fs_rad1*N,(0,360))
    fs_mask2 = sector_mask((N,N),(N/2,N/2),fs_rad2*N,(0,360))
    fs_mask3 = np.logical_not(sector_mask((N,N),(N/
    2,N/2),fs_rad2*N,(0,360)))
    #riplot(fs_mask3)   + + 3*fs_mask2.astype(np.int)
    fs_mask = (fs_mask1.astype(np.int) ) * fs_mask3.astype(np.int)
    fs_mask = fs_mask/norm(fs_mask,1)
    sh = fs_mask.shape
#    fs_mask[sh[0]/2-1:sh[0]/2+1,:] = 0
#    fs_mask[:,sh[0]/2-1:sh[0]/2+1] = 0
    #riplot(rs_mask)
    #riplot(fs_mask)

#     h = fftshift( ifft2( ifftshift( H ) ) );
#     H = fftshift( fft2( ifftshift( h ) ) );

    fs_mask = fftshift(fs_mask)
#    phase = fftshift(np.angle(random_fzp21(N,130,5)))
    phase = 2 * np.pi * np.random.uniform(size=(N,N))
    psi_f = fftshift(fs_mask * np.exp(2j*phase))
#    rs_mask = fftshift(rs_mask)
#    fs_mask = fftshift(fs_mask)
    psi_f = ifftshift(psi_f)
    if do_plot:
        applot(psi_f,'psi_f init')
    it = 50
    for i in range(it):
#        print i
        psi = fftshift(ifft2(psi_f,norm='ortho'))


        if do_plot:
            applot(psi,'psi')
#        psir = psi.real
#        psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
        psi = rs_mask * np.exp(1j*np.angle(psi))#(psir + 1j* psii)
        psi = psi/norm(psi)
#        plotcx(psi)
        if do_plot:
            applot(psi,'psi masked')

        psi_f = fft2(ifftshift(psi),norm='ortho')
        if do_plot:
            applot(psi_f,'psi_f')
#        plotcx(fftshift(psi_f))
#        psi_fangle = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
#        psi_fangle = nd.zoom(nd.zoom(np.angle(psi_f),0.5),2)
        psi_fangle = np.angle(psi_f)
        # riplot(psi_fangle)
        # psi_fangle[b] += np.pi/2
        # riplot(psi_fangle)
        psi_f = fs_mask * psi_f
        # phase = fftshift(nd.gaussian_filter(fftshift(np.angle(psi_f)),1))
        # psi_f = np.abs(psi_f) * np.exp(1j*phase)
        psi_f = psi_f/norm(psi_f)
#        plotcx(psi_f)
        if do_plot:
            applot(psi_f,'psi_f masked')

    psi = fftshift(fft2(psi_f,norm='ortho'))
#    applot(psi,'final psi')
#    psi_f = fft2(ifftshift(psi))
#    applot(psi_f,'final psi_f')
#    psi_fangle = np.angle(psi_f)
#    psi_f = fs_mask * np.exp(1j*psi_fangle)
##    applot(psi_f,'final psi_f masked')
#    psi_f = ifftshift(psi_f)
#    applot(psi_f,'final psi_f masked shifted')
#    psi = ifft2(psi_f,norm='ortho')
#    plotcx(psi)
#    bins = 150
#    digi = np.digitize(np.angle(psi_f),np.linspace(-1,1,bins)*2*np.pi)
#    riplot(digi,'digi')
#    psi_f = np.abs(psi_f) * np.exp(1j*digi*1/bins*2.0*np.pi)
#    psi = fft2(psi_f,norm='ortho')



    # psi_f = np.abs(psi_f) * np.exp(1j*nd.zoom(nd.zoom(np.angle(psi_f),0.5),2))
    # psi = fft2(psi_f,norm='ortho')
#    psi = fftshift(psi)
    plotcx(fftshift(psi_f),'/home/philipp/drop/Public/phase_plate')
    plotcx(psi)
    applot(psi)
    plotcx(fftshift( fft2( ifftshift( psi ) ) ))

    return np.real(psi).astype(np.float32), np.imag(psi).astype(np.float32)

def blr_probe3(N,rs_rad,fs_rad1,fs_rad2):
    divs = 60
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
    phase = np.ones_like(fs_mask)#fftshift(nd.gaussian_filter(fftshift(np.pi*np.random.uniform(size=(N,N))),2))
    psi_f = fs_mask * np.exp(2j*phase)
    riplot(fftshift(psi_f)        )

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
#    psir = psi.real
#    psii = psi.imag
#        psir = nd.gaussian_filter(psi.real,5)
#        psii = nd.gaussian_filter(psi.imag,5)
#    psi = rs_mask * np.exp(1j*np.angle(psi))
#    psi = psi/norm(psi)
#        plotcx(psi)

#    psi_f = fft2(psi,norm='ortho')


#    from skimage.restoration import unwrap_phase
#    psi = ifft2(psi_f,norm='ortho')
    # plotcx(psi_f)
    # applot(psi)

#    psia = nd.gaussian_filter(np.abs(psi),5)
#    psip = nd.gaussian_filter(np.angle(psi),5)
#    psi2 = psia * np.exp(1j*psip)
##    plotcx(psi2)
#    angles = np.digitize(np.angle(psi2),np.arange(10)*np.pi/10) * np.pi/10
#    psi3 = np.abs(psi2) * np.exp(1j*angles)
#    plotcx(psi3)

    # plotcx(psi)
#    psi_fabs = np.abs(psi_f)
#
#    psi_fangle = unwrap_phase(np.angle(psi_f))
#    psi_fu = psi_fabs * np.exp(1j*psi_fangle)
    # applot(fftshift(psi_f))
    return np.real(psi).astype(np.float32), np.imag(psi).astype(np.float32)

def rgb2hsv(rgb):
    """
    Reverse to :any:`hsv2rgb`
    """
    eps = 1e-6
    rgb=np.asarray(rgb).astype(float)
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    s = (maxc-minc) / (maxc+eps)
    s[maxc<=eps]=0.0
    rc = (maxc-rgb[:,:,0]) / (maxc-minc+eps)
    gc = (maxc-rgb[:,:,1]) / (maxc-minc+eps)
    bc = (maxc-rgb[:,:,2]) / (maxc-minc+eps)

    h =  4.0+gc-rc
    maxgreen = (rgb[:,:,1] == maxc)
    h[maxgreen] = 2.0+rc[maxgreen]-bc[maxgreen]
    maxred = (rgb[:,:,0] == maxc)
    h[maxred] = bc[maxred]-gc[maxred]
    h[minc==maxc]=0.0
    h = (h/6.0) % 1.0

    return np.asarray((h, s, v))

def hsv2complex(cin):
    """
    Reverse to :any:`complex2hsv`
    """
    h,s,v = cin
    return v * np.exp(np.pi*2j*(h-.5)) /v.max()

def rgb2complex(rgb):
    """
    Reverse to :any:`complex2rgb`
    """
    return hsv2complex(rgb2hsv(rgb))
    
#r,i = blr_probe4(256,0.20,0.3,0.00)
#r,i = fzp(256,500,6)
##pr = r + 1j* i
#h5write('/home/philipp/drop/Public/fzp1.h5',{'pr' : r, 'pi':i})
#plotcx(pr,'/home/philipp/drop/Philipp/mypapers/lowdose/data/figure5_probes/blr_fourier.eps')
#fpr = fftshift( fft2( ifftshift( pr ) ) )
#plotcx(fpr)
#f,ax = plt.subplots()
#ax.hist(np.abs(pr))
#plt.show()
# r, i = fpr.real, fpr.imag
# f=plt.figure(figsize=(10, 13))
# plt.hist(r[r!=0],bins=20)
# plt.show()
# f=plt.figure(figsize=(10, 13))
# plt.hist(i[i!=0],bins=20)
# plt.show()
#im = nd.imread('/home/philipp/test.png')
#imcx = rgb2complex(im)
#imcx = nd.zoom(imcx.real,0.2) + 1j* nd.zoom(imcx.imag,0.2)
#applot(imcx)
#applot(fftshift(fft2(imcx)))
#print im.shape
#pp = random_fzp(256,800)
#focus = focused_probe(300e3, 256, 2.5, 3e-3, 3.5e3, C3_um = 1000, C5_mm=1, tx = 0,ty =0, Nedge = 15, plot=True)
#f= focus[0] + 1j*focus[1]
#plotcx(pp)
#plotcx(np.log10(np.abs(fftshift(fft2(pp)))))
#gr_probe(384)
# file1 = '/home/philipp/projects/papers/lowdose/data/4v6x/fzp/sgd/00010_s_4V6X_200_ov_72_d_145000000_nu_0p15_run_1_TWF_64.h5'
# r,i = fzp(256,300)
# p = r + 1j*i
# plotcx(fftshift( fft2( ifftshift( p ) ) ),os.path.dirname(file1) + '/fzp_probe_fourier')
# plotcx(p,os.path.dirname(file1) + '/fzp_probe')
def taperarray(N,edge):
    xx,yy = np.mgrid[0:N,0:N]
    xx1 = np.flipud(xx)
    xx2 = np.minimum(xx,xx1)
    yy1 = np.fliplr(yy)
    yy2 = np.minimum(yy,yy1)
    rr = np.minimum(xx2,yy2).astype(np.float)

    rr[rr<=edge] /= edge
    rr[rr>edge] = 1
    rr *= np.pi/2
    rr = np.sin(rr)
    print xx2
    fig, ax = plt.subplots()
    # imax = ax.imshow(rr)
    # plt.colorbar(imax)
    # plt.show()

# taperarray(50,10)
