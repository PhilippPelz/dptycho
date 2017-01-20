# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:34:48 2016

@author: philipp
"""

import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import ifft2, fft2, fftshift,ifftshift
import scipy.ndimage as nd
from pyE17 import utils as u
#from pyE17.io.h5rw import h5read, h5write

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

def focused_probe(E, N, d, alpha_rad, defocus_nm,det_pix = 40e-6, C3_um = 1000, C5_mm=1, tx = 0,ty =0, Nedge = 15, plot=False):
    emass = 510.99906;   # electron rest mass in keV
    hc = 12.3984244;     # h*c
    lam = hc/np.sqrt(E*1e-3*(2*emass+E*1e-3)) # in Angstrom
#    print 'lambda : %g' % lam
    alpha = alpha_rad
    tilt_x = 0
    tilt_y = 0

    phi = {
                '11':30,
                '22':0,
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

    a_dtype = {'names':['10', '11','20', '22',
                             '31','33',
                             '40', '42', '44',
                             '51', '53', '55',
                             '60', '62', '64', '66'],
                    'formats':['f8']*16}
    a0 = np.zeros(1,a_dtype)
    a0['10']=0
    a0['11']=0
    a0['20']=defocus_nm  # defocus: -60 nm
    a0['22']=0
    a0['31']=0
    a0['33']=0
    a0['22']=2.3
    a0['40']=C3_um # C3/spherical aberration: 1000 um
    a0['42']=0
    a0['44']=0
    a0['60']=C5_mm    # C5/Chromatic aberration: 1 mm

    dk = 1.0/(N*d)
    qmax = np.sin(alpha)/lam
    ktm = np.arcsin(qmax*lam)
    detkmax = np.arcsin(lam/(2*d))
    d_alpha = detkmax/(N/2)


    z = det_pix*N*0.5/lam

    print 'alpha         [mrad]     = %g' % (alpha*1000)
    print 'alpha_max det [mrad]     = %g' % (detkmax*1000)

    print 'qmax                     = %g' % qmax
    print 'beam  dmin [Angstrom]    = %g' % (1/qmax)
    print 'dkmax                    = %g' % (dk*N/2)
    print 'detec dmin [Angstrom]    = %g' % (1/(dk*N/2))
    print 'z                 [m]    = %g' % z

    scalekmax = d_alpha*50
    print 'scale bar     [mrad]     = %g' % (scalekmax*1000)

    kx, ky = np.meshgrid(dk*(-N/2.+np.arange(N))+tilt_x,dk*
                               (-N/2.+np.arange(N))+tilt_y)

    k2 = np.sqrt(kx**2+ky**2)
    #riplot(k2,'k2')
    ktheta = np.arcsin(k2*lam)
    kphi = np.arctan2(ky,kx)
    #riplot(ktheta,'ktheta')
    scaled_a = a0.copy().view(np.float64)
    scales =np.array([10, 10,           #a_2x, nm->A
                      10, 10,           #a_2x, nm->A
                      10, 10,             #a_3x, nm->A
                      1E4, 1E4, 1E4,      #a_4x, um->A
                      1E4, 1E4, 1E4,      #a_5x, um->A
                      1E7, 1E7, 1E7, 1E7,  #a_6x, mm->A
                      ],dtype=np.float64)
    scaled_a *= scales
    a = scaled_a.view(a_dtype)

    cos = np.cos
    chi = 2.0*np.pi/lam*(1.0*(a['11']*cos(2*(kphi-phi['11']))+a['10'])*ktheta +                  1.0/2*(a['22']*cos(2*(kphi-phi['22']))+a['20'])*ktheta**2 +
            1.0/3*(a['33']*cos(3*(kphi-phi['33']))+a['31']*cos(1*(kphi-phi['31'])))*ktheta**3 +
            1.0/4*(a['44']*cos(4*(kphi-phi['44']))+a['42']*cos(2*(kphi-phi['42']))+a['40'])*ktheta**4 +
            1.0/5*(a['55']*cos(5*(kphi-phi['55']))+a['53']*cos(3*(kphi-phi['53']))+a['51']*cos(1*(kphi-phi['51'])))*ktheta**5 +
            1.0/6*(a['66']*cos(6*(kphi-phi['66']))+a['64']*cos(4*(kphi-phi['64']))+a['62']*cos(2*(kphi-phi['62']))+a['60'])*ktheta**6)
    #riplot(chi,'chi')
    arr = np.zeros((N,N),dtype=np.complex);
    arr[ktheta<ktm] = 1
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
    rs_mask1 = np.logical_not(sector_mask((N,N),(N/2,N/2),0.05*N,(0,360)))
    # print 'riplot'
    # riplot(rs_mask1)
    # arr*=rs_mask1
#    arr =np.pad(arr,arr.shape,'constant', constant_values=0)
    arr = fftshift(arr)

    arr_real = fftshift(ifft2(arr))
    arr_real /= np.linalg.norm(arr_real)
#    arr_real = nd.zoom(arr_real.real,0.5) + 1j * nd.zoom(arr_real.imag,0.5)

    if plot:
        applot(fftshift(arr),'arr')
        applot(arr_real,'arr_real')
    return np.real(arr_real).astype(np.float32), np.imag(arr_real).astype(np.float32)
def plotcx(x, savePath=None):
    fig, (ax1) = plt.subplots(1,1,figsize=(8,8))
    imax1 = ax1.imshow(u.imsave(x), interpolation='nearest')
    ax1.set_xticks([])
    ax1.set_yticks([])
#    ax1.add_patch(
#        patches.Rectangle(
#            (170, 220),   # (x,y)
#            50,          # width
#            7,          # height
#            facecolor="white"
#        )
#        )
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=400)
    plt.show()

# N=128
# r,i = focused_probe(300e3, N, d = 0.85, alpha_rad=9.8e-3, defocus_nm = 2e2, det_pix = 140e-6, C3_um = 0, C5_mm=0, \
#                     tx = 0,ty =0, Nedge = 5, plot=False)
# pr = r+1j*i
# fpr = fftshift( fft2( ifftshift( pr ) ) )
# plotcx(fpr)
# #pr = nd.gaussian_filter(r,1.2) + 1j * nd.gaussian_filter(i,1.2)
# plotcx(pr)
# h5write('/home/philipp/drop/Public/probe_def_128x128_10mrad.h5',{'pr' : pr.real, 'pi':pr.imag})
# rs_mask1 = np.logical_not(sector_mask((N,N),(N/2,N/2),0.03*N,(0,360)))
# p = a+1j*b
# p *= rs_mask1
# ftp = fftshift(fft2(p))
# #fs_mask1 = np.logical_not(sector_mask((N,N),(N/2,N/2),0.18*N,(0,360)))
# #ftp2 = ftp*fs_mask1
# applot(ftp,'ftp')
#
# applot(ifft2(fftshift(ftp)),'p')
