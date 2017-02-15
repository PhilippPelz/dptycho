# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:37:15 2016

@author: philipp
"""
from pyE17.utils import imsave
import pyE17.utils.unwrap_levels
from pyE17.io.h5rw import h5read, h5write
import pyE17.utils as u
import scipy.misc
import scipy as sp
from PIL import Image
from scipy import ndimage as nd
import numpy as np
from math import *
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import sparse as sp
from scipy.sparse import linalg as la
from numpy.fft import fft2, ifftshift, fftshift, ifft2, fft, ifft
from scipy import misc
from numpy.linalg import norm
import matplotlib.patches as patches
import os  
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.misc as m
import skimage.data as d
def applot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
    im1, im2 = np.abs(img), np.angle(img)
    im2 -= np.min(im2)
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
def rplot(r,barsize,barlabel, savePath=None, title='', cm=plt.cm.get_cmap('gray')):
    fig, ax = plt.subplots(1,1)
    ang = r
    imax1 = ax.imshow(ang, interpolation='nearest', cmap=cm)
    cbar1 = plt.colorbar(imax1)
    ax.set_title(title)
    ax.add_patch(
    patches.Rectangle(
        (170, 200),   # (x,y)
        barsize,          # width
        7,          # height
        facecolor="white"
    )
    )   
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    plt.axis('off')
    if savePath is not None:
        fig.savefig(savePath + '.png', dpi=300)    
    
radius = 0
cam = d.camera().astype(np.float)/d.camera().max()
z = np.array(d.coins().shape).astype(np.float)/np.array(d.camera().shape)
im = nd.zoom(d.coins().astype(np.float)/d.coins().max()/50,1/z) * np.exp( 1j * d.camera().astype(np.float)/d.camera().max())
im = np.exp(1j*cam*0.1)
fim = fftshift(fft2(ifftshift(im)))
ring = sector_mask(fim.shape,np.array(fim.shape)/2,radius,(0,360)) 
mask = np.exp(1j * np.pi/2) * ring + np.logical_not(ring)* np.exp(1j * 0) 
fim *= mask     
im1 = fftshift(ifft2(ifftshift(fim))) 

applot(im,'im')
applot(np.log10(np.abs(fim))*np.exp(1j*np.angle(fim)),'fim')
rplot(ring,1,'')
applot(mask,'mask', cmap=['hot','gray'])
applot(im1,'im1')
