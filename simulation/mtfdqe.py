# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:25:14 2016

@author: philipp pelz
"""

import pandas as pd
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt

from scipy.ndimage import filters as filter
from math import *
import scipy.io
import scipy.ndimage
from scipy import interpolate as i

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
def MTF_DQE_2D(cam,binning,s,path):
    if cam == 'K2':
        df = pd.read_csv(path + '/K2Count3eps2.csv')
        dqe = df['DQE'].values
        mtf = df['MTF'].values
        q = df['Frequency'].values
        from scipy import interpolate as i
        size = np.array((3840,3712))
        bsize = size / binning
        dqef = i.interp1d(q,dqe,kind='quadratic')
        mtff = i.interp1d(q,mtf,kind='quadratic')

        xx,yy = np.mgrid[-s/2:s/2,-s/2:s/2]
        rr = np.sqrt(xx**2 + yy**2)
        qq = rr / np.max(size)

        mtfq = mtff(qq)
        dqeq = dqef(qq)
        mtf2D = fftshift(mtfq)
        dqe2D = fftshift(filter.gaussian_filter(dqeq,3))

        f, ax = plt.subplots(1,1,figsize=(10,7))
        # cax = ax.imshow(mtf2D)
        # plt.colorbar(cax)
        # plt.show()

        f, ax = plt.subplots(1,1,figsize=(10,7))
        # cax = ax.imshow(dqe2D)
        # plt.colorbar(cax)
        # plt.show()

        return mtf2D, dqe2D

def MTF_DQE_2D2(cam,binning,s,path):
    if cam == 'K2':
        df = pd.read_csv(path + '/K2Count3eps20.csv')
        dqe = df['DQE'].values
        mtf = df['MTF'].values
        q = df['Frequency'].values

        size = np.array((3712,3712))
        bsize = size / binning
        dqef = i.interp1d(q,dqe,kind='cubic')
        mtff = i.interp1d(q,mtf,kind='cubic')

        xx,yy = np.mgrid[-s/2:s/2,-s/2:s/2]
        rr = np.sqrt(xx**2 + yy**2)
        qq = rr / (s/2)

        mtfq = mtff(qq)
        dqeq = dqef(qq)

        scipy.io.savemat('/home/philipp/drop/Public/InSilicoTEM/code/MTFs/DQE_K2Summit_300.mat', mdict={'dqe': dqeq})
        scipy.io.savemat('/home/philipp/drop/Public/InSilicoTEM/code/MTFs/MTF_K2Summit_300.mat', mdict={'mtf': mtfq})

        mtf2D = fftshift(mtfq)
        dqe2D = fftshift(dqeq)

        # f, ax = plt.subplots(1,1,figsize=(10,7))
        # cax = ax.imshow(mtf2D)
        # plt.colorbar(cax)
        # plt.show()

        # f, ax = plt.subplots(1,1,figsize=(10,7))
        # cax = ax.imshow(dqe2D)
        # plt.colorbar(cax)
        # plt.show()

        return mtf2D, dqe2D

def round_scan_ROI_positions(dr, lx, ly, nth):
    """\
    Round scan positions with ROI, defined as in spec and matlab.
    """
    rmax = np.sqrt( (lx/2)**2 + (ly/2)**2 )
    nr = np.floor(rmax/dr) + 1
    positions = []
    for ir in range(1,int(nr+2)):
        rr = ir*dr
        dth = 2*np.pi / (nth*ir)
        th = 2*np.pi*np.arange(nth*ir)/(nth*ir)
        x1 = rr*np.sin(th)
        x2 = rr*np.cos(th)
        positions.extend([(xx1,xx2) for xx1,xx2 in zip(x1,x2) if (np.abs(xx1) <= ly/2) and (np.abs(xx2) <= lx/2)])
    positions.extend([(0,0)])
    pos = np.array(positions)
    f,a = plt.subplots(figsize=(5,5))
    a.scatter(pos[:,0],pos[:,1])
    plt.show()
    pos -= np.min(pos,axis=0)
    return pos

def spiral_scan_ROI_positions(dr,lx,ly):
    """\
    Spiral scan positions. ROI
    """
    alpha = np.sqrt(4*np.pi)
    beta = dr/(2*np.pi)

    rmax = .5*np.sqrt(lx**2 + ly**2)
    positions = []
    for k in xrange(1000000000):
        theta = alpha*np.sqrt(k)
        r = beta * theta
        if r > rmax: break
        x,y = r*np.sin(theta), r*np.cos(theta)
        if abs(x) > lx/2: continue
        if abs(y) > ly/2: continue
        positions.append( (x,y) )
    return positions

def raster_positions(npos,size):
    s = sqrt(2) * size /2
    s2 = sqrt(2) * size /2
    n = int(2 * int(sqrt(npos)))
    phi = 44

    while True:
        ar = np.linspace(-s2,s2,n).astype(np.int)
        x = np.repeat(ar,n)
        y = np.tile(ar,n)
        pos = np.zeros((n**2,2))
        pos[:,0] = x
        pos[:,1] = y

        theta = np.radians(40)
        c, s1 = np.cos(theta), np.sin(theta)
        R = np.matrix([[c, -s1], [s1, c]])
        pos = pos.dot(R)
        pos += np.array([2,5])
        valid = np.logical_and(np.abs(pos[:,0]) < size/2 , np.abs(pos[:,1]) < size/2)
        nvalid = valid.sum()
        valid1 = np.tile(valid.reshape((n**2,1)),(1,2))
        posv = pos[valid1].reshape((nvalid,2))
        # print 'nvalid = %d' % nvalid
#        f, ax = plt.subplots(1,1,figsize=(10,7))
#        cax = ax.scatter(posv[:,0],posv[:,1])
#        plt.show()
        if nvalid <= npos:
            break
    #    phi += 2
#        print 'phi = %d, s = %d' % (phi,s2)
    #    if phi >= 90:
        s2 += 1
    #        phi %= 90

    posv -= np.min(posv)
#    print pos
    return posv

def raster_positions_overlap(size,probe_mask,overlap,pos_shift=[2,5],phi=44,random_max=None):
    s = sqrt(2) * size /2
    s2 = sqrt(2) * size /2
    n = int(2 * int(sqrt(4000)))

    sh = probe_mask.shape
    ov = np.zeros(tuple(np.array(sh)*2))
    def r(size,overlap):
        phi = 40 + np.random.randint(-2,2)
        n = int(2 * int(sqrt(4000)))
        s = sqrt(2) * size /2
        s2 = sqrt(2) * size /2
        while True:
            ov[:] = 0
            ar = np.linspace(-s2,s2,n).astype(np.int)
            x = np.repeat(ar,n)
            y = np.tile(ar,n)
            pos = np.zeros((n**2,2))
            pos[:,0] = x
            pos[:,1] = y

            theta = np.radians(phi)
            c, s1 = np.cos(theta), np.sin(theta)
            R = np.matrix([[c, -s1], [s1, c]])
            pos = pos.dot(R)
            pos += np.array(pos_shift)
            if random_max is not None:
                r = np.random.randint(-random_max,random_max,pos.shape)
#                print r
                pos += np.random.randint(-random_max,random_max,pos.shape)

            valid = np.logical_and(np.abs(pos[:,0]) < size/2 , np.abs(pos[:,1]) < size/2)
            nvalid = valid.sum()
            valid1 = np.tile(valid.reshape((n**2,1)),(1,2))
            posv = pos[valid1].reshape((nvalid,2))
            # print 'nvalid = %d' % nvalid
    #        f, ax = plt.subplots(1,1,figsize=(10,7))
    #        cax = ax.scatter(posv[:,0],posv[:,1])
    #        plt.show()
            pos1 = posv[posv.shape[0]/2]
            pos2 = posv[(posv.shape[0]/2)+1]
            pos = np.array([pos1,pos2]).squeeze()
            pos -= np.min(pos,axis=0)

            if pos.max() > sh[0]:
                pos1 = posv[(posv.shape[0]/2)+1]
                pos2 = posv[(posv.shape[0]/2)+2]
                pos = np.array([pos1,pos2]).squeeze()
                pos -= np.min(pos,axis=0)

    #        print pos
    #        print pos[0,0],pos[0,0]+sh[0],pos[0,1],pos[0,1]+sh[1]

            ov[pos[0,0]:pos[0,0]+sh[0],pos[0,1]:pos[0,1]+sh[1]] += probe_mask
            ov[pos[1,0]:pos[1,0]+sh[0],pos[1,1]:pos[1,1]+sh[1]] *= probe_mask

            # f, ax = plt.subplots(1,1,figsize=(10,7))
            # ax.imshow(ov)
            # plt.show()

            ovlap =  ov.sum()/float(probe_mask.sum())
            # print ovlap#, posv.sha1pe[0]

    #        print
            if np.abs(ovlap - overlap) < 0.02:
                break
        #    phi += 2
    #        print 'phi = %d, s = %d' % (phi,s2)
        #    if phi >= 90:
            s2 += 1
        #        phi %= 90
        return posv
    try:
        posv = r(size,overlap)
    except e:
        posv = r(size,overlap)

    posv -= np.min(posv,axis=0)
    # f, ax = plt.subplots(1,1,figsize=(10,7))
    # cax = ax.scatter(posv[:,0],posv[:,1])
    # plt.show()
#    f, ax = plt.subplots(1,1,figsize=(10,7))
#    ax.imshow(ov)
#    plt.show()
#    print posv
#    f,a = plt.subplots(figsize=(5,5))
#    a.scatter(posv[:,0],posv[:,1])
#    plt.show()
    return posv



#N = 256
#raster_positions_overlap(384,sector_mask((N,N),(N/2,N/2),0.2*N,(0,360)), 0.50,pos_shift=[2,5],phi=44,random_max=4)
# MTF_DQE_2D2('K2',1,N,'/home/philipp/projects/dptycho/simulation/')
