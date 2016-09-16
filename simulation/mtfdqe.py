# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:25:14 2016

@author: philipp pelz
"""

import pandas as pd
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from scipy import interpolate as i
from math import *
import inspect, os

def MTF_DQE_2D(cam,binning,s,path):
    if cam == 'K2':
        df = pd.read_csv(path + '/K2Count3eps2.csv')
        dqe = df['DQE'].values
        mtf = df['MTF'].values
        q = df['Frequency'].values

        size = np.array((3840,3712))
        bsize = size / binning
        dqef = i.interp1d(q,dqe,kind='quadratic')
        mtff = i.interp1d(q,mtf,kind='quadratic')

        xx,yy = np.mgrid[-s/2:s/2,-s/2:s/2]
        rr = np.sqrt(xx**2 + yy**2)
        qq = rr / np.max(size)

        mtf2D = fftshift(mtff(qq))
        dqe2D = fftshift(dqef(qq))

        # f, ax = plt.subplots(1,1,figsize=(10,7))
        # cax = ax.imshow(mtf2D)
        # plt.colorbar(cax)
        # plt.show()
        #
        # f, ax = plt.subplots(1,1,figsize=(10,7))
        # cax = ax.imshow(dqe2D)
        # plt.colorbar(cax)
        # plt.show()

        return mtf2D, dqe2D

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






#pos(71,384)
#MTF_DQE_2D('K2',2,1024)
