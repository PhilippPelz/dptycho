# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:05:02 2016

@author: philipp
"""

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
def riplot(img, suptitle='Image', savePath=None, cmap=['hot','hot'], title=['Abs','Phase'], show=True):
    im1, im2 = np.real(img), np.imag(img)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 10))
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
def applot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
    im1, im2 = np.abs(img), np.angle(img)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 10))
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
def aplplot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
    im1, im2 = np.log10(np.abs(img)), np.angle(img)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 10))
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
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def show(x):
    plt.figure(figsize=(10,10))
    plt.imshow(x)
    plt.show()
    
cam = nd.zoom(misc.imread('/home/philipp/dl/cameraman.bmp', flatten= 1),1/4.0)
arg1 = cam/cam.max()
#print cam.shape
show(arg1)

arg = nd.zoom(scipy.misc.ascent().astype(np.float),1/8.0)
arg /= arg.max()
#print arg.shape
abso = nd.zoom(rgb2gray(scipy.misc.face()),(2/24.0,1/16.0))
abso/=abso.max()
psi = (abso * np.exp(2j*np.pi*arg1)).astype(np.complex64)
os = psi.shape
n2 = np.prod(os)
psiv = psi.reshape((n2,1))
applot(psi,'object')

w = (h5read("~/ball.h5")['probe']).astype(np.complex64)
#applot(w,'probe')
wv = sp.diags(w.ravel(),0)
ps = w.shape
m2 = np.prod(ps)
m = ps[0]

W = fft(np.eye(m))/ sqrt(m)
i, j = np.ogrid[0:m,0:m]
omega = np.exp( - 2 * pi * 1J / m )
W2 = np.power( omega, i * j ) / sqrt(m)
Fi = np.kron(W,W) 
Fistar = Fi.conj().transpose()
print Fi.shape


def Ti(x):
    ind = np.arange(m2).reshape(ps)
    psi0 = np.zeros(os)
    psi0[x[0]:x[0]+ps[0],x[1]:x[1]+ps[1]] = ind
#    show(psi0)
    psi0_bc = np.broadcast_to(psi0.reshape((1,n2)),(m2,n2))
    wind = np.broadcast_to(np.arange(m2).reshape((m2,1)),(m2,n2))
    Ti = psi0_bc == wind
#    show(Ti)
    return sp.csr_matrix(Ti)


pos = np.arange(0,os[0]-ps[0]+1,ps[0]/4)
posx = np.tile(pos,pos.size)
posy = np.repeat(pos,pos.size)
pos = np.zeros((pos.size**2,2))
pos[:,0] = posx
pos[:,1] = posy
K = pos.shape[0]

F = sp.block_diag([Fi for i in range(K)])
Fstar = F.conj().transpose()
print F.shape
#print pos
#print R.shape
#show(R)

Qi = []
for i, posi in enumerate(pos):
    t = Ti(posi)
#    print (m2,n2), t.shape
    Qi.append(wv.dot(t))
    print i
#print Qi    
Q = sp.vstack(Qi)
Qstar = Q.conj().transpose()
d = np.array((Qstar.dot(Q)).diagonal())
d += 1e-10
#riplot(sp.diags(d,0).toarray()[0:2**11,0:2**11])
print d
QstarQinv = sp.diags(1/d,0)
#riplot(QstarQinv.toarray()[0:2**11,0:2**11])
print (K*m2,n2), Q.shape 

def P_Q(x): return Q.dot(QstarQinv.dot(Qstar.dot(x)))
def P_FQ(y): return F.dot(P_Q(Fstar.dot(y)))

z = Q.dot(psiv)
z = z.reshape((K,ps[0],ps[1]))
Fz = fft2(z,norm="ortho")

#for i in range(K):
#    applot(Fz[i])
    
Fz2 = Fi.dot(z[0:m2])
iz = Fistar.dot(Fz2)
applot(iz.reshape(ps))
#applot(Fz[0])
#print np.allclose(Fz2.reshape(ps),Fz[0])

a = np.abs(Fz)
av = a.reshape((K*m2,1))

percentile = np.percentile(a,98)
Ta0 = a > percentile

#riplot(Ta0[0])
avgtperc = (a.ravel() > percentile).astype(np.int16)
Ta = sp.diags(avgtperc,0)
print Ta.shape


    
F = F.tocsr()
FTa = P_Q(Fstar.dot(Ta))
FTal = FTa.tolil()
FFTa = F.dot(FTa)
FFTal = F.tolil()
TaFFTa = Ta.dot(FFTa)
TaFFTal = TaFFTa.tolil()
Tal = Ta.tolil()
#M = Ta.dot(FTa)
#Ml = M.tolil()
#print M.shape
#riplot(Ml[0:2**11,0:2**11].toarray())
riplot(TaFFTal[0:2**12,0:2**12].toarray())

ev, evec = sp.linalg.eigsh(TaFFTa,k=1)
print ev
#
print (evec>1e-6).sum()
#
psi1 = QstarQinv.dot(Qstar.dot(evec))
applot(psi1.reshape(os))

def At(z): return QstarQinv.dot(Qstar.dot(Fstar.dot(z)))
def A(psi): return F.dot(Q.dot(psi))


psi0 = np.random.randn(n2)
psi0 /= np.linalg.norm(psi0)
normest = sqrt(a.sum()/a.size)

Ta2 = a < 9*normest**2
atr = Ta0 * a
riplot(atr[0])
for i in range(50):
    z0 = atr.ravel() * A(psi0)
    z0r = z0.reshape((K,ps[0],ps[1]))  
#    applot(z0r[0])
#    applot(z0r[1])
    psi0 = At(z0)
#    applot(psi1.reshape(os))

psi2 = normest * psi0

applot(psi,'object')
applot(psi2.reshape(os))

