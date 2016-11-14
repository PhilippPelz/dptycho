# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:53:45 2016

@author: philipp
"""
from PIL import Image
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
    cout = hsv2complex(rgb2hsv(rgb))
    return (cout.real,cout.imag)

def complex2hsv(cin, vmin=None, vmax=None):
    """\
    Transforms a complex array into an RGB image,
    mapping phase to hue, amplitude to value and
    keeping maximum saturation.

    Parameters
    ----------
    cin : ndarray
        Complex input. Must be two-dimensional.

    vmin,vmax : float
        Clip amplitude of input into this interval.

    Returns
    -------
    rgb : ndarray
        Three dimensional output.

    See also
    --------
    complex2rgb
    hsv2rgb
    hsv2complex
    """
    # HSV channels
    h = .5*np.angle(cin)/np.pi + .5
    s = np.ones(cin.shape)

    v = abs(cin)
    if vmin is None: vmin = 0.
    if vmax is None: vmax = v.max()
    #print vmin, vmax
    assert vmin < vmax
    v = (v.clip(vmin,vmax)-vmin)/(vmax-vmin)

    return np.asarray((h,s,v))

def complex2rgb(r,i, **kwargs):
    """
    Executes `complex2hsv` and then `hsv2rgb`

    See also
    --------
    complex2hsv
    hsv2rgb
    rgb2complex
    """
    cin = r + 1j * i
    return hsv2rgb(complex2hsv(cin,**kwargs))

def hsv2rgb(hsv):
    """\
    HSV (Hue,Saturation,Value) to RGB (Red,Green,Blue) transformation.

    Parameters
    ----------
    hsv : array-like
        Input must be two-dimensional. **First** axis is interpreted
        as hue,saturation,value channels.

    Returns
    -------
    rgb : ndarray
        Three dimensional output. **Last** axis is interpreted as
        red, green, blue channels.

    See also
    --------
    complex2rgb
    complex2hsv
    rgb2hsv
    """
    # HSV channels
    h,s,v = hsv

    i = (6.*h).astype(int)
    f = (6.*h) - i
    p = v*(1. - s)
    q = v*(1. - s*f)
    t = v*(1. - s*(1.-f))
    i0 = (i%6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    rgb = np.zeros(h.shape + (3,), dtype=h.dtype)
    rgb[:,:,0] = 255*(i0*v + i1*q + i2*p + i3*p + i4*t + i5*v)
    rgb[:,:,1] = 255*(i0*t + i1*v + i2*v + i3*q + i4*p + i5*p)
    rgb[:,:,2] = 255*(i0*p + i1*p + i2*t + i3*v + i4*v + i5*q)

    return rgb
    
def imsave(a, filename=None, vmin=None, vmax=None, cmap=None):
    """
    Take array `a` and transform to `PIL.Image` object that may be used
    by `pyplot.imshow` for example. Also save image buffer directly
    without the sometimes unnecessary Gui-frame and overhead.

    Parameters
    ----------
    a : ndarray
        Two dimensional array. Can be complex, in which case the amplitude
        will be optionally clipped by `vmin` and `vmax` if set.

    filename : str, optionsl
        File path to save the image buffer to. Use '\*.png' or '\*.png'
        as image formats.

    vmin,vmax : float, optional
        Value limits ('clipping') to fit the color scale.
        If not set, color scale will span from minimum to maximum value
        in array

    cmap : str, optional
        Name of the colormap for colorencoding.

    Returns
    -------
    im : PIL.Image
        a `PIL.Image` object.

    See also
    --------
    complex2rgb

    Examples
    --------
    >>> from ptypy.utils import imsave
    >>> from matplotlib import pyplot as plt
    >>> from ptypy.resources import flower_obj
    >>> a = flower_obj(512)
    >>> pil = imsave(a)
    >>> plt.imshow(pil)
    >>> plt.show()

    converts array a into, and returns a PIL image and displays it.

    >>> pil = imsave(a, /tmp/moon.png)

    returns the image and also saves it to filename

    >>> imsave(a, vmin=0, vmax=0.5)

    clips the array to values between 0 and 0.5.

    >>> imsave(abs(a), cmap='gray')

    uses a matplotlib colormap with name 'gray'
    """
    if str(cmap) == cmap:
        cmap= mpl.cm.get_cmap(cmap)

    if a.dtype.kind == 'c':
        # Image is complex
        #if cmap is not None:
            #logger.debug('imsave: Ignoring provided cmap - input array is complex')
        i = complex2rgb(a.real,a.imag, vmin=vmin, vmax=vmax)
        im = Image.fromarray(np.uint8(i), mode='RGB')

    else:
        if vmin is None:
            vmin = a.min()
        if vmax is None:
            vmax = a.max()
        im = Image.fromarray((255*(a.clip(vmin,vmax)-vmin)/(vmax-vmin)).astype('uint8'))
        if cmap is not None:
            r = im.point(lambda x: cmap(x/255.0)[0] * 255)
            g = im.point(lambda x: cmap(x/255.0)[1] * 255)
            b = im.point(lambda x: cmap(x/255.0)[2] * 255)
            im = Image.merge("RGB", (r, g, b))
        #b = (255*(a.clip(vmin,vmax)-vmin)/(vmax-vmin)).astype('uint8')
        #im = Image.fromstring('L', a.shape[-1::-1], b.tostring())

    if filename is not None:
        im.save(filename)
    return im    
    
    
import numpy as np
import matplotlib.pyplot as plt
abso = np.random.randn(100,100) * 50
cx = abso * np.exp(1j*np.random.randn(100,100)*np.pi)
absmax = np.abs(cx).max()
print np.abs(cx).max(), np.angle(cx).max()
rgb = complex2rgb(cx.real,cx.imag)

cx2real,cx2imag = rgb2complex(rgb)
cx2 = cx2real + 1j* cx2imag
cx2*= absmax
print np.abs(cx2).max(), np.angle(cx2).max()

f,a = plt.subplots()
a.imshow(imsave(cx))
plt.show()




f,a = plt.subplots()
a.imshow(imsave(cx2))
plt.show()
