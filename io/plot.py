import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure as MPLFigure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NaviToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import time
import sys
import threading

import math
from PIL import Image
from mayavi import mlab
import gtk
import sys
import os
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

def load_image(filename):
    a = np.array(Image.open(filename))
    cx = rgb2complex(a)

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

def plot(img, title='Image', savePath=None, cmap='hot', show=True):
    fig, ax = plt.subplots()
    cax = ax.imshow(img, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)
    plt.close()

def zplot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
    im1, im2 = img
    fig, (ax1,ax2) = plt.subplots(1,2)
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
    plt.close()

def cx_test(cx_array):
    a = float2_to_complex(cx_array)
    plot(a.real)
    plot(a.imag)

def scatter_positions(pos1,pos2):
    fig, ax = plt.subplots()
    ax.scatter(pos1[0],pos1[1],c='r')
    ax.scatter(pos2[0],pos2[1],c='b')
    plt.show()


def plot3d(arr,title,vmin = 0,vmax = 0.7):
    mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
    src = mlab.pipeline.scalar_field(arr)
    mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)
    mlab.colorbar(title=title, orientation='vertical', nb_labels=7)
    mlab.show()

# http://matplotlib.org/users/navigation_toolbar.html
class ReconPlot():

    def __init__(self,data, interactive = True, suptitle='Image', interp='nearest', error_labels = [r'$\frac{||[I-P_F]z||}{||a||}$', r'$\frac{||[I-P_Q]z||}{||a||}$']):
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        self.interp = interp
        self.interactive = interactive
        self.suptitle = suptitle
        obre,obim, prre,prim, bg, pos, err = data

        self.error_labels = error_labels

        obre = obre[:,0,:,:]
        obim = obim[:,0,:,:]
        prre = prre[0,...]
        prim = prim[0,...]

        x = np.linspace(1, err[0].size, err[0].size)

        No = obre.shape[0]
        Np = prre.shape[0]

        self.fig = plt.figure()
        self.fig.suptitle(self.suptitle)

        probe_spaces_right_of_ob = 2 * No
        # print probe_spaces_right_of_ob
        extra_rows_for_pr = max(math.ceil((Np - probe_spaces_right_of_ob) / 4 ),0)
        # print extra_rows_for_pr
        nrows = int(No + extra_rows_for_pr + 1)
        # print nrows
        gs = gridspec.GridSpec(nrows, 4)

        self.ob_axes = []
        self.pr_axes = []
        self.ob_caxes = []
        self.pr_caxes = []
        self.pos_axes = self.fig.add_subplot(gs[0,2])
        self.pos_axes.set_title('Positions')
        self.bg_axes = self.fig.add_subplot(gs[0,3])
        self.bg_axes.set_title('Background')
        self.bg_axes.tick_params(labelbottom='off',labelleft='off',labeltop='off',left='off',bottom='off',top='off',right='off')


        # probes next to object ======================================================
        ob_rows = 0
        n_pr = 0
        for i,x in enumerate(obre):
            ob_rows = i
            # print [i,0]
            # print [i,1]
            obamp = self.fig.add_subplot(gs[i,0])
            obph = self.fig.add_subplot(gs[i,1])
            self.set_tick_params(obamp)
            self.set_tick_params(obph)
            obamp.set_title('$|O_{%d}|$'%i)
            obph.set_title('$Arg(O_{%d})$'%i)
            div_obamp = make_axes_locatable(obamp)
            div_obph = make_axes_locatable(obph)
            cax1 = div_obamp.append_axes("right", size="10%", pad=0.05)
            cax2 = div_obph.append_axes("right", size="10%", pad=0.05)
            if i > 0:
                # print [i,2]
                # print [i,3]
                pr1 = self.fig.add_subplot(gs[i,2])
                pr2 = self.fig.add_subplot(gs[i,3])
                self.set_tick_params(pr1)
                self.set_tick_params(pr2)
                pr1.set_title('$\Psi_{%d}$'%n_pr)
                n_pr +=1
                pr2.set_title('$\Psi_{%d}$'%n_pr)
                n_pr +=1
                self.pr_axes.append(pr1)
                self.pr_axes.append(pr2)
                div_probe1 = make_axes_locatable(pr1)
                div_probe2 = make_axes_locatable(pr2)
                self.pr_caxes.append(div_probe1)
                self.pr_caxes.append(div_probe2)
            self.ob_axes.append([obamp,obph])
            self.ob_caxes.append([cax1,cax2])


        # probes next to errors ======================================================
        ob_rows += 1
        # print [ob_rows,2]
        # print [ob_rows,3]
        self.err_axes = self.fig.add_subplot(gs[ob_rows,:2])
        self.err_axes.set_yscale("log", nonposy='clip')

        pr1 = self.fig.add_subplot(gs[ob_rows,2])
        pr2 = self.fig.add_subplot(gs[ob_rows,3])
        pr1.set_title('$\Psi_{%d}$'%n_pr)
        n_pr +=1
        pr2.set_title('$\Psi_{%d}$'%n_pr)
        n_pr +=1

        self.set_tick_params(pr1)
        self.set_tick_params(pr2)
        self.pr_axes.append(pr1)
        self.pr_axes.append(pr2)


        #probes below everything ======================================================
        col = 0
        for j,pr in enumerate(prre):
            if j >= len(self.pr_axes):
                if col == 0: ob_rows += 1
                # print [ob_rows,col]
                pr1 = self.fig.add_subplot(gs[ob_rows,col])
                pr1.tick_params(labelbottom='off',labelleft='off',labeltop='off',left='off',bottom='off',top='off',right='off')
                pr1.set_title('$\Psi_{%d}$'%n_pr)
                n_pr +=1
                col = (col + 1) % 4
                self.pr_axes.append(pr1)

        self.ob_imaxes = None
        self.pr_imaxes = None
        self.im_errors = None
        self.im_errors2 = None
        self.im_pos = None
        self.has_data = False
        self.legend = None
        # print 'wp4'

    def set_tick_params(self,ax):
        ax.tick_params(labelbottom='off',labelleft='off',labeltop='off',left='off',bottom='off',top='off',right='off')

    def update(self,data, cmap=['hot','hsv']):
        obamp,obph, prre,prim, bg, pos, errors = data

        obamp = obamp[:,0,:,:]
        obph = obph[:,0,:,:]
        prre = prre[0,...]
        prim = prim[0,...]
        pr = prre + 1j*prim

        ints = np.sum(np.real(pr*pr.conj()),(1,2))
        ints_percent = ints/ints.sum()

        #print pr.min(), pr.max()
        if self.ob_imaxes is None:
            self.ob_imaxes = []
            self.ob_cbars = []
            for i,(oa,op) in enumerate(zip(obamp,obph)):
                # print oa.shape
                # print op.shape
                imax1 = self.ob_axes[i][0].imshow(oa, interpolation=self.interp, cmap=plt.cm.get_cmap(cmap[0]))
                imax2 = self.ob_axes[i][1].imshow(op, interpolation=self.interp, cmap=plt.cm.get_cmap(cmap[1]))
                cbar1 = plt.colorbar(imax1, cax=self.ob_caxes[i][0])
                cbar2 = plt.colorbar(imax2, cax=self.ob_caxes[i][1])
                self.ob_cbars.append([cbar1,cbar2])
                self.ob_imaxes.append([imax1,imax2])
        else:
            for i,(oa,op) in enumerate(zip(obamp,obph)):
                obamp, obph = self.ob_imaxes[i]
                obamp.set_data(oa)
                obph.set_data(op)

        if self.pr_imaxes is None:
            self.pr_imaxes = []
            for i,p in enumerate(pr):
                iprobe = self.pr_axes[i].imshow(self.imsave(p), interpolation=self.interp)
                self.pr_axes[i].set_title(r'$\Psi_{%d} - %-5.2g$ percent'%(i,ints_percent[i]*100))
                self.pr_imaxes.append(iprobe)
        else:
            for i,p in enumerate(pr):
                pra = self.pr_imaxes[i]
                pra.set_data(self.imsave(p))
                self.pr_axes[i].set_title(r'$\Psi_{%d}$ - %-5.2g$ percent'%(i,ints_percent[i]*100))

        if self.im_errors is not None:
            del self.im_errors[:]
        if self.im_errors2 is not None:
            del self.im_errors2[:]

        x = np.linspace(1, errors[0].size, errors[0].size)
        for i,err in enumerate(errors):
            self.im_errors = self.err_axes.plot(x,err, label=self.error_labels[i])

        if self.im_pos is not None:
            self.im_pos.remove()

        self.bg_pos = self.pos_axes.imshow(op, interpolation=self.interp)
        self.im_pos = self.pos_axes.scatter(pos[:,0],pos[:,1],c='r',s=5,marker='+')

        if self.legend is None:
            self.legend = self.err_axes.legend()

        self.has_data = True

    def stop():
        self.stop = True

    def start_plotting(self,data):
        self.stop = False

        while True:
#            No = 2
#            Np = 8
#            M = 200
#            Nx = 400
#            Ny = 400
#            K = 150
#            ob = np.random.randn(No,1,Nx,Ny,2).astype(np.float32)
#            pr = np.random.randn(1,Np,M,M,2).astype(np.float32)
#            bg = np.random.randn(M,M).astype(np.float32)
#            bg = 1
#            pos = np.random.randn(2,K).astype(np.float32)
#            err_mod = x = np.linspace(0, 2 * np.pi, 400).astype(np.float32)
#            err_Q = x = np.linspace(0, 2 * np.pi, 400) * 2
#            data = [ob, pr, bg, pos, err_mod, err_Q ]
#            self.update(data)
            if self.has_data:
                self.draw()
            elif self.stop:
                break
            pause(.1)

    def complex2hsv(self,cin, vmin=None, vmax=None):
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

    def complex2rgb(self,cin, **kwargs):
        """
        Executes `complex2hsv` and then `hsv2rgb`

        See also
        --------
        complex2hsv
        hsv2rgb
        rgb2complex
        """
        return self.hsv2rgb(self.complex2hsv(cin,**kwargs))

    def hsv2rgb(self,hsv):
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


    def imsave(self,a, filename=None, vmin=None, vmax=None, cmap=None):
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
            i = self.complex2rgb(a, vmin=vmin, vmax=vmax)
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
    def create_new_window(self):
        win = gtk.Window()
        win.connect("destroy", lambda x: gtk.main_quit())
        win.set_default_size(1280,1024)
        win.set_title(self.suptitle)
        vbox = gtk.VBox()
        win.add(vbox)

        canvas = FigureCanvas(self.fig)  # a gtk.DrawingArea
        toolbar = NaviToolbar(canvas, win)
        vbox.pack_start(toolbar, False, False)
        vbox.pack_start(canvas)
        return win

    def draw(self,savePath):
        if self.interactive:
            plt.draw()
            time.sleep(0.1)
        else:
#         print 'show'
            win = self.create_new_window()
            win.show_all()
            if savePath is not None:
                self.fig.savefig(savePath + '.png', dpi=600)
            gtk.main()
#             self.fig.show()
        self.has_data = False

def run():
    No = 2
    Np = 8
    M = 200
    Nx = 400
    Ny = 400
    K = 150
    ob = np.random.randn(No,1,Nx,Ny,2).astype(np.float32)
    pr = np.random.randn(1,Np,M,M,2).astype(np.float32)
    bg = np.random.randn(M,M).astype(np.float32)
    bg = 1
    pos = np.random.randn(2,K).astype(np.float32)
    err_mod = x = np.linspace(0, 2 * np.pi, 400).astype(np.float32)
    err_Q = x = np.linspace(0, 2 * np.pi, 400) * 2
    data = [ob, pr, bg, pos, err_mod, err_Q ]
    plt.ion()
    p = ReconPlot(data,False)
    p.start_plotting(data)


# a = np.random.randn(50,50)
# b = np.random.randn(50,50)
# pos1=np.random.randn(2,100)
# pos2=np.random.randn(2,100)
# c = a + 1j*b
# err1 = x = np.linspace(0, 2 * np.pi, 400)
# err2 = x = np.linspace(0, 2 * np.pi, 400) * 2
# # plot(a)
# # zplot([a,b])
# # recon_plot([np.abs(c),np.angle(c),c,err1,err2])
# scatter_positions(pos1, pos2)
