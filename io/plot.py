import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyE17.utils as u
plt.style.use('ggplot')

def plot(img, title='Image', savePath=None, cmap='hot', show=True):
    fig, ax = plt.subplots()
    cax = ax.imshow(img, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    if show:
        plt.show()
    if savePath is not None:
        plt.imsave(savePath + '.png', img)

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
        plt.imsave(savePath + '.png', img)

    plt.close()

import numpy as np
from mayavi import mlab
def plot3d(arr,title,vmin = 0,vmax = 0.7):
    mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
    src = mlab.pipeline.scalar_field(arr)
    mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)
    mlab.colorbar(title=title, orientation='vertical', nb_labels=7)
    mlab.show()

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ReconPlot():

    def __init__(self,imgs, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
        gs = gridspec.GridSpec(3, 3)
        obamp, obph, probe, err_mod, err_Q = imgs
        x = np.linspace(1, err_mod.size, err_mod.size)
        
    #     plt.ion()
        
        self.fig = plt.figure()
        self.ax_obamp = plt.subplot(gs[:2,0])
        self.ax_obph = plt.subplot(gs[:2,1])
        self.ax_probe = plt.subplot(gs[:2,2])
        self.ax_errors = plt.subplot(gs[2,:])
        
        div_obamp = make_axes_locatable(self.ax_obamp)
        div_obph = make_axes_locatable(self.ax_obph)
        div_errors = make_axes_locatable(self.ax_errors)
        div_probe = make_axes_locatable(self.ax_probe)
        
        self.fig.suptitle(suptitle, fontsize=20)
        self.imax1 = self.ax_obamp.imshow(obamp, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
        self.imax2 = self.ax_obph.imshow(obph, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
        self.iprobe = self.ax_probe.imshow(u.imsave(probe), interpolation='nearest')
        self.im_errors = self.ax_errors.scatter(x,err_mod,c='r', label='$||[I-P_F]z||$')
        self.im_errors2 = self.ax_errors.scatter(x,err_Q,c='b', label='$||[I-P_Q]z||$')
        
        legend = self.ax_errors.legend()
        
        cax1 = div_obamp.append_axes("right", size="10%", pad=0.05)
        cax2 = div_obph.append_axes("right", size="10%", pad=0.05)
        
        cbar1 = plt.colorbar(self.imax1, cax=cax1)
        cbar2 = plt.colorbar(self.imax2, cax=cax2)
        
        self.ax_obamp.set_title(title[0])
        self.ax_obph.set_title(title[1])
        plt.tight_layout()
        if show:
            plt.show()
    #         plt.pause(1)
        if savePath is not None:
            fig.savefig(savePath + '.png', dpi=600)
            
    def update(self,imgs):
        obamp, obph, probe, err_mod, err_Q = imgs
    
        
        
#     plt.close()

a = np.random.randn(50,50)
b = np.random.randn(50,50)
c = a + 1j*b
err1 = x = np.linspace(0, 2 * np.pi, 400)
err2 = x = np.linspace(0, 2 * np.pi, 400) * 2
# plot(a)
# zplot([a,b])
recon_plot([np.abs(c),np.angle(c),c,err1,err2])
