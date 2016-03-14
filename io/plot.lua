local classic = require 'classic'
local py = require('fb.python')
local argcheck = require('argcheck')
local dataloader = require 'dptycho.io.dataloader'

local PYTHON_PLOTFILE = "/home/philipp/projects/dptycho-devel/plot.py"
--local PYTHON_PLOTFILE = "./plot.py"


local c = classic.class(...)

function c:_init()
  local d = dataloader()
  local plotcode = d:loadText(PYTHON_PLOTFILE)
--  print(plotcode)
--  py.exec(plotcode)
  py.exec([=[import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from mayavi import mlab
def plot3d(arr,title,vmin = 0,vmax = 0.7):
    mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0, 0, 0))
    src = mlab.pipeline.scalar_field(arr)
    mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)
    mlab.colorbar(title=title, orientation='vertical', nb_labels=7)
    mlab.show()]=])

--  print('loaded')
  return self
end

local plot = argcheck{
            nonamed=true,
            name = "plot",
            {name="self", type='table'},
            {name="img", type='torch.DoubleTensor'},
            {name="title", default='Image', type='string'},
            {name="savepath", default=py.None,type='string'},
            {name="cmap", default='hot',type='string'},
            {name="show", default=true, type='bool'},
            call =
                function (self,img, title, savepath, cmap, show)
                  py.eval('plot(img,title,savepath,cmap,show)',{img = img, title=title, savepath=savepath, cmap=cmap, show=show})
                end
            }
plot = argcheck{
            nonamed=true,
            name = "plot",
            overload = plot,
            {name="self", type='table'},
            {name="img", type='torch.FloatTensor'},
            {name="title", default='Image', type='string'},
            {name="savepath", default=py.None,type='string'},
            {name="cmap", default='hot',type='string'},
            {name="show", default=true, type='bool'},
            call =
                function (self,img, title, savepath, cmap, show)
--                  print(torch.type(img))
                  py.eval('plot(img,title,savepath,cmap,show)',{img = img, title=title, savepath=savepath, cmap=cmap, show=show})
                end
            }
plot = argcheck{
            nonamed=true,
            name = "plot",
            overload = plot,
            {name="self", type='table'},
            {name="img", type='torch.ZFloatTensor'},
            {name="suptitle", default='Image', type='string'},
            {name="savepath", default=py.None, type='string'},
            {name="cmap", default={'hot','hsv'}, type='table'},
            {name="title", default={'Abs','Phase'}, type='table'},
            {name="show", default=true, type='bool'},
            call =
                function (self,img, suptitle, savepath, cmap, title, show)
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = {img:abs(),img:arg()}, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
                end
            }
c.plotcompare = argcheck{
            nonamed=true,
            name = "plot",
            overload = plot,
            {name="self", type='table'},
            {name="imgs", type='table'},
            {name="suptitle", default='Image', type='string'},
            {name="savepath", default=py.None, type='string'},
            {name="cmap", default={'hot','hot'}, type='table'},
            {name="title", default={'Img1','Img2'}, type='table'},
            {name="show", default=true, type='bool'},
            call =
                function (self,imgs, suptitle, savepath, cmap, title, show)
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = imgs, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
                end
            }
c.plot3d = argcheck{
            nonamed=true,
            name = "plot3d",
            {name="self", type='table'},
            {name="arr", type='torch.FloatTensor'},
            {name="title", default='Image', type='string'},
            {name="vmin", default=0, type='number'},
            {name="vmax", default=0.7, type='number'},
            call =
                function (self, arr, title, vmin, vmax)
                  py.eval('plot3d(arr,title,vmin,vmax)',{arr = arr, title=title, vmin=vmin, vmax=vmax})
                end
            }
c.plot = plot
return c
