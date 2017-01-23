local classic = require 'classic'
local py = require('fb.python')
local argcheck = require('argcheck')
local dataloader = require 'dptycho.io.dataloader'

local PYTHON_PLOTFILE = "/home/philipp/philipp/projects/dptycho/io/plot.py"
--local PYTHON_PLOTFILE = "./plot.py"


local c = classic.class(...)

function c:_init()
  local d = dataloader()
  self.plotcode = d:loadText(PYTHON_PLOTFILE)
  py.exec(self.plotcode)
  return self
end

c.init_reconstruction_plot = argcheck{
            nonamed=true,
            name = "init_reconstruction_plot",
            {name="self", type='table'},
            {name="data", type='table'},
            {name="super_title", default = 'Reconstruction', type='string'},
            {name="error_labels", default={'$\frac{||[I-P_F]z||}{||a||}$', '$\frac{||[I-P_Q]z||}{||a||}$'}, type='table'},
            call =
                function (self,data,super_title,error_labels)
                  -- py.exec(self.plotcode)
                  py.exec(
[=[
#test_cx_plot(data[0])
global p
p = ReconPlot(data, interactive = False, suptitle=sup_title, interp='nearest', error_labels=error_labels)
]=]
                  ,{data=data,sup_title=super_title ,error_labels=error_labels})
                end
            }

c.update_reconstruction_plot = argcheck{
            nonamed=true,
            name = "update_reconstruction_plot",
            {name="self", type='table'},
            {name="data", type='table'},
            {name="save_file_path", default='~/', type='string'},
            call =
                function (self, data, save_file_path)
                  py.eval('p.update(data)',{data = data})
                  py.eval('p.draw(save_file_path)',{save_file_path=save_file_path})
                end
            }

c.shutdown_reconstruction_plot = argcheck{
            nonamed=true,
            name = "shutdown_reconstruction_plot",
            {name="self", type='table'},
            call =
                function (self)
                  -- py.exec('p.stop()')
                end
            }

local plot = argcheck{
            nonamed=true,
            name = "plot",
            {name="self", type='table'},
            {name="img", type='torch.DoubleTensor'},
            {name="title", default='Image', type='string'},
            {name="savepath", default=py.None,type='string'},
            {name="show", default=true, type='boolean'},
            {name="cmap", default='viridis',type='string'},
            call =
                function (self, img, title, savepath, show, cmap)
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
            {name="show", default=true, type='boolean'},
            {name="cmap", default='viridis',type='string'},
            call =
                function (self, img, title, savepath, show, cmap)
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
            {name="show", default=true, type='boolean'},
            {name="cmap", default={'Greys','hsv'}, type='table'},
            {name="title", default={'Abs','Phase'}, type='table'},
            call =
                function (self,img, suptitle, savepath, show, cmap, title)
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = {img:abs(),img:arg()}, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
                end
            }
plot = argcheck{
            nonamed=true,
            name = "plot",
            overload = plot,
            {name="self", type='table'},
            {name="img", type='torch.ZCudaTensor'},
            {name="suptitle", default='Image', type='string'},
            {name="savepath", default=py.None, type='string'},
            {name="show", default=true, type='boolean'},
            {name="cmap", default={'Greys','hsv'}, type='table'},
            {name="title", default={'Abs','Phase'}, type='table'},
            call =
                function (self,img, suptitle, savepath, show, cmap, title)
                  local img1 = img:zfloat()
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = {img1:abs(),img1:arg()}, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
                end
            }

c.plotcx = argcheck{
            nonamed=true,
            name = "plot",
            {name="self", type='table'},
            {name="img", type='torch.ZCudaTensor'},
            call =
                function (self,img)
                  local img1 = img:zfloat()
                  py.eval('plotcx(img)',{img = {img1:re(),img1:im()}})
                end
            }

c.plotReIm = argcheck{
            nonamed=true,
            name = "plot",
            {name="self", type='table'},
            {name="img", type='torch.ZFloatTensor'},
            {name="suptitle", default='Image', type='string'},
            {name="savepath", default=py.None, type='string'},
            {name="show", default=true, type='boolean'},
            {name="cmap", default={'viridis','viridis'}, type='table'},
            {name="title", default={'Re','Im'}, type='table'},
            call =
                function (self,img, suptitle, savepath, show, cmap, title)
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = {img:re(),img:im()}, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
                end
            }

c.plotcompare = argcheck{
            nonamed=true,
            name = "plot",
            overload = plot,
            {name="self", type='table'},
            {name="imgs", type='table'},
            {name="title", default={'Img1','Img2'}, type='table'},
            {name="suptitle", default='comparison', type='string'},
            {name="savepath", default=py.None, type='string'},
            {name="show", default=true, type='boolean'},
            {name="cmap", default={'viridis','viridis'}, type='table'},
            call =
                function (self,imgs, title, suptitle, savepath, show, cmap)
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

c.hist = argcheck{
            nonamed=true,
            name = "hist",
            {name="self", type='table'},
            {name="x", type='torch.FloatTensor'},
            {name="bins", default='100', type='number'},
            {name="title", default='Histogram', type='string'},
            call =
                function (self, x, bins,title)
                  py.eval('hist(x,bins,title)',{x = x, bins=bins,title=title})
                end
            }
c.plot = plot

c.scatter_positions = argcheck{
            nonamed=true,
            name = "scatter_positions",
            {name="self", type='table'},
            {name="pos1", type='torch.FloatTensor'},
            {name="pos2", type='torch.FloatTensor'},
            call =
                function (self, pos1, pos2)
                  py.eval('scatter_positions(pos1,pos2)',{pos1 = pos1, pos2=pos2})
                end
            }
c.scatter_positions2 = argcheck{
            nonamed=true,
            name = "scatter_positions2",
            {name="self", type='table'},
            {name="pos1", type='torch.FloatTensor'},
            call =
                function (self, pos1)
                  py.eval('scatter_positions2(pos1)',{pos1 = pos1})
                end
            }

return c
