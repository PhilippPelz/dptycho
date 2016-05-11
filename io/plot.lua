local classic = require 'classic'
local py = require('fb.python')
local argcheck = require('argcheck')
local dataloader = require 'io.dataloader'

local PYTHON_PLOTFILE = "/home/philipp/projects/dptycho/io/plot.py"
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
            {name="save_file_path", default='~/', type='string'},
            call =
                function (self,data,save_file_path)
                  -- py.exec(self.plotcode)
                  py.exec(
[=[
#test_cx_plot(data[0])
p = ReconPlot(data,save_file_path)
p.start_plotting(data)
]=]
                  ,{data = data,save_file_path=save_file_path})
                end
            }

c.update_reconstruction_plot = argcheck{
            nonamed=true,
            name = "update_reconstruction_plot",
            {name="self", type='table'},
            {name="data", type='table'},
            call =
                function (self,data)
                  print('update')
                  py.exec('p.update(data)',{data = data})
                end
            }

c.shutdown_reconstruction_plot = argcheck{
            nonamed=true,
            name = "shutdown_reconstruction_plot",
            {name="self", type='table'},
            call =
                function (self)
                  py.exec('p.stop()')
                end
            }

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
            {name="show", default=false, type='bool'},
            call =
                function (self,img, suptitle, savepath, cmap, title, show)
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = {img:abs(),img:arg()}, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
                end
            }

c.plotReIm = argcheck{
            nonamed=true,
            name = "plot",
            {name="self", type='table'},
            {name="img", type='torch.ZFloatTensor'},
            {name="suptitle", default='Image', type='string'},
            {name="savepath", default=py.None, type='string'},
            {name="cmap", default={'hot','hot'}, type='table'},
            {name="title", default={'Re','Im'}, type='table'},
            {name="show", default=true, type='bool'},
            call =
                function (self,img, suptitle, savepath, cmap, title, show)
                  py.eval('zplot(img,suptitle,savepath,cmap,title,show)',{img = {img:re(),img:im()}, suptitle=suptitle, savepath=savepath, cmap=cmap,title=title, show=show})
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


return c
