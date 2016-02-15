local classic = require 'classic'
local py = require('fb.python')
local argcheck = require('argcheck')
local dataloader = require 'io.dataloader'

local PYTHON_PLOTFILE = "/home/philipp/projects/dptycho-devel/plot.py"
--local PYTHON_PLOTFILE = "./plot.py"

local c = classic.class(...)

function c:_init()
  local d = dataloader()
  local plotcode = d:loadText(PYTHON_PLOTFILE)
--  print(plotcode)
  py.exec(plotcode)
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
c.plot = plot            
return c
