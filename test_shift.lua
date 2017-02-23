require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local znn = require "dptycho.znn"
local plot = require 'dptycho.io.plot'
local plt = plot()

local path = '/home/philipp/dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/'
local file = 'scan289.h5'

local f = hdf5.open(path..file,'r')
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new({1,256,256})
probe[1]:copyIm(pi):copyRe(pr)
probe[1]:div(probe:abs():max())
f:close()

-- local t = torch.ZCudaTensor.new(1,50,50):fill(0)
local t = probe
local test = probe:clone()

-- sl = t[{1,{4,10},{4,10}}]

-- t:fillRe(2)
-- sl:fill(1)
-- sl = t[{2,{30,40},{30,40}}]
-- sl:fillRe(1)
-- sl:fillIm(2)

local dest = torch.CudaTensor.new(1,256,256):fill(0)
local fw = torch.CudaTensor.new(1,50,50):fill(0)
local bw = torch.CudaTensor.new(1,50,50):fill(0)
local t1 = torch.CudaTensor.new(1,50,50):fill(0)
local t2 = torch.CudaTensor.new(1,50,50):fill(0)
local dest = torch.ZCudaTensor.new(1,256,256):fillRe(0):fillIm(0)
-- local fw = torch.CudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- local bw = torch.CudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- plt:plotcompare({t[1]:re():float(),dest[1]:re():float()})
-- plt:plotcompare({t[2]:re():float(),dest[2]:re():float()})
-- plt:plot(t[1]:re():float())
for a = -10,10 do
  s = {a/10.0,a/10.0}
  pprint(s)
  dest:shift(t,torch.FloatTensor(s))
  -- plt:plotReIm(probe:clone():add(-1,dest)[1]:zfloat(),'diff')
  -- plt:plot(probe:clone():add(-1,dest)[1]:zfloat(),'diff')
  plt:plot(dest[1]:zfloat(),'abs ph')
end
-- fw:shift(t,torch.FloatTensor({-1,0}))
-- plt:plot(fw[1]:float())
-- bw:shift(t,torch.FloatTensor({1,0})):mul(-1)
-- plt:plot(bw[1]:float())
-- dest:dx(t,fw,bw)
-- plt:plotReIm(dest[1]:zfloat())

-- a = torch.ZFloatTensor(1)
-- a[1] = 1+2i
-- a = a:zcuda()
-- r = a:dot(a)
-- print(r)

--   a = torch.FloatTensor({{1, 0,  0,  0, 0},
--                   {0, 2,  0, 0,  0},
--                   {0, 0,  3, 0,  0},
--                   {0, 0,  0, 4,  0},
--                   {0, 0,  0, 0,  5}})
--
--   b = torch.FloatTensor({{2},
--                   {4},
--                   {6},
--                   {8},
--                   {10}})
--
-- print(a)
-- print(b)
--
--   x = torch.gesv(b, a)
--
-- print(x)
--
--   print(b:dist(a * x))
