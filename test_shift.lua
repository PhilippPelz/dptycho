require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'

local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local znn = require "dptycho.znn"
local plot = require 'dptycho.io.plot'
local plt = plot()


local t = torch.CudaTensor.new(1,50,50):fill(0)
-- local t = torch.CudaTensor.new(1,50,50):fillRe(0):fillIm(0)

sl = t[{1,{4,10},{4,10}}]

-- t:fillRe(2)
sl:fill(1)
-- sl = t[{2,{30,40},{30,40}}]
-- sl:fillRe(1)

local dest = torch.CudaTensor.new(1,50,50):fill(0)
local fw = torch.CudaTensor.new(1,50,50):fill(0)
local bw = torch.CudaTensor.new(1,50,50):fill(0)
local t1 = torch.CudaTensor.new(1,50,50):fill(0)
local t2 = torch.CudaTensor.new(1,50,50):fill(0)
-- local dest = torch.CudaTensor.new(1,50,50):fillRe(0):fillIm(0)
-- local fw = torch.CudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- local bw = torch.CudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- plt:plotcompare({t[1]:re():float(),dest[1]:re():float()})
-- plt:plotcompare({t[2]:re():float(),dest[2]:re():float()})
-- plt:plot(t[1]:re():float())
-- dest:shift(t,torch.FloatTensor({2.3,2.3}))
-- fw:shift(t,torch.FloatTensor({-1,0}))
-- plt:plot(fw[1]:float())
-- bw:shift(t,torch.FloatTensor({1,0})):mul(-1)
-- plt:plot(bw[1]:float())
dest:dx(t,fw,bw)
plt:plot(dest[1]:float())

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
