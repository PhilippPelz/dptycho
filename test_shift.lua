require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
<<<<<<< HEAD

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
=======
-- local plot = require 'dptycho.io.plot'
-- local builder = require 'dptycho.core.netbuilder'
-- local optim = require "optim"
-- local znn = require "dptycho.znn"
-- local plt = plot()
--
--
-- local t = torch.ZCudaTensor.new(2,50,50):fillRe(0):fillIm(0)
--
-- sl = t[{1,{5,10},{5,10}}]
-- sl:fillRe(1)
--
-- sl = t[{2,{30,40},{30,40}}]
-- sl:fillRe(1)
--
-- local dest = torch.ZCudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- local fw = torch.ZCudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- local bw = torch.ZCudaTensor.new(2,50,50):fillRe(0):fillIm(0)
-- -- plt:plotcompare({t[1]:re():float(),dest[1]:re():float()})
-- -- plt:plotcompare({t[2]:re():float(),dest[2]:re():float()})
--
-- -- dest:shift(t,torch.FloatTensor({-0.3,-0.7}))
-- dest:dy(t,fw,bw)
--
-- -- plt:plot(t:add(-1,dest)[1]:re():float())
>>>>>>> 5400af4241e2b79a156cfca7ef79ec71554c2453
-- plt:plotcompare({fw[1]:re():float(),bw[1]:re():float()})
-- plt:plotcompare({fw[2]:re():float(),bw[2]:re():float()})
-- plt:plotcompare({t[1]:re():float(),dest[1]:re():float()})
-- plt:plotcompare({t[2]:re():float(),dest[2]:re():float()})

<<<<<<< HEAD

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
=======
  a = torch.FloatTensor({{6.80, -2.11,  5.66,  5.97,  8.23},
                  {-6.05, -3.30,  5.36, -4.44,  1.08},
                  {-0.45,  2.58, -2.70,  0.27,  9.04},
                  {8.32,  2.71,  4.35,  -7.17,  2.14},
                  {-9.67, -5.14, -7.26,  6.08, -6.87}}):t()

  b = torch.FloatTensor({{4.02,  6.19, -8.22, -7.57, -3.03},
                  {-1.56,  4.00, -8.67,  1.75,  2.86},
                  {9.81, -4.09, -4.57, -8.61,  8.99}}):t()






  x = torch.gesv(b, a)


  print(b:dist(a * x))
>>>>>>> 5400af4241e2b79a156cfca7ef79ec71554c2453
