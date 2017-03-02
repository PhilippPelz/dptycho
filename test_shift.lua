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

function shift_fourier(a,shift)
  local nx = a:size(1)
  local ny = a:size(2)
  local shx = -shift[1]
  local shy = -shift[2]

  local x = torch.FloatTensor(nx,1)
  x:copy(torch.linspace(-nx/2,nx/2 -1,nx))
  local x = torch.repeatTensor(x,1,ny)
  local y = torch.repeatTensor(torch.linspace(-ny/2,ny/2 -1,ny),nx,1):float()

  x:mul(shx/nx)
  y:mul(shy/ny)

  local xc = x:cuda()
  local yc = y:cuda()

  xc:fftshift()
  yc:fftshift()
  -- plt:plot(xc:float())
  -- plt:plot(yc:float())
  xc:add(yc)

  -- plt:plot(xc:float())

  xc:mul(2*math.pi)

  ramp = torch.ZCudaTensor.new(nx,ny):polar(1,xc)

  -- plt:plot(ramp)

  -- plt:plot(a,'probe before shift')
  a:fft():cmul(ramp):ifft()
  -- plt:plot(a,'probe after shift')
  return a
end

function shift_fourier_batched(a,shift)
  local batches = a:size(1)
  local nx = a:size(2)
  local ny = a:size(3)
  local shx = -shift[1]
  local shy = -shift[2]

  local x = torch.FloatTensor(nx,1)
  x:copy(torch.linspace(-nx/2,nx/2 -1,nx))
  local x = torch.repeatTensor(x,1,ny)
  local y = torch.repeatTensor(torch.linspace(-ny/2,ny/2 -1,ny),nx,1):float()

  x:mul(shx/nx)
  y:mul(shy/ny)

  local xc = x:cuda()
  local yc = y:cuda()

  xc:fftshift()
  yc:fftshift()
  -- plt:plot(xc:float())
  -- plt:plot(yc:float())
  xc:add(yc)

  -- plt:plot(xc:float())

  xc:mul(2*math.pi)

  local ramp = torch.ZCudaTensor.new(nx,ny):polar(1,xc)

  ramp = ramp:view(1,nx,ny):expandAs(a)

  -- plt:plot(ramp)

  -- plt:plot(a,'probe before shift')
  a:fftBatched():cmul(ramp):ifftBatched()
  -- plt:plot(a,'probe after shift')
  return a
end

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
for a = -30,30 do
  s = {a/3.0,a/2.0}
  pprint(s)
  dest:shift(t,torch.FloatTensor(s))
  plt:plot(dest[1]:zfloat(),'shifted bilinear')
  local fsh = shift_fourier_batched(t:clone(),torch.FloatTensor(s))
  plt:plot(fsh[1]:zfloat(),'shifted fourier')
  -- plt:plotReIm(probe:clone():add(-1,dest)[1]:zfloat(),'diff')
  -- plt:plot(probe:clone():add(-1,dest)[1]:zfloat(),'diff')
end
