require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'
require 'nn'
require 'cunn'

local cudnn = require 'cudnn'
local znn = require 'dptycho.znn'
local u = require 'dptycho.util'
local stats = require "dptycho.util.stats"
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local dataloader = require 'dptycho.io.dataloader'
local plt = plot()

cudnn.benchmark = true
cudnn.fastest = true

local path = '/home/philipp/projects/dptycho/'
local file = 'simul.h5'
local f = hdf5.open(path..file, 'r')
local measure = f:read('/measure'):all():cuda()
local pos = f:read('/pos'):all()
f:close()

local path = '/home/philipp/projects/dptycho/'
local file = 'probe_object.h5'
local f = hdf5.open(path..file, 'r')
local obi = f:read('/ob_imag'):all()
local pri = f:read('/pr_imag'):all()
f:close()

-- plt:plot(pri:float())
local probe = torch.randn(measure:size()):view(1,pri:size(1),pri:size(2))
local object = torch.randn(obi:size())
local dO = torch.ZCudaTensor(object:size()):zero()
local tmp = torch.ZCudaTensor(probe:size())

local y = 1
local wse = znn.WSECriterion(y):cuda()

local step = 30
local nsteps = 10
local psh = probe:size(2)
local probe_size = probe:size():totable()
local mask = znn.SupportMask(probe_size,probe_size[#probe_size]/2)

local state = {
  learningRate = 1e-1,
  --lrd=1e-6,
  momentum=0.9,
  nesterov=false
}

local err = 0
local dLdW
for i=1,pos:size(1) do
  local ind = (i-1)*nsteps+j
  local xs, ys = pos[i][1],pos[i][2]
  local slice = {{xs,xs+psh-1},{ys,ys+psh-1}}

  local net = nn.Sequential()
  local source = znn.Source(nil,probe)
  -- source:immutable()

  net:add(source)
  net:add(mask)
  net:add(znn.CMul(object[slice],dO[slice]))
  -- propagate to fourier space
  net:add(znn.FFT())
  -- in: cx[R,M,M]  out: f[R,M,M]
  net:add(znn.ComplexAbs(tmp))
  -- in: [R,M,M]  out: [R,M,M]
  net:add(znn.Square())
  -- sum the mode intensities
  -- in: [R,M,M]  out: [1,M,M]
  net:add(znn.Sum(1,probe_size))
  -- make it 2 - dimensional
  -- in: [1,M,M]  out: [M,M]
  net:add(znn.Select(1,1))
  net:add(znn.Sqrt())

  local out = net:forward()
  err = wse:forward(out,measure[ind])
  dLdW = wse:backward(out,measure[ind])
  net:backward(nil,dLdW)



  plt:plot(out:float(),'output')
end

-- measure = measure:float()
-- local f = hdf5.open(path..'simul.h5', 'w')
-- f:write('/measure',measure)
-- f:write('/pos',pos)
-- f:close()
-- print(pos)
