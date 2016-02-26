require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local dataloader = require 'dptycho.io.dataloader'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local cunn = require 'cunn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local plt = plot()

-- torch.setdefaulttensortype('torch.CudaTensor')
--torch.setdefaulttensortype('torch.ZCudaTensor')
--print(type(torch.ZCudaTensor()))
--local data = u.load_sim_and_allocate('/home/philipp/projects/slicepp/Examples/configs/ball2.h5')
local data = u.load_sim_and_allocate_stacked('/home/philipp/projects/slicepp/Examples/configs/ball2.h5',true)

local params = {}
params.data = data
params.is_simulation = true
params.E = 80e3
params.M = data.probe:size(1)

-- pprint(data.real_deltas)
-- pprint(data.atompot)

local model = builder.build_model(params)

for k,v in ipairs(model) do
  local res = v:forward(data.probe)
  pprint(res)
  plt:plot(res[1]:float(),'result')
end









-- local ctor = torch.ZCudaTensor
--
-- local params = {}
-- local geo = {}
-- geo.z = 50e-2
-- geo.E = 2e5
-- geo.pix_size = 15e-6
-- geo.M = 256
-- params.geo = geo
--
-- params.is_simulation = true
-- params.sim_data = data
-- params.r_j = data.positions
-- params.I_i = data.measurements
--
-- local net = nn.Sequential()
-- for i=1,data.nslices do
--   local arm = nn.Sequential()
--   arm:add(znn.ConvParams(data.Wsize,data.atompot,data.inv_atompot,data.real_deltas[i],data.gradWeights[i]))
--   arm:add(znn.Sum(1,ctor))
--
--   net:add(znn.CMulModule(arm,ctor))
--   net:add(znn.ConvFFT2D(data.prop,data.bwprop))
-- end
-- --net:add(znn.FFT())
-- net:add(znn.ComplexAbs())
-- net:add(znn.Square())
-- -- add nn.Sum(1) here for multi mode
-- -- net:add(nn.Sqrt())
--
-- local res = net:forward(data.probe)
--
-- --psh = res:fftshift()
-- --plt:plot(psh:zfloat(),'shifted')
-- plt:plot(res:float(),'shifted')
