require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local dataloader = require 'dptycho.io.dataloader'
local znn = require 'dptycho.znn'
require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local plt = plot()

local function printMem()
  local freeMemory, totalMemory = cutorch.getMemoryUsage(cutorch.getDevice())
  freeMemory = freeMemory / (1024*1024)
  totalMemory = totalMemory / (1024*1024)
  print(string.format('free: %d MB, total: %d MB, used: %d MB',freeMemory,totalMemory,totalMemory -freeMemory))
end
-- torch.setdefaulttensortype('torch.CudaTensor')
--torch.setdefaulttensortype('torch.ZCudaTensor')
--print(type(torch.ZCudaTensor()))
--local data = u.load_sim_and_allocate('/home/philipp/projects/slicepp/Examples/configs/ball2.h5')
local data = u.load_sim_and_allocate_stacked('/home/philipp/projects/slicepp/Examples/configs/ball2.h5',true)

local params = {}
params.data = data
params.is_simulation = true
params.E = 80e3
params.M = data.probe:size(2)
params.dx = 0.240602

-- pprint(data.deltas)
-- pprint(data.atompot)

local model = builder.build_model(params)


local r_atom = math.floor(params.Vsize / params.dx)
r_atom = ( r_atom % 2 == 0 ) and ( r_atom - 1 ) or r_atom
print(string.format('r_atom = %d',r_atom))

local nInputPlane = data.Znums:size(1)
local nOutputPlane = data.Znums:size(1)
local filter = torch.ones(nOutputPlane,nInputPlane,1,r_atom,r_atom):cuda()
local bias = torch.zeros(nOutputPlane):cuda()
local dT, dW, dH = 1,1,1
local padT, padW, padH = 1, (r_atom-1)/2, (r_atom-1)/2

local net = nn.Sequential()

net:add(znn.VolumetricConvolutionFixedFilter(nInputPlane, nOutputPlane, 1, r_atom, r_atom , dT, dW, dH, padT, padW, padH, filter, bias))
net:add(nn.Threshold(1,1,true):cuda())

-- pprint(data.deltas[1])
local res = net:forward(data.deltas)

pprint(res)
print(res:sum())

printMem()

-- for k,v in ipairs(model) do
--   local res = v:forward(data.probe)
--   print(k)
--   -- pprint(res)
--   -- plt:plot(res:float(),'result')
-- end








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
--   arm:add(znn.ConvParams(data.Wsize,data.atompot,data.inv_atompot,data.deltas[i],data.gradWeights[i]))
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
