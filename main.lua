require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local dataloader = require 'dptycho.io.dataloader'

require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local znn = require 'dptycho.znn'

cudnn.benchmark = true
cudnn.fastest = true
local u = require 'dptycho.util'
local stats = require "dptycho.util.stats"
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
params.M = data.probe:size(2)
params.dx = 0.240602

-- pprint(data.deltas)
-- pprint(data.atompot)

-- table of networks
local model = builder.build_model(params)
-- data.deltas:size():totable()
local init = stats.truncnorm(data.deltas:size():totable(),0,1,0.3,0.05)
-- plt:plot(init,'init 1')
-- pprint(init)
data.deltas:copy(init)
pprint(data.deltas)
local r_atom = math.floor(params.Vsize / params.dx)
r_atom = ( r_atom % 2 == 0 ) and ( r_atom - 1 ) or r_atom
print(string.format('r_atom = %d',r_atom))
local r_atom_xy = r_atom
local r_atom_z = 1

-- plt:plot(data.a_k[1]:float(),'measurement 1')

local net = model[1]
local y = 1
local wse = znn.WSECriterion(y):cuda()
local weight_regul = znn.AtomRadiusPenalty(data.Znums:size(1),r_atom_xy,r_atom_z,0.1)
-- local d  = data.deltas:select(2,1)
-- pprint(d)
local weight_penalty = weight_regul:forward(data.deltas)
print(string.format('weight penalty: %f',weight_penalty))

local a_model = net:forward()
-- plt:plot(a_model:clone():float(),'a_modelput 1')
local error = wse:forward(a_model,data.a_k[1])

-- print(a_model:max())
-- print(a_model:min())
-- print(a_model:float())
-- print(data.a_k[1]:max())
-- print(data.a_k[1]:min())
-- print(data.a_k[1]:float())
print(string.format('Error of measurement %d: %-3.5f',1,error))
pprint(data.a_k[1])
local dLdW = wse:backward(a_model,data.a_k[1])
-- plt:plot(dLdW:float(),'dLdW')
print('before backward')
net:backward(nil,dLdW)

-- local m = znn.SupportMask({params.M,params.M},params.M/2)

-- local m = u.initial_probe({params.M,params.M},0.5)

-- local net = znn.AtomRadiusPenalty(data.Znums:size(1),r_atom,1)
--
-- pprint(data.deltas)
-- plt:plot(data.deltas[1][3]:float(),'deltas 3')
-- local res = net:forward(data.deltas)
-- plt:plot(res[1][3]:float(),'res 3')
-- pprint(res)
-- print(res)

u.printMem()

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
