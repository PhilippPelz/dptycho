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
local optim = require "optim"

local plt = plot()


local path = '/home/philipp/projects/slicepp/Examples/configs/'
local file = 'ball2.h5'
local data = u.load_sim_and_allocate_stacked(path,file,true)

local params = {}
params.data = data
params.is_simulation = true
params.E = 80e3
params.M = data.probe:size(2)
params.dx = 0.240602

-- table of networks
local build = builder(params)
-- collectGarbage()
-- data.deltas:size():totable()
local init = stats.truncnorm(data.deltas:size():totable(),0,1,0.3,0.05)
-- plt:plot(init,'init 1')
-- pprint(init)
data.deltas:copy(init)
-- pprint(data.deltas)

-- plt:plot3d(data.deltas[1]:float())
-- plt:plot(data.a_k[1]:float(),'measurement 1')

local model
local weight_penalty
local y = 1
local wse = znn.WSECriterion(y):cuda()
local weight_regul = znn.AtomRadiusPenalty(data.Znums:size(1),params.r_atom_xy,params.r_atom_z,0.1)

local gradParameters = data.gradWeights
local parameters = data.deltas

-- local d  = data.deltas:select(2,1)
-- pprint(d)

-- local err, outputs
-- feval = function(x)
--   --  model:zeroGradParameters()
--    outputs = model:forward()
--    err = criterion:forward(outputs, labels)
--    local gradOutputs = criterion:backward(outputs, labels)
--    model:backward(inputs, gradOutputs)
--    return err, gradParameters
-- end
-- optim.sgd(feval, parameters, optimState)
-- local diff = torch.CudaTensor(data.a_k[i])

local state = {
  learningRate = 1e-2
}
local dLdW, err, regul_err, outputs, mask
local epochs = 10

local feval = function(x)
  --  model:zeroGradParameters()
   return err, gradParameters
end
local max_grad = 1e3
local timer = torch.Timer()
for e=1,epochs do
  regul_err = weight_regul:forward(parameters)
  print(string.format('%03d regul error: %-3.5f',e,regul_err))

  local dRdW = weight_regul:backward(parameters)
  -- pprint(dRdW)
  -- plt:plot3d(dRdW[1]:float())
  -- plt:plot3d(build.coverage[1]:float(),'build.coverage')
  err = 0
  for i=1, params.K do
    model = build:get_tower(i)
    outputs = model:forward()
    -- plt:plot(outputs:clone():float(),'model output '..i)

    -- plt:plotcompare({outputs:float(),data.a_k[i]:float()})
    err = err + wse:forward(outputs,data.a_k[i])
    -- print(string.format('%03d WS    error: %-3.5f',i,err))
    dLdW = wse:backward(outputs,data.a_k[i])
    -- plt:plot(dLdW:float(),'dLdW')
    model:backward(nil,dLdW)

    collectgarbage()
    -- u.printMem()
  end
  print(string.format('gradParameters min: %f max:%f',gradParameters:min(),gradParameters:max()))
  -- pprint(gradParameters)
  -- pprint(build.coverage)
  print(string.format('dRdW           min: %f max:%f',dRdW:min(),dRdW:max()))
  print(string.format('coverage       min: %f max:%f',build.coverage:min(),build.coverage:max()))
  u.printf('average error: %f',err/params.K)
  -- pprint(dRdW)
  -- plt:plot3d(dRdW[1]:float())
  mask = torch.gt(gradParameters,max_grad)
  gradParameters:maskedFill(mask,max_grad)
  mask = torch.lt(gradParameters,-max_grad)
  gradParameters:maskedFill(mask,-max_grad)
  -- plt:plot3d(gradParameters[1]:float(),'gradParameters',0,1)
  gradParameters:add(dRdW)
  -- plt:plot3d(gradParameters[1]:float(),'gradParameters add')

  gradParameters:cmul(build.coverage)
  optim.adam(feval, parameters, state)
  u.printf('Time elapsed after epoch %d: %f seconds',e, timer:time().real)
  print('----------------------------------------------------------------')
  -- plt:plot3d(parameters[1]:float(),'parameters epoch ' .. e)
  collectgarbage()
  u.printMem()
end
--
--   weight_penalty = weight_regul:forward(data.deltas)
--   print(string.format('weight penalty: %f',weight_penalty))
-- end
local function generate_measurements()
  local f = hdf5.open('/home/philipp/projects/slicepp/Examples/configs/a_k.h5', 'w')
  for i=1, params.K do
    model = build:get_tower(i)
    outputs = model:forward()
    -- plt:plot(outputs:float(),'model output '..i)
    -- plt:plotcompare({outputs:float(),data.a_k[1]:float()})
    data.a_k[i]:copy(outputs)
    print(i)
  end

  f:write('/a_k',data.a_k:float())
  f:close()
end

-- generate_measurements()
