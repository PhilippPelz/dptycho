require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'
require 'nn'
require 'cunn'

local dataloader = require 'dptycho.io.dataloader'
local cudnn = require 'cudnn'
local znn = require 'dptycho.znn'
local u = require 'dptycho.util'
local stats = require "dptycho.util.stats"
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local plt = plot()

cudnn.benchmark = true
cudnn.fastest = true
local path = '/home/philipp/projects/slicepp/Examples/configs/'
local file = 'ball2.h5'
local data = u.load_sim_and_allocate_stacked(path,file,true)

local params = {}
params.data = data
params.is_simulation = true
params.E = 80e3
params.M = data.probe:size(2)
params.dx = 0.240602

local build = builder(params)
local init = stats.truncnorm(data.deltas:size():totable(),0,1,0.3,0.05)
-- plt:plot(init,'init 1')
-- pprint(data.deltas)
data.deltas[1][40]:copy(init[1][40])
-- pprint(data.deltas)

-- plt:plot3d(data.deltas[1]:float())
-- plt:plot(data.a_k[1]:float(),'measurement 1')

local model, del, grad
local weight_penalty = 1e-3
local y = 1
local wse = znn.WSECriterion(y):cuda()
local weight_regul = znn.AtomRadiusPenalty(data.Znums:size(1),params.r_atom_xy,params.r_atom_z,weight_penalty)

local gradParameters = data.gradWeights
local parameters = data.deltas
local state = {
  learningRate = 1e-1,
  --lrd=1e-6,
  momentum=0.9,
  nesterov=false
}
local dLdW, dRdW, err, regul_err, outputs, mask
local epochs = 10


local max_grad = 1e3
local timer = torch.Timer()
for e=1,epochs do

  -- pprint(dRdW)
  -- plt:plot3d(dRdW[1]:float())
  -- plt:plot3d(build.coverage[1]:float(),'build.coverage')
  err = 0
  --params.K
  for i=1,1  do

    model, del, grad = build:get_tower(i)

    local feval = function(x)
      --  model:zeroGradParameters()
       return err, grad
    end

    regul_err = weight_regul:forward(parameters)
    outputs = model:forward()
    -- plt:plotcompare({outputs:float(),data.a_k[i]:float()})
    err = err + wse:forward(outputs,data.a_k[i])
    dLdW = wse:backward(outputs,data.a_k[i])
    -- plt:plot(dLdW:float(),'dLdW')
    dRdW = weight_regul:backward(del)
    model:backward(nil,dLdW)

    collectgarbage()

    -- u.printMem()
    -- pprint(dRdW)
    -- plt:plot(outputs:float(),'model output '..i)
    -- plt:plotcompare({outputs:float(),data.a_k[i]:float()})
  -- end
    -- pprint(dLdW)
    -- pprint(dRdW)
    -- print(string.format('coverage       min: %f max:%f',build.coverage:min(),build.coverage:max()))
    -- u.printf('average error: %f',err/params.K)
    u.printf('E: %-10.2f   E_reg: %-10.2f   min,max: param (%-7.2f,%-7.2f,%-7.2f) dParam (%-7.2f,%-7.2f) dRdW (%-7.2f,%-7.2f)',err,regul_err,parameters:min(),parameters:max(),parameters:mean(),grad:min(),grad:max(),dRdW:min(),dRdW:max())
    err = 0
    -- pprint(dRdW)
    -- plt:plot(dRdW[1][39]:float())
    -- mask = torch.gt(gradParameters,max_grad)
    -- gradParameters:maskedFill(mask,max_grad)
    -- mask = torch.lt(gradParameters,-max_grad)
    -- gradParameters:maskedFill(mask,-max_grad)
    -- plt:plot3d(gradParameters[1]:float(),'gradParameters',0,1)
    -- grad:add(dRdW)
    -- plt:plot3d(grad[1]:float(),'grad')
    -- gradParameters:cmul(build.coverage)
    -- optim.adam(feval, parameters, state)
    optim.sgd(feval, del, state)
    -- plt:plot3d(parameters[1]:float(),'Parameters')
    gradParameters:zero()
    -- print('----------------------------------------------------------------')
  end
  u.printf('Time elapsed after epoch %d: %f seconds',e, timer:time().real)
  print('----------------------------------------------------------------')
  -- plt:plot3d(parameters[1]:float(),'parameters epoch ' .. e)
  collectgarbage()
  -- u.printMem()
end

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
