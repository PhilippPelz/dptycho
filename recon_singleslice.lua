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
data.nslices = 1

local f2 = hdf5.open('probe.h5', 'r')
local pr = f2:read('/pr'):all():cuda()
local pi = f2:read('/pi'):all():cuda()
data.probe[1]:copyIm(pi):copyRe(pr)
f2:close()

local inv_probe_intens = data.probe:abs():pow(-2)

local f = hdf5.open('/home/philipp/projects/slicepp/Examples/configs/a_k1.h5', 'r')
data.solution = f:read('/deltas'):all():cuda()
data.a_k = f:read('/a_k'):all():cuda()
-- data.positions = f:read('/pos'):all()
f:close()

local init = stats.truncnorm(data.deltas:size():totable(),0,1,0.3,0.05)
-- data.deltas:copy(init)


local s = {{},{1},{},{}}
data.gradWeights = data.gradWeights[s]
data.deltas = data.deltas[s]
-- data.deltas:randn(data.deltas:size())
data.deltas:zero()

local params = {}
params.data = data
params.is_simulation = true
params.E = 80e3
params.M = data.probe:size(2)
params.dx = 0.240602

-- table of networks
local build = builder(params)

local model, del, grad
local weight_penalty


local gradParameters = data.gradWeights
local parameters = data.deltas

pprint(gradParameters)

local state = {
  learningRate = 1e-2,
  lrd=1e-6,
  momentum=0.9,
  nesterov=false
}

local dLdW, dRdW, err, regul_err, outputs, mask
local epochs = 50
local l1weight = 0.05
local max_grad = 1e3

local y = 1e-8
local wse = znn.WSECriterion(y):cuda()
local l1 = znn.WeightedL1Cost(l1weight)
local timer = torch.Timer()

local feval = function(x)
  --  model:zeroGradParameters()
   return err, gradParameters
end
local avg_err = 0
for e=1,epochs do
--
--   -- pprint(dRdW)
--   -- plt:plot3d(dRdW[1]:float())
--   -- plt:plot3d(build.coverage[1]:float(),'build.coverage')
--   err = 0
-- params.K
  for i=1, 1 do
--
    model, del, grad = build:get_tower(i)
    outputs = model:forward()
    err = wse:forward(outputs,data.a_k[i])
    -- u.printf("%d err = %-5.2g",i,err)
    -- plt:plotcompare({outputs:float():log(),data.a_k[i]:float():log()})
    dLdW = wse:backward(outputs,data.a_k[i])
    -- u.printf("dLdW mean: %f   min: %f   max: %f",dLdW:mean(),dLdW:min(),dLdW:max())
    -- plt:plot(dLdW:float(),'dLdW')
--     dRdW = weight_regul:backward(del)
    model:backward(nil,dLdW)



    avg_err = avg_err + err
--
--     collectgarbage()

--     -- u.printMem()
--     -- pprint(dRdW)
--     -- plt:plot(outputs:clone():float(),'model output '..i)
--     -- plt:plotcompare({outputs:float(),data.a_k[i]:float()})
  end
  -- pprint(parameters)
  -- pprint(data.solution)
  local err_solution = wse:forward(parameters,data.solution)
  local l1cost = l1:forward(parameters)
  local dL1dW = l1:backward(parameters)
  u.printf("average err = %-5.2g  solution err = %-5.2g   l1=%-5.2g time: %f s",avg_err/params.K,err_solution,l1cost,timer:time().real)
  avg_err = 0

  -- plt:plotcompare({dL1dW[1][1]:float(),dL1dW[2][1]:float()})
  -- plt:plotcompare({gradParameters[1][1]:float(),gradParameters[2][1]:float()})

  gradParameters:add(dL1dW)
--     -- optim.adam(feval, parameters, state)
  -- plt:plotcompare({parameters[1][1]:float(),parameters[2][1]:float()})
  optim.sgd(feval, parameters, state)
  --     -- plt:plot3d(parameters[1]:float(),'Parameters')
  gradParameters:zero()
  -- print('----------------------------------------------------------------')
  collectgarbage()
--   u.printMem()
end

local function generate_measurements()
  local f = hdf5.open('/home/philipp/projects/slicepp/Examples/configs/a_k1.h5', 'w')
  for i=1, params.K do
    model = build:get_tower(i)
    outputs = model:forward()
    -- plt:plot(outputs:float():log(),'model output '..i)
    -- plt:plotcompare({outputs:float(),data.a_k[1]:float()})
    data.a_k[i]:copy(outputs)
    print(i)
  end

  f:write('/a_k',data.a_k:float())
  f:write('/deltas',data.deltas:float())
  -- f:write('/probe_r',data.probe:zfloat())
  f:close()
end

-- generate_measurements()
