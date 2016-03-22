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
local f2 = hdf5.open('probe.h5', 'r')
local pr = f2:read('/pr'):all():cuda()
local pi = f2:read('/pi'):all():cuda()
f2:close()
data.nslices = 1
data.probe[1]:copyIm(pi):copyRe(pr)
-- plt:plot(data.probe[1]:zfloat(),'probe')
-- pprint(data)
local s = {{},{1},{},{}}
data.deltas:zero()
-- print('here')
data.deltas = data.deltas[s]
-- print('here')
data.gradWeights = data.gradWeights[s]
-- print('here')
-- local d = data.deltas[1][s]
-- local d2 = data.deltas[3][s]
-- print(data.deltas[2]:max())
-- plt:plotcompare({d:float(),d2:float()})
local s = data.deltas:size(4)
local pos = torch.Tensor(2,200,2):uniform():mul(s*2/3):add(s/10):int()
for i = 1,pos:size(1) do
  for j=1,pos:size(2) do
    local x = math.max(1,math.min(pos[i][j][1],data.deltas:size(4)))
    local y = math.max(1,math.min(pos[i][j][2],data.deltas:size(4)))
    pos[i][j][1] = x
    pos[i][j][2] = y
    local slice = {{i},{1},{x},{y}}
    -- pprint(slice)
    data.deltas[slice]:fill(3)
  end
end
-- print('here')
pprint(data)

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
local y = 1
local wse = znn.WSECriterion(y):cuda()
local weight_regul = znn.AtomRadiusPenalty(data.Znums:size(1),params.r_atom_xy,params.r_atom_z,0.1)

local gradParameters = data.gradWeights
local parameters = data.deltas


local state = {
  learningRate = 1,
  lrd=1e-6,
  momentum=0.9,
  nesterov=false
}
local dLdW, dRdW, err, regul_err, outputs, mask
local epochs = 10


local max_grad = 1e3
local timer = torch.Timer()
-- for e=1,epochs do
--
--   -- pprint(dRdW)
--   -- plt:plot3d(dRdW[1]:float())
--   -- plt:plot3d(build.coverage[1]:float(),'build.coverage')
--   err = 0
--   for i=1, params.K do
--
--     model, del, grad = build:get_tower(i)
--
--     local feval = function(x)
--       --  model:zeroGradParameters()
--        return err, grad
--     end
--
--     regul_err = weight_regul:forward(del)
--     outputs = model:forward()
--     err = err + wse:forward(outputs,data.a_k[i])
--     dLdW = wse:backward(outputs,data.a_k[i])
--     dRdW = weight_regul:backward(del)
--     model:backward(nil,dLdW)
--
--     collectgarbage()
--     -- plt:plot(dLdW:float(),'dLdW')
--     -- u.printMem()
--     -- pprint(dRdW)
--     -- plt:plot(outputs:clone():float(),'model output '..i)
--     -- plt:plotcompare({outputs:float(),data.a_k[i]:float()})
--   -- end
--     -- pprint(gradParameters)
--     -- pprint(build.coverage)
--     -- print(string.format('coverage       min: %f max:%f',build.coverage:min(),build.coverage:max()))
--     -- u.printf('average error: %f',err/params.K)
--     u.printf('E: %-10.2f   E_reg: %-10.2f   min,max: dParam (%-7.2f,%-7.2f) dRdW (%-7.2f,%-7.2f)',err,regul_err,grad:min(),grad:max(),dRdW:min(),dRdW:max())
--     err = 0
--     -- pprint(dRdW)
--     -- plt:plot3d(dRdW[1]:float())
--     -- mask = torch.gt(gradParameters,max_grad)
--     -- gradParameters:maskedFill(mask,max_grad)
--     -- mask = torch.lt(gradParameters,-max_grad)
--     -- gradParameters:maskedFill(mask,-max_grad)
--     -- plt:plot3d(gradParameters[1]:float(),'gradParameters',0,1)
--     grad:add(dRdW)
--
--     -- gradParameters:cmul(build.coverage)
--     -- optim.adam(feval, parameters, state)
--     optim.sgd(feval, del, state)
--     -- plt:plot3d(parameters[1]:float(),'Parameters')
--     gradParameters:zero()
--     -- print('----------------------------------------------------------------')
--   end
--   u.printf('Time elapsed after epoch %d: %f seconds',e, timer:time().real)
--   print('----------------------------------------------------------------')
--   -- plt:plot3d(parameters[1]:float(),'parameters epoch ' .. e)
--   collectgarbage()
--   u.printMem()
-- end

--   weight_penalty = weight_regul:forward(data.deltas)
--   print(string.format('weight penalty: %f',weight_penalty))
-- end
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
  -- f:write('/pos',pos)
  f:close()
end

generate_measurements()
