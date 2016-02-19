local classic = require 'classic'
local dataloader = require 'dptycho.io.dataloader'
require 'pprint'
require 'zcutorch'
local m = classic.module(...)

-- m:class("Allocator")

function m.load_sim_and_allocate(file)
  local ret = {}
  local d = dataloader()
  local data = d:loadHDF5(file)
--  pprint(data)
  ret.prop = data.propagator:zcuda()
  ret.bwprop = ret.prop:clone():conj()
  ret.probe = data.probe:zcuda()
  ret.atompot = {}
  ret.inv_atompot = {}
  for Z, pot in pairs(data.atompot) do
  --  print(Z)
    ret.atompot[Z] = pot:mul(pot:nElement()):zcuda()
    ret.inv_atompot[Z] = ret.atompot[Z]:pow(-1)
    -- plt:plot(inv_pot[Z]:zfloat())
  end
  ret.real_deltas = {}
  ret.gradWeights = {}
  for Z, delta in pairs(data.deltas) do
    ret.real_deltas[Z] = data.deltas[Z]:re():cuda()
    ret.gradWeights[Z] = ret.real_deltas[Z]:clone():zero()
  end
  ret.nslices = data.deltas[data.Znums[1]]:size():totable()[1]
  return ret
end

function m.load_sim_and_allocate_stacked(file)
  local ret = {}
  local d = dataloader()
  local data = d:loadHDF5(file)
--  pprint(data)
  ret.nslices = data.deltas[1]:size():totable()[1]
  ret.prop = data.propagator:zcuda()
  ret.bwprop = ret.prop:clone():conj()
  ret.probe = data.probe:zcuda()
  ret.atompot = {}
  ret.inv_atompot = {}
  local s = data.atompot[1]:size():totable()
--  pprint(s)
  local potsize = {#data.atompot,s[1],s[2]}
  ret.Wsize = potsize
--  pprint(potsize)
  ret.atompot = torch.ZCudaTensor().new(potsize)
  ret.inv_atompot = torch.ZCudaTensor().new(potsize)
--  pprint(ret.atompot:size())
--  pprint(ret.inv_atompot:size())
  local i = 1
  for Z, pot in pairs(data.atompot) do
--    print(Z)
    local s = ret.atompot[i]
--    print('after s')
--    pprint(s)
    ret.atompot[i]:copy(pot:mul(pot:nElement()))
--    pprint(ret.atompot)
    local pow = ret.atompot[i]:pow(-1)
--    pprint(pow)
    ret.inv_atompot[i]:copy(pow)
    i = i + 1
    -- plt:plot(inv_pot[Z]:zfloat())
  end
--  print('wp 1')
  ret.real_deltas = torch.CudaTensor(ret.nslices,#data.atompot,s[1],s[2])
--  pprint(ret.real_deltas)
--  print('wp 2')
  ret.gradWeights = torch.CudaTensor(ret.nslices,#data.atompot,s[1],s[2]):zero()
--  print('wp 3')
  for j =1, ret.nslices do
    local k = 1
    for Z, _ in pairs(data.deltas) do
      local s = ret.real_deltas[j]
--      pprint(s)
--      pprint(data.deltas[Z])
      ret.real_deltas[j][k]:copy(data.deltas[Z][j]:re())
      k = k + 1
    end
  end
  return ret
end

return m
