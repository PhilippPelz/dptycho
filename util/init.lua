local classic = require 'classic'
local dataloader = require 'dptycho.io.dataloader'
local plot = require 'dptycho.io.plot'
local plt = plot()
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
  ret.positions = data.positions
  ret.atompot = {}
  ret.inv_atompot = {}
  for Z, pot in pairs(data.atompot) do
  --  print(Z)
    ret.atompot[Z] = pot:mul(pot:nElement()):zcuda()
    ret.inv_atompot[Z] = ret.atompot[Z]:pow(-1)
    -- plt:plot(inv_pot[Z]:zfloat())
  end
  ret.deltas = {}
  ret.gradWeights = {}
  for Z, delta in pairs(data.deltas) do
    ret.deltas[Z] = data.deltas[Z]:re():cuda()
    ret.gradWeights[Z] = ret.deltas[Z]:clone():zero()
  end
  ret.nslices = data.deltas[data.Znums[1]]:size():totable()[1]
  return ret
end

function m.load_sim_and_allocate_stacked(file,ptycho)
  local ret = {}
  local d = dataloader()
  local data = d:loadHDF5(file,ptycho)
--  pprint(data)
  ret.nslices = data.deltas[1]:size():totable()[1]
  ret.prop = data.propagator:zcuda()
  ret.bwprop = ret.prop:clone():conj()
  ret.probe = data.probe:zcuda()
  ret.positions = data.positions
  ret.Znums = data.Znums

  local ps = ret.probe[1]:size():totable()
  local s = data.atompot[1]:size():totable()
  local offset = math.floor((s[1] - ps[1])/2)
  local mi = {offset,offset+ps[1]-1}
  local middle = {mi,mi}
  -- pprint(middle)
--  pprint(s)
  local potsize = {#data.atompot,ps[1],ps[2]}
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
    -- pprint(pot)
    local p = pot:mul(pot:nElement()):fftshift()
    -- plt:plot(p,'middle of atompot 1')
    -- pprint(p)
    p = p[middle]
    -- pprint(p)
    p:fftshift()
    -- plt:plot(p,'middle of atompot 2')/

    ret.atompot[i]:copy(p)
--    pprint(ret.atompot)
    -- local pow = ret.atompot[i]:clone()
--    pprint(pow)
    ret.inv_atompot[i]:copy(ret.atompot[i]):pow(-1)
    i = i + 1
    -- plt:plot(inv_pot[Z]:zfloat())
  end
--  print('wp 1')
  ret.deltas = torch.CudaTensor(#data.atompot,ret.nslices,s[1],s[2])
--  pprint(ret.deltas)
--  print('wp 2')
  ret.gradWeights = torch.CudaTensor(#data.atompot,ret.nslices,s[1],s[2]):zero()
--  print('wp 3')
  for j =1, ret.nslices do
    for k, delta in ipairs(data.deltas) do
      -- local s = ret.deltas[j]
--      pprint(s)
--      pprint(data.deltas[Z])
      ret.deltas[k][j]:copy(data.deltas[k][j]:re())      
    end
  end
  return ret
end

return m
