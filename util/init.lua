local classic = require 'classic'
local dataloader = require 'dptycho.io.dataloader'
local plot = require 'dptycho.io.plot'
local plt = plot()
local py = require('fb.python')
require 'pprint'
require 'zcutorch'
local m = classic.module(...)

m:submodule("stats")
m:class("linear_schedule")
m:class("tabular_schedule")

py.exec([=[
import numpy as np
from numpy.fft import fft

def DTF2D(N):
  W = fft(np.eye(N))
  W2D = np.kron(W,W)/N
  return W2D.real, W2D.imag
]=])

function m.DTF2D(N)
  local WR,WI = py.eval('DTF2D(N)',{N=N})
  local f = torch.FloatTensor(N,N,2)
  z[{{},{},{1}}]:copy(WR)
  z[{{},{},{2}}]:copy(WI)
  local z = torch.ZFloatTensor(N,N):copy(f)
  return z
end

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
    ret.deltas[Z] = delta:re():cuda()
    ret.gradWeights[Z] = delta:clone():zero()
  end
  ret.nslices = data.deltas[data.Znums[1]]:size():totable()[1]
  return ret
end

function m.load_sim_and_allocate_stacked(path,file,ptycho)
  local ret = {}
  local d = dataloader()
  local data = d:loadHDF5(path,file,ptycho)
--  pprint(data)
  ret.nslices = data.deltas[1]:size():totable()[1]
  ret.prop = data.propagator:zcuda()
  ret.bwprop = ret.prop:clone():conj()
  ret.probe = data.probe:zcuda()
  ret.positions = data.positions
  ret.Znums = data.Znums
  ret.a_k = data.a_k:cuda()--:sqrt()

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
  for _, pot in pairs(data.atompot) do
--    print(Z)
    -- local s = ret.atompot[i]
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

function m.printMem()
  local freeMemory, totalMemory = cutorch.getMemoryUsage(cutorch.getDevice())
  freeMemory = freeMemory / (1024*1024)
  totalMemory = totalMemory / (1024*1024)
  print(string.format('free: %d MB, total: %d MB, used: %d MB',freeMemory,totalMemory,totalMemory -freeMemory))
end

function m.printram(str)
  -- print('================ MEMORY REPORT ================')
  -- print(str)
  -- -- free -m
  -- local handle = io.popen('free -m | cat')
  -- if handle then
  --   local result = handle:read("*a")
  --   print(result)
  --   handle:close()
  -- end
  -- print('===============================================')
end

function m.initial_probe(size1,support_ratio)
  local ratio = support_ratio or 0.5
  local size = m.copytable(size1)
  size[#size+1] = 2
  -- pprint(size)
  local r = torch.randn(unpack(size))
  -- pprint(r)
  local res = torch.ZFloatTensor(unpack(size1)):copy(r)
  -- pprint(res)
  local mod = znn.SupportMask(size1,size1[#size1]*ratio)
  res = mod:forward(res)
    -- pprint(res)

  plt:plot(res,'probe')
  return res
end

function m.copytable(obj, seen)
  if type(obj) ~= 'table' then return obj end
  if seen and seen[obj] then return seen[obj] end
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do
    if k ~= '_classAttributes' then
      -- print(k)
      -- pprint(v)
      res[m.copytable(k, s)] = m.copytable(v, s)
    end
  end
  return res
end

function m.meshgrid(x,y)

end 

function m.printf(s,...)
  return io.write(s:format(...)..'\n')
end

function m.debug(s,...)
  if DEBUG then
    print(string.format(s,...))
  end
end

return m
