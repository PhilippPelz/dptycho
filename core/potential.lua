local classic = require 'classic'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local ffi = require 'ffi'
local plot = require 'io.plot'
local plt = plot()
local c = classic.class(...)

ffi.cdef([[
double vzatom(int Z, double radius);
double vzatomLUT(int Z, double rsq);
double vatom(int Z, double radius);
]])
C = ffi.load('slicelib')

function c.static.get_atomic_potential(Z,size,dx)
  local x = torch.repeatTensor(torch.linspace(-size/2*dx,size/2*dx,size),size,1)
  -- print(x)
  local y = x:clone():t()
  -- print(y)
  local r = (x:pow(2) + y:pow(2)):sqrt()
  plt:plot(r,'r^2')
  local rt = r:view(r:nElement()):totable()
  local v = {}
  for i, ri in ipairs(rt) do
    -- print(ri)
    v[i] = C.vzatom(Z,ri)
    print(v[i])
  end
  local vt = torch.Tensor(v):resizeAs(r)
  plt:plot(vt,'v')
end

return c
