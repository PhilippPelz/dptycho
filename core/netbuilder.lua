local classic = require 'classic'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()

local c = classic.class(...)

function c:_init(params)
  if params.is_simulation then
    self.data = params.data
  else
    self.data = {}
  end

  self.p = params

  params.Vsize = params.Vsize or 2.0

  local r_atom = math.floor(params.Vsize / params.dx)
  r_atom = ( r_atom % 2 == 0 ) and ( r_atom - 1 ) or r_atom
  print(string.format('r_atom = %d',r_atom))
  params.r_atom_xy = r_atom
  params.r_atom_z = 1

  params.L = self.data.nslices
  params.R = self.data.probe:size(1)
  params.Z = self.data.Znums:size(1)
  params.K = self.data.positions:size(1)

  -- pprint(data)

  -- normalize positions
  local max = torch.max(self.data.positions,1):totable()[1]
  local min = torch.min(self.data.positions,1):totable()[1]
  local s1, s2 = max[1] - min[1], max[2]-min[2]
  -- print('maximum ' )
  -- pprint(max)

  -- local pos = data.positions - min:expandAs(data.positions)
  -- local size = (max):totable()

  -- TODO unused until now, and wrong
  params.Nx = max[1] + params.M
  params.Ny = max[2] + params.M

  -- self.data.deltas = self.data.deltas[{{},{},{min[1],max[1]+params.M},{min[2],max[2]+params.M}}]
  -- self.data.gradWeights = self.data.gradWeights[{{},{},{min[1],max[1]+params.M},{min[2],max[2]+params.M}}]

  local E0 = params.E;
  local m0 = 9.1093822
  local c0  = 2.9979246
  local e  = 1.6021766
  local h  = 6.6260696
  local pi = 3.1415927

  local _gamma = 1.0 + E0 * e / m0 / c0 / c0 * 1e-4
  params.lam =  h / math.sqrt ( 2.0 * m0 * e ) * 1e-9 / math.sqrt ( E0 * ( 1.0 + E0 * e / 2.0 / m0 / c0 / c0 * 1e-4 ) ) -- electron wavelength (m), see De Graef p. 92
  params.lamA = params.lam * 1e10
  params.sigma =  2.0 * pi * _gamma * params.lam * m0 * e / h / h * 1e18 -- interaction constant (1/(Vm))
  self.probe_size = self.data.probe:size():totable()
  self.mask = znn.SupportMask(self.probe_size,self.probe_size[#self.probe_size]/2)
  self.out_tmp = torch.ZCudaTensor.new({math.max(self.p.Z,self.p.R),self.p.M,self.p.M})

  -- calculate coverage
  self:get_coverage()
end

function c:get_coverage()
  local size = self.data.deltas:size():totable()
  local view = self.data.deltas:size():totable()
  -- pprint(view)
  table.remove(size,1)
  table.remove(size,1)
  view[1] = 1
  view[2] = 1
  -- pprint(view)
  self.coverage = torch.CudaTensor(unpack(size)):fill(1e-4)

  local ones = torch.CudaTensor():ones(self.p.M,self.p.M)
  for i = 1, self.data.positions:size(1) do
    local pos = self.data.positions[i]
    local slice = {{pos[1]+1,pos[1]+self.p.M},{pos[2]+1,pos[2]+self.p.M}}
    self.coverage[slice]:add(ones)
    -- if i % 2 == 0 then
    --   plt:plot(self.coverage:float(),'coverage')
    -- end
  end
  self.coverage:pow(-1)
  local mask = torch.gt(self.coverage,1)
  self.coverage:maskedFill(mask,0)
  -- pprint(self.coverage)
  self.coverage = self.coverage:view(unpack(view))
  self.coverage = self.coverage:expand(self.data.deltas:size())
  -- pprint(self.coverage)
  -- plt:plot(self.coverage:float(),'coverage')
end

function c:get_tower(i)
  local ctor = torch.ZCudaTensor
  local pos = self.data.positions[i]
  -- self.out_tmp:zero()
  -- local I = data.probe:clone():abs():pow(2):sum()
  -- print(string.format('integrated intensity probe: %f',I))
  local net = nn.Sequential()
  -- in: nil,     out: [R,M,M]
  local source = znn.Source(ctor,self.data.probe)
  source:immutable()
  net:add(source)
  -- in: [R,M,M]  out: [R,M,M]
  net:add(self.mask)



  for l=1,self.p.L do
    local slice = {{},{l},{pos[1]+1,pos[1]+self.p.M},{pos[2]+1,pos[2]+self.p.M}}
    -- print('slice')
    -- pprint(slice)
    local deltas = self.data.deltas[slice]
    local grads = self.data.gradWeights[slice]
    -- local cover = self.coverage[slice]
    local deltas_size = deltas:size():totable()

    local arm = nn.Sequential()
    -- in: nil,     out: f[Z,M,M]
    arm:add(znn.ConvParams(self.data.Wsize,self.data.atompot,self.data.inv_atompot,deltas,grads,self.out_tmp:narrow(1,1,self.p.Z)))
    -- in: f[Z,M,M]  out: f[Z,M,M]
    arm:add(znn.Threshold(0,0,true):cuda())
    -- in: f[Z,M,M]  out: f[1,M,M]
    arm:add(znn.Sum(1,deltas_size))


    -- in: c[R,M,M]  out: c[R,M,M]
    net:add(znn.CMulModule(arm,ctor,self.out_tmp:narrow(1,1,self.p.R)))
    -- in: [R,M,M]  out: [R,M,M]
    net:add(znn.ConvFFT2D(self.data.prop,self.data.bwprop))
  end
  -- propagate to fourier space
  net:add(znn.FFT())
  -- in: cx[R,M,M]  out: f[R,M,M]
  net:add(znn.ComplexAbs(self.out_tmp[1]))
  -- in: [R,M,M]  out: [R,M,M]
  net:add(znn.Square())
  -- sum the mode intensities
  -- in: [R,M,M]  out: [1,M,M]
  net:add(znn.Sum(1,self.probe_size))
  -- make it 2 - dimensional
  -- in: [1,M,M]  out: [M,M]
  net:add(znn.Select(1,1))
  net:add(znn.Sqrt())

  -- u.printf("built a %d layer neural network",#net.modules)
  local slice = {{},{},{pos[1]+1,pos[1]+self.p.M},{pos[2]+1,pos[2]+self.p.M}}
  return net, self.data.deltas[slice], self.data.gradWeights[slice]
end

return c
