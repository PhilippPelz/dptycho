local classic = require 'classic'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local plot = require 'dptycho.io.plot'
local plt = plot()

local c = classic.class(...)

function c:_init()
end

function c.static.build_model(params)
  local data = {}
  local ctor = torch.ZCudaTensor

  if params.is_simulation then
    data = params.data
  else
    data = {}
  end

  params.Vsize = params.Vsize or 2.5
  params.L = data.nslices

  pprint(data)

  -- normalize positions
  local max = torch.max(data.positions,1):totable()[1]
  pprint(max)
  -- local min = torch.min(data.positions,1)
  -- local pos = data.positions - min:expandAs(data.positions)
  -- local size = (max):totable()
  params.Nx = max[1] + params.M
  params.Ny = max[2] + params.M

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

  local nets = {}
  local pos = data.positions

  local out_tmp = torch.ZCudaTensor.new({data.Znums:size(1),params.M,params.M})

  for i=1,pos:size(1) do
    local slice = {{},{pos[i][1]+1,pos[i][1]+params.M},{pos[i][2]+1,pos[i][2]+params.M}}
    local net = nn.Sequential()
    net:add(znn.Source(ctor,data.probe))
    for l=1,params.L do

      -- pprint(slice)
      local deltas = data.real_deltas[l][slice]
      local grads = data.gradWeights[l][slice]
      -- pprint(deltas)
      -- plt:plot(deltas[1]:float(),string.format('deltas position %d layer %d',i,l))

      local arm = nn.Sequential()
      -- in: nil,     out: [Z,M,M]
      arm:add(znn.ConvParams(data.Wsize,data.atompot,data.inv_atompot,deltas,grads,out_tmp))
      -- in: [Z,M,M]  out: [M,M]
      arm:add(znn.Sum(1,ctor))

      -- in: [R,M,M]  out: [R,M,M]
      net:add(znn.CMulModule(arm,ctor,out_tmp[1]))
      -- in: [R,M,M]  out: [R,M,M]
      net:add(znn.ConvFFT2D(data.prop,data.bwprop))
    end
    net:add(znn.FFT())
    net:add(znn.ComplexAbs())
    net:add(znn.Square())
    nets[#nets+1] = net
  end
  return nets
end

return c
