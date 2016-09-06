
require 'pprint'
local plot = require 'dptycho.io.plot'
local plt = plot()
local c, parent = torch.class('znn.AtomRadiusPenalty', 'nn.Criterion')

function c:__init(Znums,r_atom_xy,r_atom_z,weight)
  local nInputPlane = Znums
  local nOutputPlane = Znums
  local filter = torch.ones(nOutputPlane,nInputPlane,r_atom_z,r_atom_xy,r_atom_xy):cuda()
  local bias = torch.zeros(nOutputPlane):cuda()
  local dT, dW, dH = 1,3,3
  local padT, padW, padH = (r_atom_z-1)/2, (r_atom_xy-1)/2, (r_atom_xy-1)/2
  local kT, kW, kH = r_atom_z, r_atom_xy, r_atom_xy

  self.net = nn.Sequential()
  self.net:add(znn.VolumetricConvolutionFixedFilter(nInputPlane, nOutputPlane, kT, kW, kH , dT, dW, dH, padT, padW, padH, filter, bias))
  self.net:add(znn.AddConst(-1))
  self.net:add(nn.Threshold(0,0,true):cuda())
  self.net:add(znn.WeightedL1Cost(weight))
end

function c:updateOutput(input,target)
  return self.net:forward(input)
end

function c:updateGradInput(input, target)
    local gradinput = self.net:backward(input,target)
    return gradinput
end
return c
