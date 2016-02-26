local cudnn = require 'cudnn'
require 'pprint'
local v, parent
   = torch.class('znn.VolumetricConvolutionFixedFilter','cudnn.VolumetricConvolution')

function v:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH ,filter,bias)
  -- parent:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
  -- print('here')
  -- pprint(filter)
  -- pprint(bias)
  self.weight = filter
  self.bias = bias
  -- print(torch.typename(self.weight))
  -- print(torch.typename(self.bias))
  self.gradWeight = torch.CudaTensor()
  self.gradBias = torch.CudaTensor()

  dT = dT or 1
  dW = dW or 1
  dH = dH or 1

  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.kT = kT
  self.kW = kW
  self.kH = kH
  self.dT = dT
  self.dW = dW
  self.dH = dH
  self.padT = padT or 0
  self.padW = padW or self.padT
  self.padH = padH or self.padW

  self.gradInput = torch.CudaTensor()
  self.output = torch.CudaTensor()

  -- self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
  -- self.bias = torch.Tensor(nOutputPlane)
  -- self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
  -- self.gradBias = torch.Tensor(nOutputPlane)
  -- temporary buffers for unfolding (CUDA)
  self.finput = torch.CudaTensor()
  self.fgradInput = torch.CudaTensor()
  -- self:reset()
end

function v:accGradParameters(input, gradOutput, scale)

end
return v
