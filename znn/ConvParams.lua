require 'nn'
require 'dptycho.znn'
local c, parent = torch.class('znn.ConvFFT2D', 'nn.Module')

function c:__init(W,dW,filter,inv_filter)
  parent.__init(self)
  self.weight = W
  self.gradWeight = dW
  self.filter = filter
  self.inv_filter = inv_filter
  self.gradInput = torch.ZCudaTensor()
  self.output = torch.ZCudaTensor()
end

function c:forward()
   return self:updateOutput()
end

function c:updateOutput()
   local output =  self.weight:clone():fftBatched():cmul(self.filter):ifftBatched()
   return output:sum(1)
end

function c:updateGradInput(input, gradOutput)
    local gradInput = gradOutput:clone():fft()
    gradInput:expandAs(self.weight):cmul(self.inv_filter):ifftBatched()
    return gradOutput
end

return c
