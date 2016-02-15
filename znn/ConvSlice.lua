local nn = require 'nn'
require 'dptycho.znn'
local c, parent = torch.class('znn.ConvSlice', 'nn.Module')
local plot = require 'io.plot'
local plt = plot()

function c:__init(sigma,filters,inv_filters,weights,gradWeights)
--   parent.__init(self)
   self.gradInput = torch.ZCudaTensor()
   self.output = torch.ZCudaTensor()
   self.filters = filters
   self.inv_filters = inv_filters
   self.weights = weights
   self.gradWeights = gradWeights
   self.sigma = sigma
end

function c:updateOutput(input)
   self.output:resizeAs(input):zero()
   local accum = torch.ZCudaTensor(input:size()):zero()
   local tmp = torch.ZCudaTensor(input:size()):zero()

   for Z, W in pairs(self.weights) do
      local conv = tmp:copyRe(W):fft():cmul(self.filters[Z]):ifft()
--      plt:plot(conv:zfloat(),'conv_'..Z)
      accum:add(conv)
      plt:plot(accum:zfloat(),'accum_'..Z)
      tmp:zero()
   end
   pprint(accum)
   local accum = accum:mul(self.sigma):re() 
   plt:plot(accum:float(),'accum_total')
   self.output:polar(0,accum)
   plt:plot(self.output:zfloat(), 'output')
   return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:fft(gradOutput):cmul(self.inv_filter):ifft()
    return self.gradInput
end

return c
