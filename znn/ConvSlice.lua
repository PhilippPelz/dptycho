local nn = require 'nn'
require 'dptycho.znn'
local c, parent = torch.class('znn.ConvSlice', 'nn.Module')
local plot = require 'dptycho.io.plot'
local plt = plot()

function c:__init(filters,inv_filters,weights,gradWeights)
--   parent.__init(self)
   self.gradInput = torch.ZCudaTensor()
   self.output = torch.ZCudaTensor()
   self.filters = filters
   self.inv_filters = inv_filters
   self.weights = weights
   self.gradWeights = gradWeights
end

function c:updateOutput(input)
--    plt:plot(input:zfloat(),'in')
   self.output:resizeAs(input):zero()
   local accum = torch.ZCudaTensor(input:size()):zero()
   local tmp = torch.ZCudaTensor(input:size()):zero()

   for Z, W in pairs(self.weights) do
      local conv = tmp:copyRe(W):fft():cmul(self.filters[Z]):ifft()
--      plt:plot(conv:zfloat(),'conv_'..Z)
      accum:add(conv)
--      plt:plot(accum:zfloat(),'accum_'..Z)
      tmp:zero()
   end   
   self.output:polar(1,accum:re())
   self.sum = self.output:clone()
--   plt:plot(self.sum:zfloat(),'sum')
   self.output:cmul(input)
--   plt:plot(self.output:zfloat(),'out')
   return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)        
    self.gradInput:copy(self.sum):cmul(gradOutput)
    local gradSum = input:clone():cmul(gradOutput)
    local tmp = torch.ZCudaTensor(input:size()):zero()
    for Z, W in pairs(self.gradWeights) do
      local deconv = tmp:copy(gradSum):fft():cmul(self.inv_filters[Z]):ifft()
--      plt:plot(conv:zfloat(),'conv_'..Z)
      self.gradWeights[Z] = deconv:re()
--      plt:plot(accum:zfloat(),'accum_'..Z)
      tmp:zero()
   end  
    return self.gradInput
end

return c
