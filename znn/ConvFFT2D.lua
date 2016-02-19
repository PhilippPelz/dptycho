local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.ConvFFT2D', 'nn.Module')

function c:__init(filter,inv_filter)
--  parent.__init(self)
  self.filter = filter
  self.inv_filter = inv_filter
  self.gradInput = torch.ZCudaTensor()
  self.output = torch.ZCudaTensor()
end

function c:updateOutput(input)
--  plt:plot(input:zfloat(),'in')
  self.output:resizeAs(input):copy(input)
--  plt:plot(input:zfloat(),'in')
  self.output:fft()
  --   plt:plot(self.output:zfloat(),'in fft')
  self.output:cmul(self.filter)
  --   plt:plot(self.output:zfloat(),'in cmul')
  self.output:ifft()
  --   plt:plot(self.output:zfloat(),'in ifft')
  return self.output  
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:fft(gradOutput):cmul(self.inv_filter):ifft()
    return self.gradInput
end

return c