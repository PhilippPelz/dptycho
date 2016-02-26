local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.ConvFFT2D', 'nn.Module')

function c:__init(filter,inv_filter)
--  parent.__init(self)
  self.filter = filter
  self.inv_filter = inv_filter
end

function c:updateOutput(input)
--  plt:plot(input:zfloat(),'in')
  -- self.output:resizeAs(input):copy(input)
--  plt:plot(input:zfloat(),'in')
  input:fft(input)
  --   plt:plot(self.output:zfloat(),'in fft')
  input:cmul(self.filter)
  --   plt:plot(self.output:zfloat(),'in cmul')
  input:ifft()
  --   plt:plot(self.output:zfloat(),'in ifft')
  return input
end

function c:updateGradInput(input, gradOutput)
    return gradOutput:fft():cmul(self.inv_filter):ifft()
end

return c
