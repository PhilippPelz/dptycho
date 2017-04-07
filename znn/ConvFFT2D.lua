local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'dptycho.io.plot'
local plt = plot()
local c, parent = torch.class('znn.ConvFFT2D', 'nn.Module')

function c:__init(filter,inv_filter)
  parent.__init(self)
  self.filter = filter
  self.inv_filter = inv_filter
  self.gradInput = torch.ZCudaTensor()
  self.output = torch.ZCudaTensor()
end

function c:updateOutput(input)
  -- plt:plot(input[1]:zfloat(),'ConvFFT2D in')
  self.output:resizeAs(input)
--  plt:plot(input:zfloat(),'in')
  -- pprint(input)
  -- pprint(self.output)
  self.output:fftBatched(input)
  -- plt:plot(self.output[1]:zfloat(),'ConvFFT2D 2')
  self.output:cmul(self.filter)
  -- plt:plot(self.output[1]:zfloat(),'ConvFFT2D 3')
  self.output:ifftBatched()
  -- local I = input:clone():abs():pow(2):sum()
  -- print(string.format('integrated intensity: %f',I))
  -- pprint(input)
  -- plt:plot(input[1]:zfloat(),'out after prop')
  -- plt:plot(self.output[1]:zfloat(),'ConvFFT2D out')
  return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:fftBatched():cmul(self.inv_filter):ifftBatched()
    return self.gradInput
end

return c
