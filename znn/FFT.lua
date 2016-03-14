local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.FFT', 'nn.Module')

function c:__init()
 parent.__init(self)
 self.gradInput = torch.ZCudaTensor()
 self.output = torch.ZCudaTensor()
end

function c:updateOutput(input)
  -- plt:plot(input[1]:zfloat(),'fft in')
  -- local I = input:clone():abs():pow(2):sum()
  -- print(string.format('integrated intensity: %f',I))
  self.output:resizeAs(input)
  self.output:fftBatched(input)
  -- plt:plot(self.output[1]:zfloat(),'fft out')
  return self.output
end

function c:updateGradInput(input, gradOutput)
    -- plt:plot(gradOutput[1]:zfloat(),'FFT gradOutput 1 ')
    self.gradInput = gradOutput:ifftBatched()
    -- plt:plot(self.gradInput[1]:zfloat(),'FFT gradOutput after')
    return self.gradInput
end

return c
