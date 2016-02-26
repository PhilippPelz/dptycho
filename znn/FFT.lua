local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.FFT', 'nn.Module')

function c:__init()
--  parent.__init(self)
end

function c:updateOutput(input)
  -- plt:plot(input[1]:zfloat(),'fft in')
  input:fftBatched()
  -- plt:plot(input[1]:zfloat(),'fft out')
  return input
end

function c:updateGradInput(input, gradOutput)
    return gradOutput:ifft()
end

return c
