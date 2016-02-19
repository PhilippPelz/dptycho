local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.FFT', 'nn.Module')

function c:__init()
--  parent.__init(self)
end

function c:updateOutput(input)
   local output = input:clone():fft()
   return output
end

function c:updateGradInput(input, gradOutput)
    local gradInput = gradOutput:clone():ifft()
    return gradInput
end

return c