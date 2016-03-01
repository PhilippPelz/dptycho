local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.ComplexAbs', 'nn.Module')

function c:__init(tmp)
  parent.__init(self)
  self.gradInput = tmp
end

function c:updateOutput(input)
  --  print('ComplexAbs')
  --  local r = input:abs()
  --  print('ComplexAbs2')
  self.output = input:abs()
  -- plt:plot(out[1]:float(),'abs out')
  return self.output
end

function c:updateGradInput(input, gradOutput)
    -- self.gradInput:resizeAs(input)
    print('in ComplexAbs:updateGradInput')
    pprint(gradOutput)
    self.gradInput:polar(gradOutput,0)
    return self.gradInput
end

return c
