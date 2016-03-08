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
    -- print('in ComplexAbs:updateGradInput')
    -- pprint(gradOutput)
    self.gradInput:zero()
    self.gradInput:copyRe(gradOutput):cmul(input:conj())
    -- pprint(self.gradInput)
    -- plt:plot(self.gradInput:zfloat(),'ComplexAbs gradInput')
    return self.gradInput
end

return c
