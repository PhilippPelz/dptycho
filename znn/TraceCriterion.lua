local nn = require 'nn'

local c, parent = torch.class('znn.TraceCriterion', 'nn.Criterion')

function c:__init()
  parent.__init(self)
end

function c:updateOutput(R)
  self.output:resizeAs(R)
  self.output:abs(R):pow(2):sumall()
  return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    
    return self.gradInput
end

return c
