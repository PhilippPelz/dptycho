local nn = require 'nn'

local c, parent = torch.class('znn.TracePenalty', 'nn.Criterion')

function c:__init(weight)
  parent.__init(self)
  self.weight = weight
end

function c:updateOutput(R)
  self.output:resizeAs(R)
  self.output:abs(R):pow(2):sumall()
  return self.output * self.weight[1]
end

function c:updateGradInput(input)
    self.gradInput:resizeAs(input)
    -- TODO
    return self.gradInput
end

return c
