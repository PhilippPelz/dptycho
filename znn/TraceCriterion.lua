local nn = require 'nn'

local c, parent = torch.class('znn.TracePenalty', 'nn.Module')

function c:__init()
  parent.__init(self)
end

function c:updateOutput(R)
  self.output:resizeAs(R)
  self.output:abs(R):pow(2):sumall()
  return self.output
end

function c:updateGradInput(input, target)
    self.gradInput:resizeAs(input)

    return self.gradInput
end

return c
