local nn = require 'nn'

local c, parent = torch.class('znn.WeightedLinearCriterion', 'nn.Criterion')

function c:__init(y)
  parent.__init(self)
  self.y = y
end

function c:updateOutput(input,target)
  local res = input:clone():sub(target):mul(y)
  return res
end

function c:updateGradInput(input, target)
    return input:clone():mul(y)
end

return c
