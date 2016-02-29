local nn = require 'nn'
local WSECriterion, parent = torch.class('znn.WSECriterion', 'nn.Criterion')

function WSECriterion:__init(y)
   parent.__init(self)
   self.y = y
end

function WSECriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.WSECriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.y
   )
   self.output = self.output_tensor[1]
   return self.output
end

function WSECriterion:updateGradInput(input, target)
   input.THNN.WSECriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.y
   )
   return self.gradInput
end

return WSECriterion
