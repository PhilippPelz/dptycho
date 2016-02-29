local c, parent = torch.class('znn.WeightedL1Cost','nn.L1Cost')

function c:__init(weight)
   parent.__init(self)
   self.w = weight
end

function c:updateOutput(input)
   local out = parent.updateOutput(self,input)
   return out * self.w
end

function c:updateGradInput(input)
  local gradInput = parent.updateGradInput(self,input)
  return self.gradInput:mul(self.w)
end

return c
