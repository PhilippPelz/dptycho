local c, parent = torch.class('znn.MultiCriterionVariableWeights', 'nn.Criterion')

function c:__init()
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
end

function c:add(criterion, weight)
   assert(criterion, 'no criterion provided')
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

function c:updateOutput(input, target)
   self.output = 0
   for i=1,#self.criterions do
      self.output = self.output + self.weights[i][1]*self.criterions[i]:updateOutput(input, target)
   end
   return self.output
end

function c:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   for i=1,#self.criterions do
      nn.utils.recursiveAdd(self.gradInput, self.weights[i][1], self.criterions[i]:updateGradInput(input, target))
   end
   return self.gradInput
end

function c:type(type)
   for i,criterion in ipairs(self.criterions) do
      criterion:type(type)
   end
   return parent.type(self, type)
end

return c
