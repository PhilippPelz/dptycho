require 'nn'
require 'dptycho.znn'
local Select, parent = torch.class('znn.Select', 'nn.Module')

function Select:__init(dimension,index)
   parent.__init(self)
   self.dimension = dimension
   self.index = index
end

function Select:updateOutput(input)
   local output = input:select(self.dimension,self.index);
  --  self.output:resizeAs(output)
   return output--self.output:copy(output)
end

function Select:updateGradInput(input, gradOutput)
  --  self.gradInput:resizeAs(input)
  --  self.gradInput:zero()
  --  self.gradInput:select(self.dimension,self.index):copy(gradOutput)
   return gradOutput:repeatTensor(1,1,1)
end
