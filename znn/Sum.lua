require 'nn'
require 'dptycho.znn'
require 'pprint'
local u = require "dptycho.util"
local Sum, parent = torch.class('znn.Sum', 'nn.Module')

function Sum:__init(dimension,size)
  --  parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.rep = u.copytable(size)
   for i=1,#size do
     size[i] = 1
   end
   self.rep[self.dimension] = size[self.dimension]
end

function Sum:updateOutput(input)
  input:sum(self.dimension)
  return input
end

function Sum:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:view(unpack(self.rep))
  print('in Sum:updateGradInput')
  pprint(self.gradInput)
  return self.gradInput
end

return Sum
