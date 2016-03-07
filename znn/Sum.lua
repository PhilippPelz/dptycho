require 'nn'
require 'dptycho.znn'
require 'pprint'
local u = require "dptycho.util"
local Sum, parent = torch.class('znn.Sum', 'nn.Module')

function Sum:__init(dimension,size)
  --  parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.view = u.copytable(size)
   self.expand = u.copytable(size)
   for i=1,#size-2 do
     self.view[i] = 1
   end
  --  for i=#size,#size-1,-1 do
  --    self.expand[i] = 1
  --  end

end

function Sum:updateOutput(input)
  input:sum(self.dimension)
  return input
end

function Sum:updateGradInput(input, gradOutput)
  -- print('sum view')
  -- pprint(self.view)
  -- print('sum expand')
  -- pprint(self.expand)
  self.gradInput = gradOutput:view(unpack(self.view))
  -- pprint(self.gradInput)
  self.gradInput:expand(unpack(self.expand))
  -- pprint(self.gradInput)
  -- print('in Sum:updateGradInput')
  -- pprint(self.gradInput)
  return self.gradInput
end

return Sum
