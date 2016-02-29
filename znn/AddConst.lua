require 'nn'
require 'dptycho.znn'
require 'pprint'
local AddConst, parent = torch.class('znn.AddConst', 'nn.Module')

function AddConst:__init(c)
   self.c = c
end

function AddConst:updateOutput(input)
  input:add(self.c)
  -- pprint(input)
  return input
end

function AddConst:updateGradInput(input, gradOutput)
    return gradOutput
end

return AddConst
