require 'nn'
require 'dptycho.znn'
require 'pprint'
local AddConst, parent = torch.class('znn.AddConst', 'nn.Module')

function AddConst:__init(c)
   self.c = c
   self.gradInput = torch.CudaTensor()
   self.output = torch.CudaTensor()
end

function AddConst:updateOutput(input)
  self.output = input:add(self.c)
  -- print('addconst input')
  -- pprint(input)
  return self.output
end

function AddConst:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

return AddConst
