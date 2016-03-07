require 'nn'
require 'dptycho.znn'
require 'pprint'
local Square, parent = torch.class('znn.Square','nn.Module')

function Square:__init()
   parent.__init(self)
   self.gradInput = torch.CudaTensor()
   self.output = torch.CudaTensor()
end

function Square:updateOutput(input)
   return input:pow(2)
end

function Square:updateGradInput(input, gradOutput)
  -- print('in Square:updateGradInput')
  -- pprint(gradOutput)
  self.gradInput = gradOutput:cmul(input):mul(2)
  -- pprint(gradOutput)
  return self.gradInput
end
