local nn = require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.Source', 'nn.Module')

function c:__init(ctor,init)
  parent.__init(self)
  self.weight = init or ctor()
end

function c:updateOutput(input)
  return self.weight:clone()
end

function c:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

return c
