local nn = require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.Source', 'nn.Module')

function c:__init(ctor,init)
  parent.__init(self)
  self.weight = init or ctor()
  self.gradInput = torch.ZCudaTensor()
  self.output = torch.ZCudaTensor()
  self.update = true
  return self
end

function c:immutable()
  self.update = false
end

function c:mutable()
  self.update = true
end

function c:updateOutput(input)
  self.output = self.weight:clone()
  return self.output
end

function c:updateGradInput(input, gradOutput)
  if self.update then
    -- print('Source:updateGradInput')
    -- pprint(self.gradInput)
    -- pprint(gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end
  return self.gradInput
end

return c
