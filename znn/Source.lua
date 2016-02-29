local nn = require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.Source', 'nn.Module')

function c:__init(ctor,init)
  parent.__init(self)
  self.weight = init or ctor()
  self.update = true
end

function c:immutable()
  self.update = false
end

function c:mutable()
  self.update = true
end

function c:updateOutput(input)
  return self.weight:clone()
end

function c:updateGradInput(input, gradOutput)
  if self.update then
    self.gradInput = gradOutput
  end
  return gradOutput
end

return c
