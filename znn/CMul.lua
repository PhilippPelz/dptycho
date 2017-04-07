local nn = require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'dptycho.io.plot'
local plt = plot()
local c, parent = torch.class('znn.CMul', 'nn.Module')

function c:__init(weight,gradWeight)
  parent.__init(self)
  self.gradInput = torch.ZCudaTensor()
  self.output = torch.ZCudaTensor()
  self.update = true
  self.weight = weight
  self.gradWeight = gradWeight
end

function c:immutable()
  self.update = false
end

function c:mutable()
  self.update = true
end
function c:updateOutput(input)
  -- pprint(input)
  -- pprint(self.weight)
  return input:cmul(self.weight)
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    -- pprint(gradOutput)
    -- plt:plot(gradOutput:zfloat(),'CMulModule gradOutput')
    -- print('in CMulModule:updateGradInput')
    self.gradInput:copy(self.weight):cmul(gradOutput)
    -- plt:plot(self.gradInput[1]:zfloat(),'CMulModule gradInput')
    return self.gradInput
end

function c:accGradParameters(input, gradOutput, scale)
  if self.update then
    self.gradWeight:copy(input):cmul(gradOutput)
  end
  return self.gradWeight
end

return c
