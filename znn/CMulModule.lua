local nn = require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.CMulModule', 'nn.Module')

function c:__init(module,ctor,output)
  parent.__init(self)
  self.module = module
  self.gradInput = output
  self.output = torch.ZCudaTensor()
  self.gradWeight = torch.ZCudaTensor()
  self.update = true
end
function c:immutable()
  self.update = false
end

function c:mutable()
  self.update = true
end
function c:updateOutput(input)
  -- print('in CMulModule updateOutput')
  self.weight = self.module:forward(input)
  -- pprint(self.weight)
  -- pprint(input)
  -- plt:plot(self.weight[1]:float(),'weight')
  -- plt:plot(input[1]:zfloat(),'input')
  self.output:resizeAs(self.weight)
  self.output:polar(1,self.weight)
  -- plt:plot(self.output[1]:zfloat(),'self.output 1')
  self.output:expandAs(input)
  -- plt:plot(self.output[1]:zfloat(),'self.output 2')
  self.output=self.output:cmul(input)
  -- plt:plot(self.output[1]:zfloat(),'self.output 3')
  return self.output
end

function c:updateGradInput(input, gradOutput)
    -- self.gradInput:resizeAs(input)
    self.gradInput:polar(1,self.weight):cmul(gradOutput)
    return self.gradInput
end

function c:accGradParameters(input, gradOutput, scale)
  -- print('in ConvParams:updateGradInput')
  -- pprint(input)
  -- pprint(gradOutput)
  self.gradWeight:resizeAs(input)
  self.gradWeight:copy(input):conj():cmul(gradOutput)
  self.gradWeight = self.gradWeight:im():mul(-2)
  -- plt:plot(self.gradWeight[1]:float(),'CMulModule gradWeight')
  if self.update then
    self.module:backward(input,self.gradWeight)
  end
end

return c
