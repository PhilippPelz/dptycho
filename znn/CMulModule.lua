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
  self.update = true
end
function c:immutable()
  self.update = false
end

function c:mutable()
  self.update = true
end
function c:updateOutput(input)
--  print('before mul forw')
--  print('before mul forw 1')
  self.weight = self.module:forward(input)
--  print('after mul forw')
--  plt:plot(self.phase:zfloat(),'mod_out')
    -- pprint(input)
--  pprint(self.phase)
 -- plt:plot(input[1]:zfloat(),'CMulModule input')
--  print('before mul forw 2')
  self.output:resizeAs(input)
--  print('before mul forw 3')
  self.output:polar(1,self.weight)
  self.output:expandAs(input)
--  print('before mul forw 4')
  -- self.output:cmul(input)
  self.output:cmul(input)
  -- plt:plot(self.output[1]:zfloat(),'CMulModule out')
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
  self.output:cmul(gradOutput)
  if self.update then
    self.module:backward(input,self.output:im():mul(2))
  end
end

return c
