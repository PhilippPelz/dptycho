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
  self.output = output
end

function c:updateOutput(input)
--  print('before mul forw')
--  print('before mul forw 1')
  self.weight = self.module:forward(input)
--  print('after mul forw')
--  plt:plot(self.phase:zfloat(),'mod_out')
  --   pprint(input)
--  pprint(self.phase)
--  plt:plot(input:zfloat(),'input')
--  print('before mul forw 2')
  -- self.output:resizeAs(input)
--  print('before mul forw 3')
  self.output:polar(1,self.weight:re())
--  print('before mul forw 4')
  self.output:cmul(input)
  return self.output
end

function c:updateGradInput(input, gradOutput)
    -- self.gradInput:resizeAs(input)
    self.gradInput:polar(1,self.weight:re()):cmul(gradOutput)
    self.module:backward(input,gradOutput:cmul(input))
    return self.gradInput
end

return c
