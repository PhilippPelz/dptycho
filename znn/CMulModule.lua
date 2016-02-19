local nn = require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.CMulModule', 'nn.Module')

function c:__init(module,ctor)
  parent.__init(self)
  self.module = module
  self.gradInput = ctor()
  self.output = ctor()
end

function c:updateOutput(input)
--  print('before mul forw')
--  print('before mul forw 1')
  self.phase = self.module:forward(input)
--  print('after mul forw')
--  plt:plot(self.phase:zfloat(),'mod_out')
  --   pprint(input)
--  pprint(self.phase)
--  plt:plot(input:zfloat(),'input')
--  print('before mul forw 2')
  self.output:resizeAs(input)
--  print('before mul forw 3')
  self.output:polar(1,self.phase:re())
--  print('before mul forw 4')
  self.output:cmul(input)
  return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:fft(gradOutput):cmul(self.inv_filter):ifft()
    return self.gradInput
end

return c
