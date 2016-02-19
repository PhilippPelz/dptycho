local nn = require 'nn'
require 'dptycho.znn'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.CMulModule', 'nn.Module')

function c:__init(module)
  parent.__init(self)
  self.module = module
end

function c:updateOutput(input)
--  print('before mul forw')
  self.phase = self.module:forward(input)
--  print('after mul forw')
--  plt:plot(self.phase:zfloat(),'mod_out')
  --   pprint(input)
  --   pprint(self.mod_out)
--  plt:plot(input:zfloat(),'input')
  return self.output:resizeAs(input):polar(1,self.phase:re()):cmul(input)
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:fft(gradOutput):cmul(self.inv_filter):ifft()
    return self.gradInput
end

return c
