require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.ConvParams', 'nn.Module')

function c:__init(size,filter,inv_filter,W,dW, output)
  parent.__init(self)
  self.weight = W
--  pprint(W)
--  for i=1,self.weight:size(1) do
--    plt:plot(self.weight[i]:float(),'weight_'..i)
--  end
  self.gradWeight = dW
  self.filter = filter
--  plt:plot(self.filter[1]:zfloat(),'filter_'..1)
  self.inv_filter = inv_filter
  self.gradInput = torch.ZCudaTensor()
  self.output = output
end

function c:forward(input)
   return self:updateOutput(input)
end

function c:updateOutput(input)
--   self.output:resizeAs(self.weight)
--  pprint(self.output)
  self.output:zero()
  self.output:copyRe(self.weight)
--  plt:plot(self.weight[1]:float(),'weight_'..1)
--  plt:plot(self.output[1]:re():float(),'outre_'..1)
  self.output:fftBatched()
--  plt:plot(self.output[1]:zfloat(),'out_'..1)
  self.output:cmul(self.filter)
  self.output:ifftBatched()
--  print('ConvParams out')
--  pprint(self.output)
  return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradWeight:copy(gradOutput:expandAs(self.weight)):fftBatched()
    self.gradWeight:cmul(self.inv_filter):ifftBatched()
    return gradOutput
end

return c
