require 'nn'
require 'dptycho.znn'
require 'pprint'
local plot = require 'io.plot'
local plt = plot()
local c, parent = torch.class('znn.ConvParams', 'nn.Module')

function c:__init(size,filter,inv_filter,W,dW, output)
  parent.__init(self)
  -- float [Z,1,M,M]
  self.weight = W
  -- float [Z,1,M,M]
  self.gradWeight = dW

  -- if self.gradWeight:dim() == 2 then
  --   local size = {1,self.gradWeight:size(1),self.gradWeight:size(2)}
  --   self.gradWeight:view(unpack(size))
  -- end
  -- if self.weight:dim() == 2 then
  --   local size = {1,self.weight:size(1),self.weight:size(2)}
  --   self.weight:view(unpack(size))
  -- end
  -- pprint(self.gradWeight)
  -- pprint(self.weight)
  self.filter = filter
--  plt:plot(self.filter[1]:zfloat(),'filter_'..1)
  self.inv_filter = inv_filter
  self.gradInput = torch.ZCudaTensor()
  self.tmp = output
end

function c:forward(input)
   return self:updateOutput(input)
end

function c:updateOutput(input)
--   self.output:resizeAs(self.weight)
--  pprint(self.output)
  self.tmp:zero()
  self.tmp:copyRe(self.weight)
--  plt:plot(self.weight[1]:float(),'weight_'..1)
--  plt:plot(self.output[1]:re():float(),'outre_'..1)
  self.tmp:fftBatched()
--  plt:plot(self.output[1]:zfloat(),'out_'..1)
  self.tmp:cmul(self.filter)
  self.tmp:ifftBatched()
--  print('ConvParams out')
--  pprint(self.output)
  self.output = self.tmp:re()
  return self.output
end

function c:accGradParameters(input, gradOutput)
    assert(torch.type(gradOutput) == 'torch.CudaTensor')

    -- gradOutput:expandAs(self.weight)
    -- print('in ConvParams:updateGradInput')
    -- pprint(gradOutput)
    -- pprint(self.inv_filter)
    -- pprint(self.gradWeight)
    self.tmp:zero()
    self.tmp:copyRe(gradOutput)
    self.tmp:fftBatched()
    self.tmp:cmul(self.filter)
    self.tmp:ifftBatched()
    self.gradWeight:add(self.tmp:re())

    -- pprint(self.gradWeight)
    -- pprint(self.gradWeight)
    -- plt:plot(self.gradWeight[1]:float(),'gradWeight')
    return self.gradWeight
end

return c
