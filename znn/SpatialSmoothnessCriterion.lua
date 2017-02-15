local nn = require 'nn'
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'pprint'
local c, parent = torch.class('znn.SpatialSmoothnessCriterion', 'nn.Criterion')

function c:__init(tmp,grad,params)
   parent.__init(self)
   self.tmp = tmp:squeeze(2)
   self.gradInput = grad
   self.amplitude = params.amplitude
end

function c:updateOutput(input, target)
-- c0 = self.amplitude * (U.norm2(del_xf) + U.norm2(del_yf) + U.norm2(del_xb) + U.norm2(del_yb))
  local d = self.tmp
  self.output = 0
  local inp = input:squeeze(2) --squeeze away the probe dimension
  -- pprint(inp)
  -- pprint(d)
  local a = d:dx_bw(inp):normall(2)
  local b = d:dy_bw(inp):normall(2)
  local c1 = d:dx_fw(inp):normall(2)
  local e = d:dy_fw(inp):normall(2)
  self.output = self.amplitude*(a+b+c1+e)
  return self.output
end

function c:updateGradInput(input, target)
  -- plt:plot(input[1][1]:zfloat(),'input')
  self.gradInput:zero()
  local d = self.tmp
  local inp = input:squeeze(2) --squeeze away the probe dimension
  d:zero()
  d:dx_bw(inp)
  -- plt:plot(d[1]:zfloat(),'dx_bw')
  self.gradInput:add(d)
  d:zero()
  d:dy_bw(inp)
  -- plt:plot(d[1]:zfloat(),'dy_bw')
  self.gradInput:add(d)
  d:zero()
  d:dx_fw(inp)
  -- plt:plot(d[1]:zfloat(),'dx_fw')
  self.gradInput:add(-1,d)
  d:zero()
  d:dy_fw(inp)
  -- plt:plot(d[1]:zfloat(),'dy_fw')
  self.gradInput:add(-1,d)

  self.gradInput:mul(self.amplitude)
  -- plt:plot(self.gradInput[1][1]:zfloat(),'self.gradInput')
  return self.gradInput
end

return c
