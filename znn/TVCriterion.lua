local nn = require 'nn'
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'pprint'
local c, parent = torch.class('znn.TVCriterion', 'nn.Criterion')

function c:__init(tmp,grad,amp,tmp_real,tmp_real2)
   parent.__init(self)
   self.tmp = tmp:squeeze(2)
   self.tmp_real = tmp_real
   self.tmp_real2 = tmp_real2
   self.gradInput = grad
   self.amplitude = amp
end

function c:updateOutput(input, target)
-- c0 = self.amplitude * (U.norm2(del_xf) + U.norm2(del_yf) + U.norm2(del_xb) + U.norm2(del_yb))
  local d = self.tmp
  self.output = 0
  local inp = input:squeeze(2) --squeeze away the probe dimension
  -- pprint(inp)
  -- pprint(d)
  local a = d:dx_bw(inp):normall(1)
  local b = d:dy_bw(inp):normall(1)
  local c = d:dx_fw(inp):normall(1)
  local e = d:dy_fw(inp):normall(1)
  self.output = self.amplitude*(a+b+c+e)
  return self.output
end

function c:updateGradInput(input, target)
  local d_abs = self.tmp_real
  local d = self.tmp
  local mask = self.tmp_real2
  local eps = 1e-10

  self.gradInput:zero()

  d:dx_bw(input)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  self.gradInput:add(d)

  d:dy_bw(input)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  self.gradInput:add(d)

  d:dx_fw(input)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  self.gradInput:add(-1,d)

  d:dy_fw(input)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  self.gradInput:add(-1,d)

  self.gradInput:mul(self.amplitude)
  -- plt:plot(self.gradInput[1][1]:zfloat(),'self.gradInput')
  return self.gradInput
end

return c
