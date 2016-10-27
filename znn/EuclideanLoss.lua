local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'pprint'
local EuclideanLoss, parent = torch.class('znn.EuclideanLoss', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and truncated Wirtinger gradient. arXiv:1603.04746 [physics] (2016).
-- 1.Chen, Y. & Candes, E. J. Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems. arXiv:1505.05114 [cs, math, stat] (2015).

function EuclideanLoss:__init(a_h, a_lb, a_ub, gradInput, mask, buffer1, buffer2, z_buffer_real, K, No, Np, M, Nx, Ny, diagnostics, do_truncate)
   parent.__init(self)

   self.gradInput = gradInput
   self.lhs = buffer1
   self.rhs = buffer2
   self.valid_gradients = buffer1:zero()
   self.z_real = z_buffer_real
   self.mask = mask
   self.No = No
   self.Np = Np
   self.Nx = Nx
   self.Ny = Ny
   self.M = M
   self.K = K
   self.a_h = a_h
   self.a_lb = a_lb
   self.a_ub = a_ub
   self.diagnostics = diagnostics
   self.do_truncate = do_truncate
   self.output = torch.CudaTensor(1):fill(0)
end

function EuclideanLoss:updateOutput(z, a_target)
  -- plt:plot(z[1][1][1]:abs():float():log(),'in_psi abs')
  z:view_3D():fftBatched()
  self.a_model = self.z_real:normZ(z)
  -- sum over probe and object modes
  self.a_model:sum(2):sum(3)
  self.a_model:sqrt()
  -- plt:plot(a_model[1][1][1]:float():log(),'a_model')

  -- local I_m = self.a_model:clone():log():cmul(I_target)
  -- local fac = self.a_model:clone():add(-1,I_m):cmul(self.mask)
  -- local L = fac:sum()
  -- print(L)

  self.a_model.THNN.EuclideanLoss_updateOutput(self.a_model:cdata(),a_target:cdata(),self.mask:cdata(),self.output:cdata())
  -- print(self.output[1])
  return self.output[1]
end

function EuclideanLoss:calculateXsi(z,a_target)

  -- in-place calculation of a_model <-- fm * (1 - a_target/a_model)
  self.a_model.THNN.EuclideanLoss_GradientFactor(self.a_model:cdata(),a_target:cdata(),self.mask:cdata())
  -- gradinput is set to z, thats why only in-place multiply
  -- z * 2 * fm * (1 - I_target/a_model)
  local zeroGrad = torch.ge(a_target,1e-3)
  self.gradInput:cmul(self.a_model:expandAs(self.gradInput))
  -- self.gradInput:cmul(zeroGrad:expandAs(self.gradInput))
  self.gradInput:view_3D():ifftBatched()
  return self.gradInput
end

function EuclideanLoss:updateGradInput(z, a_target)
  self.gradInput = self:calculateXsi(z,a_target)
  return self.gradInput, self.valid_gradients:sum()
end

function EuclideanLoss:__call__(input, target)
--    self.output = self:forward(input, target)
--    self.gradInput = self:backward(input, target)
--    return self.output, self.gradInput
end
return EuclideanLoss
