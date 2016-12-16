local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'pprint'
local  PoissonLikelihood, parent = torch.class('znn.PoissonLikelihood', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and   Wirtinger gradient. arXiv:1603.04746 [physics] (2016).
-- 1.Chen, Y. & Candes, E. J. Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems. arXiv:1505.05114 [cs, math, stat] (2015).

function  PoissonLikelihood:__init(a_h, a_lb, a_ub, gradInput, mask, buffer1, buffer2, z_buffer_real, K, No, Np, M, Nx, Ny, diagnostics)
   parent.__init(self)

   self.gradInput = gradInput
   self.lhs = buffer1
   self.rhs = buffer2
   self.valid_gradients = buffer1
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
   self.output = torch.CudaTensor(1):fill(0)
end

function  PoissonLikelihood:updateOutput(z, I_target)
  -- plt:plot(z[1][1][1]:abs():float():log(),'in_psi abs')
  z:view_3D():fftBatched()
  self.I_model = self.z_real:normZ(z)
  -- sum over probe and object modes
  self.I_model:sum(2):sum(3)
  -- plt:plot(I_model[1][1][1]:float():log(),'I_model')

  -- local I_m = self.I_model:clone():log():cmul(I_target)
  -- local fac = self.I_model:clone():add(-1,I_m):cmul(self.mask)
  -- local L = fac:sum()
  -- print(L)

  self.I_model.THNN. TruncatedPoissonLikelihood_updateOutput(self.I_model:cdata(),I_target:cdata(),self.mask:cdata(),self.output:cdata())
  -- print(self.output[1])
  return self.output[1]
end

function  PoissonLikelihood:calculateXsi(z,I_target)
  -- ||c - |Az|^2||_1 / (K*M*M)
  -- local K_t = self.rhs:add(I_target,-1,self.I_model):norm(1)/(I_target:nElement())
  -- local z_2norm = z:normall(2)/self.M/self.Np/self.No
  -- -- sqrt(n) * |a| / ( ||a|| ||z||) with ||a|| = n because DFT
  -- self.rhs:sqrt(self.I_model):div(z_2norm)
  -- -- u.printf('I_model 11 max,min: %g, %g',self.I_model:max(),self.I_model:min())
  --
  -- -- calculate E_1, condition for denominator
  -- local E_1 = torch.ge(self.rhs,self.a_lb)
  -- self.lhs:le(self.rhs,self.a_ub)
  -- E_1:cmul(self.lhs)
  -- if self.diagnostics then
  --   -- u.printf('Percent filfilling condition     1: %2.2f',E_1:sum()/E_1:nElement()*100.0)
  --   -- local rhs = self.rhs:float()
  --   -- print(rhs:max(),rhs:min())
  --   -- plt:hist(self.rhs:float():view(self.rhs:nElement()),'rhs, cond 1')
  --   -- u.printf('K_t = %g',K_t)
  --   -- u.printf('a_h = %g',self.a_h)
  --   -- u.printf('K_t * a_h = %g',K_t*self.a_h)
  -- end
  -- -- plt:plot(E_1[1][1][1]:float(),'E1')
  --
  -- self.rhs:mul(K_t*self.a_h)
  --
  -- -- |c - |Az|^2|
  -- self.lhs:add(I_target,-1,self.I_model):abs()
  --
  -- -- local ab = self.I_model[1][1][1]:clone():fftshift():float():log()
  -- -- local ak = I_target[1]:clone():fftshift():float():log()
  -- -- local DI = ab:clone():add(-1,ak)
  -- -- plt:plotcompare({ab,ak},{'I_model','I_target'})
  -- -- plt:plot(DI:float():log(),'DI')
  --
  -- -- E_2, condition for numerator
  -- self.valid_gradients:le(self.lhs,self.rhs)
  -- if self.diagnostics then
  --   -- u.printf('Percent filfilling condition     2: %2.2f',self.valid_gradients:sum()/self.valid_gradients:nElement()*100.0)
  --   -- plt:hist(self.rhs:float():view(self.rhs:nElement()),'rhs')
  --   -- plt:hist(self.lhs:float():view(self.lhs:nElement()),'lhs')
  -- end
  --
  -- -- E_1 && E_2
  -- self.valid_gradients:cmul(E_1)
  -- if self.diagnostics then
  --   -- u.printf('Percent filfilling condition 1 & 2: %2.2f',self.valid_gradients:sum()/self.valid_gradients:nElement()*100.0)
  -- end

  -- in-place calculation of I_model <-- fm * (1 - I_target/I_model)
  self.I_model.THNN. TruncatedPoissonLikelihood_GradientFactor(self.I_model:cdata(),I_target:cdata(),self.mask:cdata())
  -- self.I_model.THNN. TruncatedPoissonLikelihood_GradientFactor(I_target:cdata(),self.I_model:cdata(),self.mask:cdata())

  -- gradinput is set to z, thats why only in-place multiply
  -- z * 2 * fm * (1 - I_target/I_model)
  self.gradInput:cmul(self.I_model:expandAs(self.gradInput))
  self.gradInput:view_3D():ifftBatched()
  return self.gradInput
end

function  PoissonLikelihood:updateGradInput(z, I_target)
  self.gradInput = self:calculateXsi(z,I_target)
  -- local vg_expanded = self.valid_gradients:view(self.K,1,1,self.M,self.M):expandAs(self.gradInput)
  -- self.gradInput:cmul(vg_expanded)
  return self.gradInput, (self.valid_gradients:sum()/self.valid_gradients:nElement()*100.0)
end

function  PoissonLikelihood:__call__(input, target)
--    self.output = self:forward(input, target)
--    self.gradInput = self:backward(input, target)
--    return self.output, self.gradInput
end
return  PoissonLikelihood
