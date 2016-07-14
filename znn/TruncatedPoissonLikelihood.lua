local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'pprint'
local TruncatedPoissonLikelihood, parent = torch.class('znn.TruncatedPoissonLikelihood', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and truncated Wirtinger gradient. arXiv:1603.04746 [physics] (2016).
-- 1.Chen, Y. & Candes, E. J. Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems. arXiv:1505.05114 [cs, math, stat] (2015).

function TruncatedPoissonLikelihood:__init(a_h, a_lb, a_ub, gradInput, mask, buffer1, buffer2, z_buffer_real, K, No, Np, M, Nx, Ny)
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
   self.output = torch.CudaTensor(1):fill(0)
end

function TruncatedPoissonLikelihood:updateOutput(z, I_target)
  -- plt:plot(z[1][1][1]:abs():float():log(),'in_psi abs')
  for k = 1, self.K do
    for o = 1, self.No do
      z[k][o]:fftBatched(z[k][o])
    end
  end
  self.I_model = self.z_real:normZ(z):sum(2):sum(3)
  -- plt:plot(I_model[1][1][1]:float():log(),'I_model')
  self.I_model.THNN.TruncatedPoissonLikelihood_updateOutput(self.I_model:cdata(),I_target:cdata(),self.mask:cdata(),self.output:cdata())
  return self.output[1]
end

function TruncatedPoissonLikelihood:calculateXsi(z,I_target)
  -- for k = 1, self.K do
  --   for o = 1, self.No do
  --     self.gradInput[k][o]:fftBatched(z[k][o])
  --   end
  -- end

  -- K x 1 x 1 x M x M -- far-field intensity
  -- local I_model = self.gradInput:norm():sum(2):sum(3)
  -- local I_model = self.z_real:normZ(z):sum(2):sum(3)

  -- plt:plot(self.I_model[1][1][1]:float():log(),'self.I_model')

  -- ||c - |Az|^2||_1 / (K*M*M)
  local K_t = self.rhs:add(I_target,-1,self.I_model):norm(1)/I_target:nElement()
  -- u.printf('L_1 mean: %g',L1_mean)
  -- sqrt(n) * |a| / ( ||a|| ||z||) with ||a|| = n because DFT
  self.rhs:sqrt(self.I_model):div(z:normall(2)/self.Nx/self.Ny/self.No)
  -- calculate E_1
  local E_1 = torch.ge(self.rhs,self.a_lb)
  self.lhs:le(self.rhs,self.a_ub)
  E_1:cmul(self.lhs)
  -- plt:plot(E_1[1][1][1]:float(),'E1')

  self.rhs:mul(K_t*self.a_h)

  -- |c - |Az|^2|
  self.lhs:add(I_target,-1,self.I_model):abs()
  -- plt:plotcompare({self.lhs[1]:float(),self.rhs[1][1][1]:float()},{'self.lhs','self.rhs'})
  self.valid_gradients:le(self.lhs,self.rhs)
  -- plt:plot(self.valid_gradients[1]:float(),'E2')
  self.valid_gradients:cmul(E_1)
  -- plt:plot(self.valid_gradients[1]:float(),'E1 && E2')

  -- plt:plot(self.I_model[1][1][1]:float():log(),'self.I_model')
  -- plt:plot(I_target[1]:float():log(),'self.I_target')
  self.I_model.THNN.TruncatedPoissonLikelihood_GradientFactor(self.I_model:cdata(),I_target:cdata(),self.mask:cdata())

  -- z * 2 * fm * (1 - I_target/I_model)
  -- plt:plot(self.I_model[1][1][1]:float(),'self.GradientFactor')
  -- plt:plot(self.gradInput[1][1][1]:zfloat():abs():log(),'self.gradInput 1')
  self.gradInput:cmul(self.I_model:expandAs(self.gradInput))
  -- plt:plot(self.gradInput[1][1][1]:zfloat():abs():log(),'self.gradInput 2')

  for k = 1, self.K do
    for o = 1, self.No do
      self.gradInput[k][o]:ifftBatched()
    end
  end

  -- plt:plot(self.gradInput[1][1][1]:zfloat(),'self.gradInput 3')

  -- pprint(self.lhs)
  -- pprint(self.rhs)
  -- plt:plotcompare({self.lhs[1]:float():log() ,self.rhs[1][1][1]:float():log() },{'lhs','rhs'})
  -- plt:plotcompare({self.lhs[2]:float():log() ,self.rhs[2][1][1]:float():log() },{'lhs2','rhs2'})
  -- plt:plotcompare({self.lhs[3]:float():log() ,self.rhs[3][1][1]:float():log() },{'lhs3','rhs3'})
  return self.gradInput
end

function TruncatedPoissonLikelihood:updateGradInput(z, I_target)
  self.gradInput = self:calculateXsi(z,I_target)

  self.valid_gradients = self.valid_gradients:view(self.K,1,1,self.M,self.M):expandAs(self.gradInput)

  -- pprint(valid_gradients)
  local sum = self.valid_gradients:sum()
  -- u.printf('valid gradients: %d, %2.2f percent', sum, sum/valid_gradients:nElement()*100.0)

  self.gradInput:cmul(self.valid_gradients)

  -- plt:plot(self.gradInput[1][1][1]:zfloat():abs():log(),'self.gradInput 111')
  -- plt:plot(self.gradInput[2][1][1]:zfloat():abs():log(),'self.gradInput 211')
  -- plt:plot(self.gradInput[3][1][1]:zfloat():abs():log(),'self.gradInput 311')

  return self.gradInput, (sum/self.valid_gradients:nElement()*100.0)
end

function TruncatedPoissonLikelihood:__call__(input, target)
--    self.output = self:forward(input, target)
--    self.gradInput = self:backward(input, target)
--    return self.output, self.gradInput
end
return TruncatedPoissonLikelihood
