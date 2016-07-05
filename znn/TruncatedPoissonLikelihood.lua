local nn = require 'nn'
local u = require 'dptycho.util'
require 'pprint'
local TruncatedPoissonLikelihood, parent = torch.class('znn.TruncatedPoissonLikelihood', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and truncated Wirtinger gradient. arXiv:1603.04746 [physics] (2016).

function TruncatedPoissonLikelihood:__init(a_h, gradInput, mask, buffer1, buffer2, z_buffer_real, par, K, No, Np, M)
   parent.__init(self)
   self.gradInput = gradInput
   self.lhs = buffer1
   self.rhs = buffer2
   self.z_real = z_buffer_real
   self.mask = mask
   self.par = par
   self.No = No
   self.Np = Np
   self.M = M
   self.K = K
   self.a_h = a_h
   self.output = torch.CudaTensor(1):fill(0)
end

function TruncatedPoissonLikelihood:updateOutput(in_psi, I_target)
  for k = 1, self.K do
    for o = 1, self.No do
      in_psi[k][o]:fftBatched(in_psi[k][o])
    end
  end
  local I_model = self.z_real:normZ(in_psi):sum(2):sum(3)
  I_model.THNN.TruncatedPoissonLikelihood_updateOutput(I_model:cdata(),I_target:cdata(),self.mask:cdata(),self.output:cdata())
  return self.output[1]
end

function TruncatedPoissonLikelihood:calculateXsi(in_psi,I_target)
  -- for k = 1, self.K do
  --   for o = 1, self.No do
  --     self.gradInput[k][o]:fftBatched(in_psi[k][o])
  --   end
  -- end

  -- K x 1 x 1 x M x M -- far-field intensity
  -- local I_model = self.gradInput:norm():sum(2):sum(3)
  local I_model = self.z_real:normZ(in_psi):sum(2):sum(3)

  self.lhs:div(I_model,in_psi:normall(2))
  self.rhs:add(I_target,-1,I_model):cmul(self.lhs):mul(self.a_h/I_target:nElement())

  self.lhs:add(I_target,-1,I_model):abs()

  I_model.THNN.TruncatedPoissonLikelihood_GradientFactor(I_model:cdata(),I_target:cdata(),self.mask:cdata())

  self.gradInput:cmul(I_model:expandAs(self.gradInput))

  for k = 1, self.K do
    for o = 1, self.No do
      self.gradInput[k][o]:ifftBatched()
    end
  end

  return self.gradInput
end

function TruncatedPoissonLikelihood:updateGradInput(in_psi, I_target)
  if not self.lhs then
    self.lhs = torch.CudaTensor(I_target:size())
  end
  if not self.rhs then
    self.rhs = torch.CudaTensor(I_target:size())
  end

  self.gradInput = self:calculateXsi(in_psi,I_target)

  local valid_gradients = torch.le(self.lhs,self.rhs)

  valid_gradients = valid_gradients:view(self.K,1,1,self.M,self.M):expandAs(self.gradInput)
  self.gradInput:cmul(valid_gradients)

  return self.gradInput
end

function TruncatedPoissonLikelihood:__call__(input, target)
--    self.output = self:forward(input, target)
--    self.gradInput = self:backward(input, target)
--    return self.output, self.gradInput
end
return TruncatedPoissonLikelihood
