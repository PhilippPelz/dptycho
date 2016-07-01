local nn = require 'nn'
local TruncatedPoissonLikelihood, parent = torch.class('znn.TruncatedPoissonLikelihood', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and truncated Wirtinger gradient. arXiv:1603.04746 [physics] (2016).

function TruncatedPoissonLikelihood:__init(a_h, gradInput, mask, buffer1, buffer2, par, K, No, Np)
  print('twf init')
   parent.__init(self)
   self.gradInput = gradInput
   self.lhs = buffer1
   self.rhs = buffer2
   self.mask = mask
   self.par = par
   self.No = No
   self.Np = Np
   self.K = K
   self.a_h = a_h
end

function TruncatedPoissonLikelihood:updateOutput(in_psi, I_target)
   in_psi.THNN.TruncatedPoissonLikelihood_updateOutput(I_target,self.mask,self.output)
   return self.output[1]
end

function TruncatedPoissonLikelihood:calculateXsi(in_psi,I_target)
  for k = 1, self.K do
    for o = 1, self.No do
      self.gradInput[k][o]:fftBatched(in_psi[k][o])
    end
  end

  -- K x 1 x 1 x M x M -- far-field intensity
  local I_model = self.gradInput:norm():sum(2):sum(3)

  self.lhs:div(I_model,in_psi:normall(2))
  self.rhs:add(I_target,-1,I_model):cmul(self.lhs):mul(self.a_h/I_target:nElement())

  self.lhs:add(I_target,-1,I_model):abs()

  I_model.THNN.TruncatedPoissonLikelihood_GradientFactor(I_target,self.mask)

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
  self.gradInput:cmul(valid_gradients:expandAs(self.gradInput))

  return self.gradInput
end
function TruncatedPoissonLikelihood:__call__(input, target)
  print('here')
--    self.output = self:forward(input, target)
--    self.gradInput = self:backward(input, target)
--    return self.output, self.gradInput
end
return TruncatedPoissonLikelihood
