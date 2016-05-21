local nn = require 'nn'
local TruncatedPoissonLikelihood, parent = torch.class('znn.TruncatedPoissonLikelihood', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and truncated Wirtinger gradient. arXiv:1603.04746 [physics] (2016).

function TruncatedPoissonLikelihood:__init(mask, buffer, No, Np)
   parent.__init(self)
   self.factor = buffer
   self.mask = mask
   self.No = No
end

function TruncatedPoissonLikelihood:updateOutput(input, target)
   input.THNN.TruncatedPoissonLikelihood_updateOutput(target,self.mask,self.output)
   return self.output[1]
end

function TruncatedPoissonLikelihood:updateGradInput(input, target)
  if not self.factor then
    self.factor = torch.CudaTensor(input:size())
  end

  for i = 1, self.No do
    self.gradInput:

  input.THNN.TruncatedPoissonLikelihood_GradientFactor(target,self.factor,self.mask)
   input.THNN.WSECriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.y
   )
   -- TODO
   return self.gradInput
end

return TruncatedPoissonLikelihood
