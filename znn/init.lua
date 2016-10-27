local classic = require 'classic'
require "cutorch"
require "nn"
require "cunn"
require "dptycho.znn.THZNN"

znn = classic.module(...)
--znn:class("ConvSlice")
--m.Sum = require 'Sum'

require 'dptycho.znn.SupportMask'
require 'dptycho.znn.Threshold'
require 'dptycho.znn.MultiCriterionVariableWeights'
require 'dptycho.znn.SqrtInPlace'
require 'dptycho.znn.Sqrt'
require 'dptycho.znn.WeightedL1Cost'
require 'dptycho.znn.TracePenalty'
require 'dptycho.znn.WeightedLinearCriterion'
require 'dptycho.znn.WSECriterion'
require 'dptycho.znn.AtomRadiusPenalty'
require 'dptycho.znn.VolumetricConvolutionFixedFilter'
require 'dptycho.znn.Source'
require 'dptycho.znn.Select'
require 'dptycho.znn.Sum'
require 'dptycho.znn.AddConst'
require 'dptycho.znn.Square'
require 'dptycho.znn.ComplexAbs'
require 'dptycho.znn.ConvSlice'
require 'dptycho.znn.CMulModule'
require 'dptycho.znn.ConvParams'
require 'dptycho.znn.ConvFFT2D'
require 'dptycho.znn.FFT'
require 'dptycho.znn.TruncatedPoissonLikelihood'
require 'dptycho.znn.EuclideanLoss'
require 'dptycho.znn.PoissonLikelihood'
require 'dptycho.znn.SpatialSmoothnessCriterion'
require 'dptycho.znn.TVCriterion'

-- collapses all dimensions from 1 up to ndim-2 into a single dimension
-- good to perform batched ffts
function view_3D(t)
  local t_size = znn.tsize(t)
  local cumsize = t_size:narrow(1,1,t_size:nElement()-2):cumprod()[-1]
  return t:view(cumsize,t_size[-2],t_size[-1])
end

function znn.tsize(t)
  return torch.IntTensor((#t):totable())
end

local zmetatable = torch.getmetatable('torch.ZCudaTensor')
rawset( zmetatable, 'view_3D', view_3D)

return znn
