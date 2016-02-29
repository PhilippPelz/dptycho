local classic = require 'classic'
require "cutorch"
require "nn"
require "cunn"
require "dptycho.znn.THZNN"

znn = classic.module(...)
--znn:class("ConvSlice")
--m.Sum = require 'Sum'

require 'dptycho.znn.SupportMask'
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

return znn
