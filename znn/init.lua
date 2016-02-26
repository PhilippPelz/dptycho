local classic = require 'classic'
require 'torch'

znn = classic.module(...)
--znn:class("ConvSlice")
--m.Sum = require 'Sum'
require 'dptycho.znn.Source'
require 'dptycho.znn.Sum'
require 'dptycho.znn.Square'
require 'dptycho.znn.ComplexAbs'
require 'dptycho.znn.ConvSlice'
require 'dptycho.znn.CMulModule'
require 'dptycho.znn.ConvParams'
require 'dptycho.znn.ConvFFT2D'
require 'dptycho.znn.FFT'
return znn
