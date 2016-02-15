local classic = require 'classic'
require 'torch'

znn = classic.module(...)
--znn:class("ConvSlice")
--m.Sum = require 'Sum'
require 'dptycho.znn.ConvSlice'
return znn