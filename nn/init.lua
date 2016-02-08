local classic = require 'classic'
require 'torch'
local nn = require 'nn'
local m = classic.module(...)

--m:class("MyClass")
m.Sum = require 'Sum'
return m