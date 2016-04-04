local classic = require 'classic'
require 'torch'
require 'ztorch'
require 'zcutorch'

local m = classic.module(...)
m:class("base_engine")
m:class("RAAR_engine")

return m
