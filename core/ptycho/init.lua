local classic = require 'classic'
require 'torch'
require 'ztorch'
require 'zcutorch'

local m = classic.module(...)
m:submodule("params")
m:class("base_engine")
m:class("base_engine_shifted")
m:class("RAAR_engine")

return m
