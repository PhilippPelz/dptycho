local classic = require 'classic'

local m = classic.module(...)
m:submodule("ptycho")
m:submodule("propagators")
m:class("netbuilder")

return m
