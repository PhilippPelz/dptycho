local classic = require 'classic'

local m = classic.module(...)
m:submodule("core")
m:submodule("io")
m:submodule("util")
m:class("MyClass")

return m