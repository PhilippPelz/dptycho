local classic = require 'classic'

local m = classic.module(...)
m:submodule("core")
m:submodule("io")
m:submodule("util")
m:submodule("znn")

return m