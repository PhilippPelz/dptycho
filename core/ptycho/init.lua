local classic = require 'classic'
require 'torch'
require 'ztorch'
require 'zcutorch'

local m = classic.module(...)
<<<<<<< HEAD
m:submodule("params")
m:class("base_engine")
m:class("base_engine_shifted")
=======
m:class("base_engine")
>>>>>>> 5400af4241e2b79a156cfca7ef79ec71554c2453
m:class("RAAR_engine")

return m