require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
local classic = require 'classic'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local znn = require "dptycho.znn"
local plt = plot()
local zt = require "ztorch.complex"
local stats = require "dptycho.util.stats"
local simul = require 'dptycho.simulation'

local s = simul.simulator()


-- print(pos:max())
-- pprint(pos)
-- print(pos)

-- plt:scatter_positions2(pos:float())
local E = 300e3
local N = 256
local d = 2.0
local alpha_rad = 3e-3
local C3_um = 500
local defocus_nm = 1.8e3
local C5_mm = 800
local tx = 0
local ty = 0
local Nedge = 10
local plot = true
local binning = 8
local dose = 3e6
local probe_size = {384,384}
local support = znn.SupportMask(probe_size,probe_size[#probe_size]*0.2)
-- local probe = torch.ZCudaTensor(table.unpack(probe_size)):fillRe(1):fillIm(0)
-- pprint(probe)
-- plt:plotReIm(probe:zfloat())
-- probe = support:forward(probe)

local pot = s:load_potential('/home/philipp/vol26.h5')
local pos = s:get_positions_raster(300,500-N)
pos = pos:int() + 1

-- local probe = s:focused_probe(E, N, d, alpha_rad, defocus_nm, C3_um , C5_mm, tx ,ty , Nedge , plot)
-- local probe = s:random_probe(N)
local probe = s:random_probe3(N,0.10,0.2,0.10)
plt:plot(probe:zfloat())
local I = s:dp_multislice(pos,probe, N, binning, E, dose)
-- local I = s:dp_projected(pos,probe, N, binning, E, dose)

for i=40,45 do
  plt:plot(I[i]:float(),'I')
end
-- local ms_out = s:exitwaves_multislice(pos,probe, N, E,dose)
-- local pa_out = s:exitwaves_projected(pos,probe, N, E,dose)
--
-- for i=1,pos:size(1) do
--   local ms = ms_out[i]:arg():float()
--   local pa = pa_out[i]:arg():float()
--   -- pprint(ms)
--   -- pprint(pa)
--   plt:plotcompare({ms,pa},{'ms exitwave','pa_exitwave'})
--   local diff = ms:clone():add(-1,pa)
--   plt:plot(diff:float(),'diff')
-- end
u.printf('P_norm                       1: %g',probe:normall(2)^2)
local f = hdf5.open('/home/philipp/simtest1e7.h5','w')
f:write('/scan_info/positions_int',pos)
f:write('/data_unshift',I:float():sqrt())
f:write('/pr',probe:re():float())
f:write('/pi',probe:im():float())
f:write('/or',s.T_proj:re():float())
f:write('/oi',s.T_proj:im():float())
