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

local pot = s:load_potential('/home/philipp/vol5.h5')
local pos = s:get_positions_raster(100,768-385)
pos = pos:int() + 1
-- print(pos:max())
-- pprint(pos)
-- print(pos)

-- plt:scatter_positions2(pos:float())
E = 300e3
N = 384
d = 2.5
alpha_rad = 3e-3
defocus_nm = 9e3
C3_um = 0
C5_mm = 0
tx = 0
ty = 0
Nedge = 20
plot = false
binning = 8

local probe_size = {384,384}
local support = znn.SupportMask(probe_size,probe_size[#probe_size]*0.2)
-- local probe = torch.ZCudaTensor(table.unpack(probe_size)):fillRe(1):fillIm(0)
-- pprint(probe)
-- plt:plotReIm(probe:zfloat())
-- probe = support:forward(probe)



-- local probe = s:focused_probe(E, N, d, alpha_rad, defocus_nm, C3_um , C5_mm, tx ,ty , Nedge , plot)
local probe = s:random_probe(N)
plt:plotReIm(probe:zfloat())
-- local I = s:dp_multislice(pos,probe, N, binning, E, 1e10)
local I = s:dp_projected(pos,probe, N, E, 1e10)
