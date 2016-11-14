require 'hdf5'
require 'torch'
local ztorch = require 'ztorch'
require 'zcutorch'
require 'pprint'
local classic = require 'classic'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local optim = require "optim"
local znn = require "dptycho.znn"
local plt = plot()
local zt = require "ztorch.complex"
local stats = require "dptycho.util.stats"
local simul = require 'dptycho.simulation'
local ptycho = require 'dptycho.core.ptycho'
local ptychocore = require 'dptycho.core'
require 'image'

local s = simul.simulator()
local pot = s:load_potential('/home/philipp/drop/Public/4V6X_200.h5')

-- pprint(pot)

local E = 300e3
local physics = u.physics(E)
local dz = 1e-9
local v_proj = pot:sum(1)[1]
-- pot = pot:zcuda()

local T_proj = torch.ZCudaTensor(v_proj:size()):polar(v_proj:im():mul(-physics.sigma*dz):exp(),v_proj:re():mul(physics.sigma*dz))

local absmax = T_proj:abs():max()

local zf = T_proj:zfloat()
local rgb = u.complex2rgb(zf)
rgb:div(rgb:max())

-- image.save('rgb.png',rgb)
-- rgb = image.load('rgb.png')
-- pprint(rgb)
local cx = u.rgb2complex(rgb)
cx:mul(absmax)
local absmax1 = cx:abs():max()
-- pprint(cx)
-- print(rgb:max())
print(absmax,absmax1)


-- plt:plot(v_proj)
-- plt:plot(T_proj)
-- plt:plot(cx)
