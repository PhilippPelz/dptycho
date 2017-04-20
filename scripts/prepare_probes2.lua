require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
-- require 'hypero'
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
local ptycho = require 'dptycho.core.ptycho'
local ptychocore = require 'dptycho.core'

local probe_type =5
local s = simul.simulator()
local N = 256
local E = 300e3
local d = 1.0
if probe_type == 1 then

  local alpha_rad = 6e-3
  local C3_um = 500
  local defocus_nm = 1.2e3
  local C5_mm = 800
  local tx = 0
  local ty = 0
  local Nedge = 5
  local plotit = true

  probe = s:random_fzp(N,500,6)
  -- plt:plot(probe:zfloat():fftshift(),'FZP')
  probe:fftshift()
  local prop = ptychocore.propagators.fresnel(N,d*1e-10,700e-9,u.physics(E).lambda)
  prop:fftshift()
  -- for i=1,10 do
  probe:cmul(prop)
  -- plt:plot(probe:zfloat(),'defocused FZP')
  probe:view(1,probe:size(1),probe:size(2)):ifftBatched()
    -- plt:plot(probe:zfloat(),'defocused FZP')
    -- probe:view(1,probe:size(1),probe:size(2)):fftBatched()
  -- end
  probe:fftshift()
  plt:plot(probe:zfloat(),'defocused RFZP')
elseif probe_type == 2 then
  -- for i = 0,15 do
    -- print(200+25*i)
    probe = s:fzp(N,250)
    -- plt:plot(probe:zfloat():fftshift(),'FZP')
    probe:fftshift()

    local prop = ptychocore.propagators.fresnel(N,d*1e-10,400e-9,u.physics(E).lambda)
    prop:fftshift()
    -- for i=1,10 do
    probe:cmul(prop)
    -- plt:plot(probe:zfloat(),'defocused FZP')
    probe:view(1,probe:size(1),probe:size(2)):ifftBatched()
    -- plt:plot(probe:zfloat(),'defocused FZP')
      -- probe:view(1,probe:size(1),probe:size(2)):fftBatched()
    -- end
    probe:fftshift()
    plt:plot(probe:zfloat(),'defocused FZP')
  -- end
    -- plt:plotcx(probe[1])
elseif probe_type == 3 then
  probe = s:random_probe2(N,0.2,0.1,0.0)
  -- probe = s:random_probe2(N,0.11,0.25,0.17)
  plt:plot(probe:zfloat(),'band limited random')
elseif probe_type == 4 then
  probe = s:random_probe2(N,0.11,0.4,0.3)
  plt:plot(probe:zfloat(),'cone band limited random')
elseif probe_type == 5 then
  local alpha_rad = 9e-3
  local C3_um = 2
  local defocus_nm = 0.5e3
  local C5_mm = 0
  local tx = 0
  local ty = 0
  local Nedge = 5
  local plotit = true
  local dp = 14*5e-6
  probe = s:focused_probe(E, N, d, alpha_rad, defocus_nm,dp, C3_um , C5_mm, tx ,ty , Nedge , plot)
  plt:plot(probe:zfloat(),'defocus')
end
local f = hdf5.open('/home/philipp/drop/Public/probe_def7.h5','w')
print('1')
f:write('/pr',probe:re():float())
print('2')
f:write('/pi',probe:im():float())
f:close()
