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
local engine = require 'dptycho.core.ptycho.DM_engine'

local path = '/home/philipp/experiments/2016-04-11/scan2/'
local file = 'scan2_data_final.h5'
local probe_file = 'probe.h5'

local M = 1536
local NP = 3
-- local FWHM = 100
-- local x = torch.repeatTensor(torch.linspace(-M/2,M/2,M),M,1)
-- -- pprint(x)
-- local y = x:clone():t()
-- local r2 = (x:pow(2) + y:pow(2))
-- local gauss = r2:div(-2*(FWHM/2.35482)^2):exp()
-- plt:plot(gauss)

local f = hdf5.open(path..file,'r')

local a = f:read('/a'):all():cuda()
-- [{{1},{},{}}]
local fmask = f:read('/fm'):all():cuda()
local pos = f:read('/scan_info/positions'):all()
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
f:close()

local x = pos[{{},{1}}]:clone()
pos[{{},{1}}]:copy(pos[{{},{2}}])
pos[{{},{2}}]:copy(x)

local x = dpos[{{},{1}}]:clone()
dpos[{{},{1}}]:copy(dpos[{{},{2}}])
dpos[{{},{2}}]:copy(x)
-- dpos[{1,1}] = 5

o_r = nil
o_i = nil
local f = hdf5.open(path..'probe.h5','r')
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor().new({1,NP,M,M})
probe[1][1]:copyIm(pi):copyRe(pr)
probe[1][1]:copyIm(pi):copyRe(pr):mul(1e4)
f:close()
-- plt:plot(probe[1][1]:zfloat())
pr = nil
pi = nil
collectgarbage()

-- u.linear_schedule(3,50,1e-9,0)
-- u.linear_schedule(3,50,500,1500),
-- frames

DEBUG = false

par = {
  Np = NP,
  No = 1,
  probe = nil,
  plot_every = 1,
  plot_start = 1,
  show_plots = false,
  beta = 1,
  fourier_relax_factor = 15e-2,
  position_refinement_start = 10,
  position_refinement_every = 3,
  position_refinement_max_disp = 3,
  probe_update_start = 2,
  probe_support = nil,
  object_inertia = 1e-7,
  probe_inertia = 1e-9,
  P_Q_iterations = 10,
  copy_solution = false,
  background_correction_start = 100,
  save_interval = 5,
  save_path = path..'/hyperscan_flipped_10/'
}

par.probe_lowpass_fwhm = u.linear_schedule(4,250,110,110)
par.object_highpass_fwhm = u.linear_schedule(6,250,400,400)
par.fm_support_radius = function(it) return nil end
par.probe_regularization_amplitude = function(it) return nil end
par.fm_mask_radius = u.linear_schedule(3,250,150,150)

par.pos = pos
par.dpos = dpos
-- par.dpos_solution = dpos_solution
-- par.solution = solution
par.a = a
par.fmask = fmask
par.probe = probe


local ngin = engine(par)
-- ngin:generate_data('/home/philipp/drop/Public/moon_subpix2.h5')
ngin:iterate(250)
