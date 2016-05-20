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

local path = '/home/philipp/experiments/2016-05-17/scan1/'
local file = 'scan1_data_final_custom.h5'
local probe_file = 'probe.h5'

-- local M = 1536
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
-- dpos[{1,1}] = 5

o_r = nil
o_i = nil
-- local pr = f:read('/pr'):all():cuda()
-- local pi = f:read('/pi'):all():cuda()
-- local probe = torch.ZCudaTensor().new({1,3,M,M})
-- probe[1][1]:copyIm(pi):copyRe(pr)
-- plt:plot(probe[1][1]:zfloat())
pr = nil
pi = nil
collectgarbage()

-- u.linear_schedule(3,50,1e-9,0)
-- u.linear_schedule(3,50,500,1500),
-- frames

DEBUG = false

par = {
  Np = 3,
  No = 1,
  probe = nil,
  plot_every = 1,
  plot_start = 1,
  show_plots = false,
  beta = 1,
  fourier_relax_factor = 15e-2,
  position_refinement_start = 6,
  position_refinement_every = 2,
  position_refinement_max_disp = 3,
  probe_update_start = 2,
  probe_support = 3/8.0,
  fm_support_radius = function(it) return nil end,
  probe_regularization_amplitude = function(it) return nil end,
  object_highpass_fwhm = function(it) return nil end,
  object_inertia = 1e-9,
  probe_inertia = 3e-9,
  P_Q_iterations = 10,
  copy_solution = false,
  background_correction_start = 100,
  save_interval = 5,
  save_path = path..'/hyperscan_custom1/'
}

par.probe_lowpass_fwhm = u.linear_schedule(8,250,220,220)


par.pos = pos
par.dpos = dpos
-- par.dpos_solution = dpos_solution
-- par.solution = solution
par.a = a
par.fmask = fmask
-- par.probe = probe


local ngin = engine(par)
-- ngin:generate_data('/home/philipp/drop/Public/moon_subpix2.h5')
ngin:iterate(250)
