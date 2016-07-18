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

local path = '/home/philipp/drop/Public/'
local file = 'moon.h5'

local ptycho = require 'dptycho.core.ptycho'

local par = {}
par.i = 50
par.DM_smooth_amplitude = 1
par.probe_change_start = 1

local f = hdf5.open(path..file,'r')

local a = f:read('/data_unshift'):all():cuda()
local fmask = a:clone():fill(1)
local pos = f:read('/scan_info/positions_int'):all():int():add(1)
-- local dpos = pos:clone():float():zero()
local dpos = f:read('/scan_info/dpos'):all():float()
local dpos_solution  = dpos:clone()
dpos:zero()
local o_r = f:read('/or'):all():cuda()
local o_i = f:read('/oi'):all():cuda()
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr):mul(1e6)
local object_solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)

o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

DEBUG = false

par = ptycho.params.DEFAULT_PARAMS_TWF()

par.Np = 1
par.No = 1
par.bg_solution = nil
par.plot_every = 20
par.plot_start = 1
par.show_plots = true
par.beta = 0.9
par.fourier_relax_factor = 5e-2
par.position_refinement_start = 50
par.position_refinement_every = 3
par.position_refinement_max_disp = 2
par.fm_support_radius = function(it) return nil end
par.fm_mask_radius = function(it) return nil end

par.probe_update_start = 3
par.probe_support = 0.5
par.probe_regularization_amplitude = function(it) return nil end
par.probe_inertia = 1e-9
par.probe_lowpass_fwhm = function(it) return nil end

par.object_highpass_fwhm = function(it) return nil end
par.object_inertia = 1e-5

par.P_Q_iterations = 10
par.copy_probe = false
par.copy_object = false
par.margin = 0
par.background_correction_start = 1e5

par.save_interval = 50
par.save_path = '/tmp/'
par.save_raw_data = false

par.O_denom_regul_factor_start = 1e-10
par.O_denom_regul_factor_end = 1e-12

par.pos = pos
par.dpos = dpos
par.dpos_solution = dpos_solution
par.object_solution = object_solution
par.probe_solution = probe
par.a = a
par.fmask = fmask
par.probe = nil

local ngin = ptycho.DM_engine(par)
-- ngin:generate_data('/home/philipp/drop/Public/moon_subpix2.h5')
ngin:iterate(250)
