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
local file = 'moon2.h5'

local ptycho = require 'dptycho.core.ptycho'

local par = {}
par.i = 50
par.DM_smooth_amplitude = 1
par.probe_change_start = 1

local f = hdf5.open(path..file,'r')

local a = f:read('/data_unshift'):all():cuda()
local fmask = a:clone():fill(1)
local pos = f:read('/scan_info/positions_int'):all():int():add(1)
local dpos = pos:clone():float():zero()
-- local dpos = f:read('/scan_info/dpos'):all():float()
-- print(dpos)
dpos:zero()
local dpos_solution  = dpos:clone()
local o_r = f:read('/or'):all():cuda()--:view(torch.LongStorage{1,1,962,962})
local o_i = f:read('/oi'):all():cuda()--:view(torch.LongStorage{1,1,962,962})
-- local f1 = hdf5.open('probe2.h5','r')
local pr = f:read('/pr'):all():cuda()--:view(torch.LongStorage{1,1,512,512})
local pi = f:read('/pi'):all():cuda()--:view(torch.LongStorage{1,1,512,512})
local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
local object_solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)
-- plt:plotReIm(probe[1][1]:zfloat())
-- plt:plot(probe[1][1]:zfloat())
-- plt:plot(object_solution[1][1]:zfloat())
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
par.plot_every = 5
par.plot_start = 1
par.show_plots = true
par.beta = 0.9
par.fourier_relax_factor = 8e-2
par.position_refinement_start = 250
par.position_refinement_every = 3
par.position_refinement_max_disp = 2
par.fm_support_radius = function(it) return nil end
par.fm_mask_radius = function(it) return nil end

par.probe_update_start = 250
par.probe_support = 0.5
par.probe_regularization_amplitude = function(it) return nil end
par.probe_inertia = 0
par.probe_lowpass_fwhm = function(it) return nil end

par.object_highpass_fwhm = function(it) return nil end
par.object_inertia = 0
par.object_init = 'trunc'
par.object_init_truncation_threshold = 90

par.P_Q_iterations = 10
par.copy_probe = true
par.copy_object = false--true
par.margin = 0
par.background_correction_start = 1e5

par.save_interval = 250
par.save_path = '/tmp/'
par.save_raw_data = true
par.run_label = 'ptycho2'


par.O_denom_regul_factor_start = 0
par.O_denom_regul_factor_end = 0

par.pos = pos
par.dpos = dpos
par.dpos_solution = dpos_solution
par.object_solution = object_solution
par.probe_solution = probe
par.a = a
par.fmask = fmask
par.P = nil
par.O = nil

par.twf.a_h = 25
par.twf.a_lb = 1e-3
par.twf.a_ub = 1e1
par.twf.mu_max = 0.01
par.twf.tau0 = 10
par.twf.nu = 1e-2

local run_config = {{17,ptycho.DM_engine},{200,ptycho.TWF_engine}}
local runner = ptycho.Runner(run_config,par)
runner:run()
