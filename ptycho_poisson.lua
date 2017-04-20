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
local file = 'moon6.h5'

local ptycho = require 'dptycho.core.ptycho'

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
probe:div(2)
plt:plotReIm(probe[1][1]:zfloat())
-- plt:plot(probe[1][1]:zfloat())
plt:plot(object_solution[1][1]:zfloat())
o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

DEBUG = false
path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/scan10/'
rawset(_G, 'path', path)
par = ptycho.params.DEFAULT_PARAMS_TWF()

par.Np = 1
par.No = 1
par.bg_solution = nil
par.plot_every = 50
par.plot_start = 0
par.show_plots = true
par.beta = 1
par.fourier_relax_factor = 8e-2
par.position_refinement_start = 20
par.position_refinement_stop = 70
par.position_refinement_every = 3
par.position_refinement_max_disp = 1.5
par.position_refinement_method = 'annealing'
par.position_refinement_trials = 4
par.fm_support_radius = function(it) return nil end
par.fm_mask_radius = function(it) return nil end

par.probe_update_start = 300
par.probe_support = nil
par.probe_support_fourier = nil
par.probe_inertia = 0
par.probe_regularization_amplitude = function(it) return nil end
par.probe_lowpass_fwhm = function(it) return nil end
par.probe_keep_intensity = true
par.probe_init = 'copy'

par.object_highpass_fwhm = function(it) return nil end
par.object_inertia = 0
par.object_init = 'const'
par.object_init_truncation_threshold = 90
par.object_update_start = 0

par.margin = 0
par.background_correction_start = 1e5

par.save_interval = 50
par.save_path = path
par.save_raw_data = false
par.run_label = 'scan289'

par.O_denom_regul_factor_start = 1e-5
par.O_denom_regul_factor_end = 1e-9

par.pos = pos
par.dpos = dpos
par.dpos_solution = nil
par.object_solution = object_solution
par.probe_solution = probe
par.a = a
par.fmask = fmask
par.P = nil
par.O = nil

par.twf.a_h = 200
par.twf.a_lb = 4e-5
par.twf.a_ub = 1e1
par.twf.mu_max = 0.01
par.twf.tau0 = 10
par.twf.nu = 1e-2
par.twf.do_truncate = true
par.twf.diagnostics = false
par.twf.support_radius = nil


par.regularizer = znn.BM3D_MSE_Criterion--SpatialSmoothnessCriterion
par.optimizer = optim.sgd

par.experiment.z = 1.2199
par.experiment.E = 300e3
par.experiment.det_pix = 14*5e-6
par.experiment.N_det_pix = 256

-- par.optim_config = {}
-- par.optim_config = {}
-- par.optim_config.maxIter = 1
-- par.optim_config.sig = 0.5
-- par.optim_config.red = 1e8
-- par.optim_config.break_on_success = true
-- par.optim_config.verbose = true
-- par.optim_state = {}

par.optim_state = {}
par.optim_config = {}
par.optim_config.learningRate = 1e-4
par.optim_config.learningRateDecay = 1/10
par.optim_config.weightDecay = 0
par.optim_config.momentum = 0

par.optim_config_probe = {}
par.optim_config_probe.learningRate = 2e-4
par.optim_config_probe.learningRateDecay = 1/10
par.optim_config_probe.weightDecay = 0
par.optim_config_probe.momentum = 0

par.regularization_params = {}
par.regularization_params.amplitude = 7e-1
par.regularization_params.start_denoising = 3
par.regularization_params.denoise_interval = 1
par.regularization_params.sigma_denoise = 0.01

par.calculate_dose_from_probe = false
par.stopping_threshold = 1e-5

par.beam_radius = 40

par.ops = require 'dptycho.core.ptycho.ops_subpixel'

local ngin = ptycho.TWF_engine(par)
-- ngin:generate_data('/media/philipp/win1/ProgramData/Dropbox/Public/moon9',1e4, true)
ngin:iterate(250)
