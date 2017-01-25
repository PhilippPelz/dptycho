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
local ptycho = require 'dptycho.core.ptycho'

local path = '/home/philipp/phil/experiments/2017-01-23_melbourne/carbon2/'
local file = 'scan.h5'

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
-- plt:plot(a[1]:float())
-- [{{1},{},{}}]
local fmask = f:read('/fm'):all():cuda()
local pos = f:read('/pos'):all()
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
-- dpos[{1,1}] = 5

local NP = 1

o_r = nil
o_i = nil
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new({1,NP,480,480})
probe[1][1]:copyIm(pi):copyRe(pr)--:mul(3e4)
f:close()
-- probe[1][1]:copyIm(pi):copyRe(pr)
-- plt:plot(probe[1][1]:zfloat())
pr = nil
pi = nil
collectgarbage()


DEBUG = false

local par = ptycho.params.DEFAULT_PARAMS()

par.Np = NP
par.No = 1
par.bg_solution = nil
par.plot_every = 10
par.plot_start = 1
par.show_plots = true
par.beta = 0.9
par.fourier_relax_factor = 8e-2
par.position_refinement_start = 15
par.position_refinement_every = 5
par.position_refinement_max_disp = 5
par.fm_support_radius = function(it) return nil end
par.fm_mask_radius = function(it) return nil end

par.probe_update_start = 10
par.probe_support = 0.6
par.probe_regularization_amplitude = function(it) return nil end
par.probe_inertia = 1e-7
par.probe_lowpass_fwhm = function(it) return nil end

par.object_highpass_fwhm = function(it) return nil end

par.object_inertia = 1e-9

par.object_init = 'const'
par.object_init_truncation_threshold = 0.8

par.P_Q_iterations = 10
par.copy_probe = true
par.copy_object = false--true
par.margin = 0
par.background_correction_start = 1e5

par.save_interval = 250
par.save_path = path..'/recons1/'
par.save_raw_data = false
par.run_label = 'carbon_black'

par.O_denom_regul_factor_start = 0
par.O_denom_regul_factor_end = 0

par.pos = pos
par.dpos = dpos
par.dpos_solution = nil
par.object_solution = nil
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


local ngin = ptycho.DM_engine_subpix(par)
-- ngin:generate_data('/home/philipp/drop/Public/moon_subpix2.h5')
ngin:iterate(250)
