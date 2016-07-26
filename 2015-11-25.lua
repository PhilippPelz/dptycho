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

local path = '/home/philipp/experiments/2015-11-25 oxford/scan2/'
local file = '2015-11-25_final_cropped_intpos.h5'
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

local a = f:read('/data'):all():cuda()
plt:plot(a[1]:float())
-- [{{1},{},{}}]
local fmask = f:read('/fmask'):all():cuda()
local pos = f:read('/scan_info/positions'):all()
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
-- dpos[{1,1}] = 5

local NP = 4

o_r = nil
o_i = nil
local f = hdf5.open(path..'probe_lowres_alpha5_neg4000.h5','r')
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new({1,NP,340,340})
probe[1][1]:copyIm(pi):copyRe(pr):mul(1e4)
f:close()
-- probe[1][1]:copyIm(pi):copyRe(pr)
-- plt:plot(probe[1][1]:zfloat())
pr = nil
pi = nil
collectgarbage()

-- u.linear_schedule(3,50,1e-9,0)
-- u.linear_schedule(3,50,500,1500),
-- frames

DEBUG = false

local par = ptycho.params.DEFAULT_PARAMS_TWF()

par.Np = NP
par.No = 1
par.bg_solution = nil
par.plot_every = 3
par.plot_start = 1
par.show_plots = true
par.beta = 0.9
par.fourier_relax_factor = 15e-2
par.position_refinement_start = 4
par.position_refinement_every = 1
par.position_refinement_max_disp = 2

par.probe_update_start = 3
par.probe_support = 0.4
par.probe_inertia = 1e-1

par.object_inertia = 1e-7

par.P_Q_iterations = 10
par.copy_probe = true
par.copy_object = false
par.margin = 0
par.background_correction_start = 1e5

par.save_interval = 50
par.save_path = path..'/hyperscan_lowres2/'
par.save_raw_data = false

par.O_denom_regul_factor_start = 1e-6
par.O_denom_regul_factor_end = 1e-9

par.pos = pos
par.dpos = dpos
par.dpos_solution = nil
par.object_solution = nil
par.probe_solution = probe
par.a = a
par.fmask = fmask
par.probe = nil


par.probe_lowpass_fwhm = function(it) return nil end--u.linear_schedule(6,250,130,130)
par.object_highpass_fwhm = function(it) return nil end--u.linear_schedule(6,250,170,170)
par.fm_mask_radius = function(it) return nil end--u.linear_schedule(1,250,170,170)
par.fm_support_radius = function(it) return nil end
par.probe_regularization_amplitude = function(it) return nil end --u.linear_schedule(6,250,170,170)--u.linear_schedule(1,250,700,900)

par.pos = pos
par.dpos = dpos
-- par.dpos_solution = dpos_solution
-- par.solution = solution
par.a = a
par.fmask = fmask
par.probe = probe


local ngin = ptycho.DM_engine(par)
-- ngin:generate_data('/home/philipp/drop/Public/moon_subpix2.h5')
ngin:iterate(250)
