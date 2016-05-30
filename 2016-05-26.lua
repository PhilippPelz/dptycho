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

local path = '/home/philipp/experiments/2016-05-26 lowres/scan1/'
local file = 'scan1_final3.h5'
local probe_file = 'probe_alpha5_neg23000.h5'

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

local NP = 6

o_r = nil
o_i = nil
local f = hdf5.open(path..probe_file,'r')
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new({1,NP,1536,1536})
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

par = {
  Np = NP,
  No = 1,
  probe = nil,
  plot_every = 1,
  plot_start = 1,
  show_plots = false,
  beta = 0.8,
  fourier_relax_factor = 15e-2,
  position_refinement_start = 5,
  position_refinement_every = 3,
  position_refinement_max_disp = 3,
  probe_update_start = 2,
  probe_support = nil,
  object_inertia = 1e-7,
  probe_inertia = 3e-8,
  P_Q_iterations = 10,
  copy_solution = false,
  background_correction_start = 100,
  save_interval = 5,
  save_path = path..'/hyperscan_5/'
}

par.probe_lowpass_fwhm = function(it) return nil end--u.linear_schedule(6,250,170,170)
par.object_highpass_fwhm = function(it) return 300 end--u.linear_schedule(6,250,170,170)
par.fm_mask_radius = function(it) return nil end--u.linear_schedule(3,250,170,170)
par.fm_support_radius = function(it) return nil end
par.probe_regularization_amplitude = function(it) return nil end --u.linear_schedule(6,250,170,170)--u.linear_schedule(1,250,700,900)

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
