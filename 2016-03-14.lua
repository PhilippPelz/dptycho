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

local path = '/home/philipp/experiments/2016-03-14 ptycho/scan/'
local file = 'scan2_data_final_flipped.h5'

local M = 1536
local NP = 3

local f = hdf5.open(path..file,'r')

local a = f:read('/a'):all():cuda()
local fmask = f:read('/fm'):all():cuda()
local pos = f:read('/scan_info/positions'):all()
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
f:close()

local f = hdf5.open(path..'probe.h5','r')
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor().new({1,NP,M,M})
probe[1][1]:copyIm(pi):copyRe(pr)
probe[1][1]:copyIm(pi):copyRe(pr):mul(1e4)
f:close()
-- dpos[{1,1}] = 5

-- print(dpos)

-- local dpos_solution = pos:clone():float():zero()
-- local dpos_solution = f:read('/scan_info/dpos'):all():float()

-- dpos:add(-1,pos:float())
-- dpos:zero()
-- local errpos = stats.truncnorm({pos:size(1),pos:size(2)},-5,5,0,3)
-- print(errpos:max(),errpos:min())
-- dpos:add(errpos:float())

-- print(dpos)
-- dpos:zero()
-- local pos = f:read('/positions_int'):all():int()
-- local dx_spec = f:read('/scan_info/dx_spec')
-- local w = f:read('/fmask'):all():cuda()
-- local o_r = f:read('/or'):all():cuda()
-- local o_i = f:read('/oi'):all():cuda()
-- local o_r = f:read('/o_r'):all():cuda()
-- local o_i = f:read('/o_i'):all():cuda()
-- local pr = f:read('/pr'):all():cuda()
-- local pi = f:read('/pi'):all():cuda()
-- local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
-- local solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)
-- local dpos = pos:clone():float():zero()
-- plt:plot(solution:zfloat(),'solution')
-- plt:plot(probe:zfloat(),'probe')
o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

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
  position_refinement_start = 6,
  position_refinement_every = 2,
  position_refinement_max_disp = 3,
  probe_update_start = 2,
  probe_support = 0.7,
  fm_support_radius = function(it) return nil end,
  probe_regularization_amplitude = function(it) return nil end,
  object_highpass_fwhm = function(it) return nil end,
  object_inertia = 1e-7,
  probe_inertia = 1e-9,
  P_Q_iterations = 10,
  copy_solution = false,
  background_correction_start = 100,
  save_interval = 5,
  save_path = path..'/hyperscan_flipped_2/'
}

par.probe_lowpass_fwhm = u.linear_schedule(2,250,220,220)
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
