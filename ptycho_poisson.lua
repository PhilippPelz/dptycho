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
local file = 'moon_subpix_poisson.h5'

local engine = require 'dptycho.core.ptycho.TWF_engine'

local par = {}
par.i = 50
par.DM_smooth_amplitude = 1
par.probe_change_start = 1
local f = hdf5.open(path..file,'r')



local pos = f:read('/scan_info/positions_int'):all():int()

-- dpos[{1,1}] = 5

local M = 128
-- local a = f:read('/data_unshift'):all():cuda()
local a = torch.CudaTensor(pos:size(1),M,M)
local fmask = a:clone():fill(1)
-- print(dpos)

local dpos_solution = pos:clone():float():zero()
-- local dpos_solution = f:read('/scan_info/dpos'):all():float()
local dpos = pos:clone():float():zero()
-- local dpos = f:read('/scan_info/positions'):all():float():add(-1,pos:float())
-- local dpos_solution  = dpos:clone()

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
-- local bg_r = f:read('/bg_r'):all():cuda()
-- local bg_i = f:read('/bg_i'):all():cuda()
-- local bg_r = f:read('/bgr'):all()
-- local bg = bg_r:pow(2):add(bg_i:pow(2)):mul(1e5)
-- plt:plot(bg_r:pow(2):add(bg_i:pow(2)))
local o_r = f:read('/or'):all():cuda()
local o_i = f:read('/oi'):all():cuda()
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
local solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)
-- local bgc = torch.ZCudaTensor.new(bg_r:size()):copyIm(bg_i):copyRe(bg_r)
-- bgc:fftshift()
-- local dpos = pos:clone():float():zero()
-- plt:plot(bgc:zfloat(),'bgc')
pprint(probe)
pprint(solution)
-- plt:plot(probe:zfloat())
-- plt:plot(solution:zfloat())

o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

-- frames

DEBUG = false

local nmodes_probe = 1
local nmodes_object = 1
-- pprint(a)

par = {
  nmodes_probe = 1,
  nmodes_object = 1,
  probe = nil,
  plot_every = 2,
  plot_start = 1,
  beta = 0.9,
  fourier_relax_factor = 5e-2,
  position_refinement_start = 100,
  position_refinement_every = 3,
  probe_update_start = 2,
  object_inertia = 1e-5,
  probe_inertia = 1e-9,
  P_Q_iterations = 10,
  copy_solution = true,
  background_correction_start = 100
}
par.pos = pos
par.dpos = dpos
par.dpos_solution = dpos_solution
par.solution = solution
par.probe_solution = probe
par.a = a
par.fmask = fmask
par.probe = probe
-- par.bg_solution = bgc:abs():mul(4e3)
-- par.bg_solution = torch.CudaTensor(M,M):zero()

local ngin = engine(par)
-- ngin:generate_data('/home/philipp/drop/Public/moon_subpix_poisson',1e6)
ngin:iterate(250)
