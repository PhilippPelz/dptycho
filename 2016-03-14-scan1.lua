require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'csvigo'
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
local file = 'scan2_data_final_custommask.h5'

local f = hdf5.open(path..file,'r')

local a = f:read('/a'):all():cuda()
local fmask = f:read('/fm'):all():cuda()
local pos = f:read('/scan_info/positions'):all()
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
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
-- local f = hdf5.open(path..'ptycho_5.h5','r')
-- local pr = f:read('/pr'):all():cuda()
-- local pi = f:read('/pi'):all():cuda()
-- local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
-- f:close()
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
  Np = 4,
  No = 1,
  probe = nil,
  plot_every = 1,
  plot_start = 1,
  show_plots = false,
  beta = 0.95,
  fourier_relax_factor = 15e-2,
  position_refinement_start = 5,
  position_refinement_every = 1,
  position_refinement_max_disp = 2,
  probe_update_start = 2,
  probe_support = 7/16.0,
  object_inertia = 1e-9,
  probe_inertia = 1e-8,
  P_Q_iterations = 10,
  copy_solution = false,
  background_correction_start = 50,
  save_interval = 5,
  margin = 0,
  save_path = path..'/hyperscan7/'
}

-- local params = csvigo.load{path=path..'tested_params.csv', mode='tidy'}
-- pprint(params)
par_save = u.copytable(par)
-- for k,v in pairs(par_save) do
--   if torch.type(v) == 'boolean' then v = string.format("%s",v) end
--   -- print(k)
--   -- pprint(params[k])
--   table.insert(params[k],v)
-- end
--
-- csvigo.save(path..'tested_params.csv',params)

par.par_save = par_save
par.fm_support_radius = function(it) return nil end
par.probe_regularization_amplitude = function(it) return nil end
par.probe_lowpass_fwhm = u.linear_schedule(3,250,200,200)
par.object_highpass_fwhm = function(it) return nil end
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
