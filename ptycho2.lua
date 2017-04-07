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
local file = 'moon21.h5'

local ptycho = require 'dptycho.core.ptycho'

local f = hdf5.open(path..file,'r')

local a = f:read('/data_unshift'):all():cuda()
local fmask = a:clone():fill(1)
local pos = f:read('/scan_info/positions_int'):all():int():add(1)
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
print(dpos)
print(pos)
local dpos_solution  = dpos:clone()
local o_r = f:read('/or'):all():cuda()--:view(torch.LongStorage{1,1,962,962})
local o_i = f:read('/oi'):all():cuda()--:view(torch.LongStorage{1,1,962,962})
-- local f1 = hdf5.open('probe2.h5','r')
local pr = f:read('/pr'):all():cuda()--:view(torch.LongStorage{1,1,512,512})
local pi = f:read('/pi'):all():cuda()--:view(torch.LongStorage{1,1,512,512})
local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
local object_solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)

-- local pa = probe:abs()
-- probe:div(10)
-- a:div(pa:max())
-- plt:plotReIm(probe[1][1]:zfloat())
print(probe:normall(2)^2)
plt:plotReIm(probe[1][1]:zfloat())
plt:plot(object_solution[1][1]:zfloat())
o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

DEBUG = false

par = ptycho.params.DEFAULT_PARAMS()
par.Np = 1
par.No = 1
par.bg_solution = nil
par.plot_every = 50
par.plot_start = 0
par.show_plots = true
par.beta = 0.8
par.fourier_relax_factor = 5e-2
par.position_refinement_start = 1000
par.position_refinement_every = 3
par.position_refinement_max_disp = 2
par.fm_support_radius = function(it) return nil end
par.fm_mask_radius = function(it) return nil end

par.probe_update_start = 2
par.probe_support = 0.3
par.probe_inertia =  1e-3
par.probe_lowpass_fwhm = function(it) return nil end
par.probe_regularization_amplitude = function(it) return nil end

par.object_highpass_fwhm = function(it) return nil end

par.object_inertia = 1e-1
par.object_initial = object_solution[1][1]
par.object_init = 'const'

par.P_Q_iterations = 10
par.copy_probe = true
par.margin = 0
par.background_correction_start = 1e5

par.save_interval = 1000
par.save_path = '/tmp/'
par.save_raw_data = true
par.run_label = 'ptycho2'

par.O_denom_regul_factor_start = 1e-3
par.O_denom_regul_factor_end = 1e-15

par.pos = pos
par.dpos = dpos
par.dpos_solution = dpos_solution
par.object_solution = object_solution
par.probe_solution = probe
par.a = a
par.fmask = fmask
par.probe = nil
par.ops = require 'dptycho.core.ptycho.ops_subpixel'
local ngin = ptycho.DM_engine(par)
-- ngin:generate_data('/media/philipp/win1/ProgramData/Dropbox/Public/moon9',1e4, true)
ngin:iterate(500)
