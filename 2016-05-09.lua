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

local path = '/home/philipp/experiments/2016-05-09/scan4/'
local file = 'scan2_data_final.h5'
local probe_file = 'probe.h5'

local f = hdf5.open(path..file,'r')

local a = f:read('/a'):all():cuda()
-- [{{1},{},{}}]
local fmask = f:read('/fm'):all():cuda()
local pos = f:read('/scan_info/positions'):all()
local dpos = pos:clone():float()
pos = pos:int()
dpos:add(-1,pos:float())
-- dpos[{1,1}] = 5

-- print(dpos)
-- print(pos)

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

-- plt:plot(a[1]:float():log())
-- plt:plot(fmask[1]:float():log())
-- local M = 1536
-- local f = hdf5.open(path..probe_file,'r')
o_r = nil
o_i = nil
-- local pr = f:read('/pr'):all():cuda()
-- local pi = f:read('/pi'):all():cuda()
-- local probe = torch.ZCudaTensor().new({1,3,M,M})
-- probe[1][1]:copyIm(pi):copyRe(pr)
-- plt:plot(probe[1][1]:zfloat())
pr = nil
pi = nil
collectgarbage()

-- local M = 1536
-- local x = torch.repeatTensor(torch.linspace(-M/2,M/2,M),M,1)
-- -- pprint(x)
-- local y = x:clone():t()
-- local r = (x:pow(2) + y:pow(2)):sqrt()
-- plt:plot(r)

-- frames
local reg_schedule = classic.class("linear_schedule")
function reg_schedule:_init(start,iterations,startval,stopval)
  self.start = start
  self.iterations = iterations
  self.startval = startval
  self.stopval = stopval
end

function reg_schedule:__call(it)
  if it < self.start or it > self.start + self.iterations then return 0
  else
    return self.startval - (self.startval - self.stopval) * (it-self.start)/self.iterations
  end
end

DEBUG = false

par = {
  nmodes_probe = 3,
  nmodes_object = 1,
  probe = nil,
  plot_every = 1,
  plot_start = 1,
  beta = 0.9,
  fourier_relax_factor = 5e-2,
  position_refinement_start = 3,
  position_refinement_every = 3,
  probe_update_start = 3,
  probe_regularization_amplitude = reg_schedule(3,50,1e-9,0),
  object_inertia = 1e-5,
  probe_inertia = 3e-7,
  P_Q_iterations = 6,
  copy_solution = false,
  background_correction_start = 30,
  save_interval = 10,
  save_path = path
}
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
