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

local path = '/home/philipp/drop/Public/'
local file = 'po2.h5'

local engine = require 'dptycho.core.ptycho.DM_engine'


local par = {}
par.i = 50
par.DM_smooth_amplitude = 1
par.probe_change_start = 1

local f = hdf5.open(path..file,'r')

local a = f:read('/data_unshift'):all():cuda()
local pos = f:read('/scan_info/positions_int'):all():int()
-- local pos = f:read('/positions_int'):all():int()
pos:add(1)
-- local dx_spec = f:read('/scan_info/dx_spec')
-- local w = f:read('/fmask'):all():cuda()
local o_r = f:read('/or'):all():cuda()
local o_i = f:read('/oi'):all():cuda()
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
local solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)
local dpos = pos:clone():float():zero()
-- plt:plotReIm(solution:zfloat(),'solution')
-- plt:plotReIm(probe:zfloat(),'probe')
o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

-- frames

local nmodes_probe = 1
local nmodes_object = 1

local ngin = engine(pos,a,nmodes_probe,nmodes_object,solution,probe,dpos)
-- ngin:generate_data('/home/philipp/drop/Public/po.h5')
ngin:iterate(200)
