require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'
require 'nn'
require 'cunn'

local cudnn = require 'cudnn'
local znn = require 'dptycho.znn'
local u = require 'dptycho.util'
local stats = require "dptycho.util.stats"
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local dataloader = require 'dptycho.io.dataloader'
local plt = plot()

cudnn.benchmark = true
cudnn.fastest = true

local path = '/home/philipp/projects/dptycho/'
local file = 'probe_object.h5'
local f = hdf5.open(path..file, 'r')
local obi = f:read('/ob_imag'):all():cuda()
local obr = f:read('/ob_real'):all():cuda()
local pri = f:read('/pr_imag'):all():cuda()
local prr = f:read('/pr_real'):all():cuda()
-- plt:plot(pri:float())
local probe = torch.ZCudaTensor(pri:size()):copyRe(prr):copyIm(pri):view(1,pri:size(1),pri:size(2))
local object = torch.ZCudaTensor(obi:size()):copyRe(obi):copyIm(obr)
local dO = torch.ZCudaTensor(object:size()):zero()
local tmp = torch.ZCudaTensor(probe:size())

local pf = object:zfloat()
-- plt:plot(pf)
-- plt:plotcompare({pf:re(),pf:im()})

local step = 30
local nsteps = 10
local psh = probe:size(2)
local probe_size = probe:size():totable()
local mask = znn.SupportMask(probe_size,probe_size[#probe_size]/2)

local measure = torch.CudaTensor(nsteps*nsteps,pri:size(1),pri:size(2))
local pos = torch.FloatTensor(nsteps*nsteps,2)

pprint(measure)
pprint(pos)
for i=1,nsteps do
  for j=1,nsteps do
    local xs, ys = i*step, j*step
    local slice = {{xs,xs+psh-1},{ys,ys+psh-1}}

    local net = nn.Sequential()
    local source = znn.Source(ctor,probe)
    -- source:immutable()

    net:add(source)
    net:add(mask)
    net:add(znn.CMul(object[slice],dO[slice]))
    -- propagate to fourier space
    net:add(znn.FFT())
    -- in: cx[R,M,M]  out: f[R,M,M]
    net:add(znn.ComplexAbs(tmp))
    -- in: [R,M,M]  out: [R,M,M]
    net:add(znn.Square())
    -- sum the mode intensities
    -- in: [R,M,M]  out: [1,M,M]
    net:add(znn.Sum(1,probe_size))
    -- make it 2 - dimensional
    -- in: [1,M,M]  out: [M,M]
    net:add(znn.Select(1,1))
    net:add(znn.Sqrt())

    local out = net:forward()
    print((i-1)*nsteps+j)
    measure[(i-1)*nsteps+j]:copy(out)
    pos[{(i-1)*nsteps+j,1}] = xs
    pos[{(i-1)*nsteps+j,2}] = ys
    -- plt:plot(out:float(),'output')
  end
end

measure = measure:float()
local f = hdf5.open(path..'simul.h5', 'w')
f:write('/measure',measure)
f:write('/pos',pos)
f:close()
-- print(pos)
