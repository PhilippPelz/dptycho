require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local dataloader = require 'dptycho.io.dataloader'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local plot = require 'dptycho.io.plot'
local plt = plot()

torch.setdefaulttensortype('torch.FloatTensor')

--local myFile = hdf5.open('/home/philipp/projects/slicepp/Examples/configs/gold.h5', 'r')
--local data = myFile:read('/atomDeltas_14'):all()

local d = dataloader()
d:loadHDF5('/home/philipp/projects/slicepp/Examples/configs/ball2.h5')

local sigma = 1.00871e+07

local islice = 19
--local input = d.atompot[26]:clone():zero():zcuda():fillRe(1)
local input = d.zprobe:zcuda()

local one = input:clone()
local inv_pot = {}
for Z, pot in pairs(d.atompot) do
--  print(Z)
  d.atompot[Z] = pot:mul(pot:nElement()):zcuda()
  inv_pot[Z] = one:clone():cdiv(d.atompot[Z])
  -- plt:plot(inv_pot[Z]:zfloat())
end
local real_deltas = {}
local gradWeights = {}
for Z, delta in pairs(d.deltas) do
  real_deltas[Z] = d.deltas[Z]:re():cuda()
  gradWeights[Z] = real_deltas[Z]:clone():zero()
end
pprint(d.atompot)
pprint(inv_pot)
pprint(real_deltas)
pprint(gradWeights)

local nslices = d.deltas[d.Znums[1]]:size():totable()[1]
local prop = d.zpropagator:zcuda()
local bwprop = prop:clone():conj()


local net = nn.Sequential()
print 'here'
for i=1,nslices do
  local W = {}
  local dW = {}
  for Z, delta in pairs(real_deltas) do
    W[Z] = real_deltas[Z][i]
--    pprint(W[Z])
    dW[Z] = gradWeights[Z][i]
--    pprint(dW[Z])
  end
  net:add(znn.ConvSlice(d.atompot,inv_pot,W,dW))
  net:add(znn.ConvFFT2D(prop,bwprop))
end

--local slice = znn.ConvSlice(d.atompot,inv_pot,real_deltas,gradWeights)
--local res = slice:forward(input)
--local slice = znn.ConvSlice(d.atompot,inv_pot,real_deltas,gradWeights)
local res = net:forward(input)

plt:plot(res:zfloat(),'Net Output')
plt:plot(d.zpotential[{islice}]:arg())