require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local dataloader = require 'dptycho.io.dataloader'
local znn = require 'dptycho.znn'
local plot = require 'dptycho.io.plot'
local plt = plot()

torch.setdefaulttensortype('torch.FloatTensor')

--local myFile = hdf5.open('/home/philipp/projects/slicepp/Examples/configs/gold.h5', 'r')
--local data = myFile:read('/atomDeltas_14'):all()

local d = dataloader()
d:loadHDF5('/home/philipp/projects/slicepp/Examples/configs/ball2.h5')

local sigma = 1.00871e+07

local slice = 19
local d1 = d.deltas[26][slice]
local d2 = d.deltas[8][slice]

local pot = d.atompot[26]:zcuda()

--local p = d.deltas[26][slice]:zcuda()
--plt:plot(d1:re())
--plt:plot(d2:re())
--print(d.atompot[14]:size())
--local pf = p:fft():cmul(pot):ifftU():zfloat()
--plt:plot(pf)
--:fillre(1)
local input = d.atompot[26]:clone():zero():zcuda():fillRe(1)
--print(torch.type(input))
--plt:plot(input:zfloat())


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
--  pprint(delta)
--  pprint(d.deltas[Z])
  -- plt:plot(delta[slice])
  -- print('WP 02')
  -- local p = d.deltas[Z]:re()
  -- pprint(p)
  real_deltas[Z] = d.deltas[Z][slice]:re():cuda()
  gradWeights[Z] = real_deltas[Z]:clone():zero()
end
pprint(d.atompot)
pprint(inv_pot)
pprint(real_deltas)
pprint(gradWeights)

local slice = znn.ConvSlice(sigma,d.atompot,inv_pot,real_deltas,gradWeights)
local res = slice:forward(input)
plt:plot(res:zfloat())
