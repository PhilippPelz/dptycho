require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'pprint'

local dataloader = require 'dptycho.io.dataloader'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()

torch.setdefaulttensortype('torch.ZCudaTensor')
--print(type(torch.ZCudaTensor()))
--local data = u.load_sim_and_allocate('/home/philipp/projects/slicepp/Examples/configs/ball2.h5')
local data = u.load_sim_and_allocate_stacked('/home/philipp/projects/slicepp/Examples/configs/ball2.h5')

pprint(data.real_deltas)
pprint(data.atompot)


local net = nn.Sequential()
--print 'here'
--for i=1,data.nslices do
--  local W = {}
--  local dW = {}
--  for Z, _ in pairs(data.real_deltas) do
--    W[Z] = data.real_deltas[Z][i]
----    plt:plot(W[Z]:float(),'weights_' .. Z .. "_" .. i)
----    pprint(W[Z])
--    dW[Z] = data.gradWeights[Z][i]
----    pprint(dW[Z])
--  end
--  net:add(znn.ConvSlice(data.atompot,data.inv_atompot,W,dW))
--  net:add(znn.ConvFFT2D(data.prop,data.bwprop))
--end
--net:add(znn.FFT())
--
--plt:plot(data.atompot[1]:zfloat(),'atompot')
for i=1,data.nslices do
  local arm = nn.Sequential()  
  arm:add(znn.ConvParams(data.Wsize,data.atompot,data.inv_atompot,data.real_deltas[i],data.gradWeights[i]))
  arm:add(nn.Sum(1))
  
  net:add(znn.CMulModule(arm))
  net:add(znn.ConvFFT2D(data.prop,data.bwprop))
end
net:add(znn.FFT())

--local slice = znn.ConvSlice(d.atompot,inv_pot,real_deltas,gradWeights)
--local res = slice:forward(input)
--local slice = znn.ConvSlice(d.atompot,inv_pot,real_deltas,gradWeights)
--plt:plot(data.probe:zfloat(),'probe')
local res = net:forward(data.probe)

psh = res:fftshift()
plt:plot(psh:zfloat(),'shifted')

-- plt:plot(res:zfloat(),'Net Output')


--    tmp = asarray(x)
--    ndim = len(tmp.shape)
--    print(ndim)
--    if axes is None:
--        axes = list(range(ndim))
--    elif isinstance(axes, integer_types):
--        axes = (axes,)
--    y = tmp
--    for k in axes:
--        n = tmp.shape[k]
--        p2 = (n+1)//2
--        mylist = concatenate((arange(p2, n), arange(p2)))
--        y = take(y, mylist, k)
--    return y
