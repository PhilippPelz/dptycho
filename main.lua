require 'hdf5'
require 'ztorch'
require 'zcutorch'

local DataLoader = require 'io.DataLoader'

local plt = require 'gnuplot'
local classic = require 'classic'

--local myFile = hdf5.open('/home/philipp/projects/slicepp/Examples/configs/gold.h5', 'r')
--local data = myFile:read('/atomDeltas_14'):all()

local d = DataLoader('/home/philipp/projects/slicepp/Examples/configs/gold.h5')

local p = d.zpropagator:zcuda()
print(torch.type(p))
local p1 = d.propagator
--print(p1:type())
--print(data:dim())
--local s = data[{4,{},{},1}]
--print(s:type())
--plt.imagesc(s,'color')
--print(s:size())
--print(data)
--print(data:)
--myFile:close()