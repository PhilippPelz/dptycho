local hdf5 = require 'hdf5'
local classic = require 'classic'
require 'zcutorch'
local plt = require 'gnuplot'
local c = classic.class(...)

function c:_init(filepath)
  local f = hdf5.open(filepath, 'r')
  self.Znums = f:read('/Znums'):all():int()
  self.deltas = {}
  local zn = self.Znums:data()
  for i = 0,self.Znums:nElement()-1 do
    local ft = f:read('/atomDeltas_' .. zn[i]):all()    
    self.deltas[zn[i]] = torch.ZFloatTensor(ft)
--    print(type(self.deltas[zn[i]]:data()))
  end  
  self.probe = f:read('/probe'):all()
--  plt.imagesc(self.probe[{{},{},1}],'color')
  self.positions = f:read('/probe'):all()
  self.propagator = f:read('/probe'):all()
  local s = torch.LongStorage({self.probe:size(1),self.probe:size(2)})
  local ps = torch.LongStorage({self.positions:size(1),self.positions:size(2)})
  self.zprobe = torch.ZFloatTensor(s):copy(self.probe)
  self.zpositions = torch.ZFloatTensor(ps):copy(self.positions)
  self.zpropagator = torch.ZFloatTensor(s):copy(self.propagator)
--  print(torch.type(self.zpropagator))
  
--  local data = myFile:read('/atomDeltas_14'):all()
end

return c