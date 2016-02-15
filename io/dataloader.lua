local hdf5 = require 'hdf5'
local classic = require 'classic'
require 'zcutorch'
require 'pprint'
local c = classic.class(...)

function c:_init()
end

function c:loadHDF5(filepath,ptycho)
  local pos = ptycho or false
  local f = hdf5.open(filepath, 'r')
  self.Znums = f:read('/Znums'):all():int()
  self.deltas = {}
  self.atompot = {}
  local zn = self.Znums:data()
  for i = 0,self.Znums:nElement()-1 do
    local d = f:read('/atomDeltas_' .. zn[i]):all():squeeze()
    local pot = f:read('/atomicPotential_' .. zn[i]):all():squeeze()

    local dsize = d:size():totable()
    table.remove(dsize,#dsize)
    local dsize = torch.LongStorage(dsize)

    local potsize = pot:size():totable()
    table.remove(potsize,#potsize)
    local potsize = torch.LongStorage(potsize)

    self.deltas[zn[i]] = torch.ZFloatTensor(dsize):copy(d)
    self.atompot[zn[i]] = torch.ZFloatTensor(potsize):copy(pot)
--    print(type(self.deltas[zn[i]]:data()))
  end
  self.probe = f:read('/probe'):all()
--  plt.imagesc(self.probe[{{},{},1}],'color')
  if pos then
    self.positions = f:read('/positions'):all():squeeze()
  end
  self.propagator = f:read('/propagator'):all():squeeze()
  local s = torch.LongStorage({self.probe:size(1),self.probe:size(2)})

  self.zprobe = torch.ZFloatTensor(s):copy(self.probe)
  self.zpropagator = torch.ZFloatTensor(s):copy(self.propagator)
--  print(torch.type(self.zpropagator))

--  local data = myFile:read('/atomDeltas_14'):all()
end
function c:loadHDF5Part(filepath,datapath)
  local f = hdf5.open(filepath, 'r')
  return f:read(datapath):all():squeeze()
end

function c:loadText(filepath)
  local open = io.open

  local function read_file(path)
      local file = open(path, "rb") -- r read mode and b binary mode
      if not file then return nil end
      local content = file:read "*a" -- *a or *all reads the whole file
      file:close()
      return content
  end

  return read_file(filepath)
end

return c
