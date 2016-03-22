local hdf5 = require 'hdf5'
local classic = require 'classic'
require 'zcutorch'
require 'pprint'
local c = classic.class(...)

function c:_init()
end

function c:loadHDF5(path,file,ptycho)
  local pos = ptycho or false
  local f = hdf5.open(path..file, 'r')

  local ret  = {}
  ret.Znums = f:read('/Znums'):all():int()
  -- local f2 = hdf5.open(path..'a_k.h5', 'r')
  -- ret.a_k = f2:read('/a_k'):all()
  -- f2:close()
  ret.a_k = f:read('/measurements'):all()
  ret.K = ret.a_k:size(1)
  ret.deltas = {}
  ret.atompot = {}
  ret.atomconv = {}
  for i = 1,ret.Znums:nElement() do
    local d = f:read('/atomDeltas_' .. ret.Znums[i]):all():squeeze()
    local pot = f:read('/atomicPotential_' .. ret.Znums[i]):all():squeeze()
--    local conv = f:read('/atomConv_' .. zn[i]):all():squeeze()

    local dsize = d:size():totable()
    table.remove(dsize,#dsize)
    dsize = torch.LongStorage(dsize)

    local potsize = pot:size():totable()
    table.remove(potsize,#potsize)
    potsize = torch.LongStorage(potsize)

    ret.deltas[i] = torch.ZFloatTensor(dsize):copy(d)
--    self.atomconv[zn[i]] = torch.ZFloatTensor(dsize):copy(conv)
    ret.atompot[i] = torch.ZFloatTensor(potsize):copy(pot)
--    print(type(self.deltas[zn[i]]:data()))
  end
  local probe = f:read('/probe'):all()
--  plt.imagesc(self.probe[{{},{},1}],'color')
  if pos then
    ret.positions = f:read('/positions'):all():squeeze()
    -- print('positions read')
  end
  local propagator = f:read('/propagator'):all():squeeze()
  local potential = f:read('/potentialSlices'):all():squeeze()

  local s = torch.LongStorage({probe:size(1),probe:size(2)})
  local s1 = torch.LongStorage({1,probe:size(1),probe:size(2)})
  local potsize = potential:size():totable()
  table.remove(potsize,#potsize)
  potsize = torch.LongStorage(potsize)

  ret.probe = torch.ZFloatTensor(s1)
  ret.probe[1]:copy(probe)
  -- print('probe info:')
  -- pprint(ret.probe)
  ret.propagator = torch.ZFloatTensor(s):copy(propagator)
  ret.potential = torch.ZFloatTensor(potsize):copy(potential)
--  pprint(ret.atompot)

  f:close()
  return ret
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
