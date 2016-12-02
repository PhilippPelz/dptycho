local classic = require 'classic'
local dataloader = require 'dptycho.io.dataloader'
local plot = require 'dptycho.io.plot'
local plt = plot()
local py = require('fb.python')
local argcheck = require 'argcheck'
require 'pprint'
require 'zcutorch'
local m = classic.module(...)

m:submodule("stats")
m:class("linear_schedule")
m:class("tabular_schedule")
m:class("physics")

py.exec([=[
import numpy as np
from numpy.fft import fft
import scipy.ndimage.filters as f

def DTF2D(N):
  W = fft(np.eye(N))
  W2D = np.kron(W,W)/N
  return W2D.real, W2D.imag

def rgb2hsv1(rgb):
    """
    Reverse to :any:`hsv2rgb`
    """
    eps = 1e-6
    rgb=np.asarray(rgb).astype(float)

    #print 'rgb shape'
    #print rgb.shape

    maxc = rgb.max(axis=0)
    minc = rgb.min(axis=0)
    v = maxc
    s = (maxc-minc) / (maxc+eps)
    s[maxc<=eps]=0.0
    rc = (maxc-rgb[0,:,:]) / (maxc-minc+eps)
    gc = (maxc-rgb[1,:,:]) / (maxc-minc+eps)
    bc = (maxc-rgb[2,:,:]) / (maxc-minc+eps)

    h =  4.0+gc-rc
    maxgreen = (rgb[1,:,:] == maxc)
    h[maxgreen] = 2.0+rc[maxgreen]-bc[maxgreen]
    maxred = (rgb[0,:,:] == maxc)
    h[maxred] = bc[maxred]-gc[maxred]
    h[minc==maxc]=0.0
    h = (h/6.0) % 1.0

    return np.asarray((h, s, v))

def hsv2complex1(cin):
    """
    Reverse to :any:`complex2hsv`
    """
    h,s,v = cin
    return v * np.exp(np.pi*2j*(h-.5)) /v.max()



def complex2hsv(cin, vmin=None, vmax=None):
    """\
    Transforms a complex array into an RGB image,
    mapping phase to hue, amplitude to value and
    keeping maximum saturation.

    Parameters
    ----------
    cin : ndarray
        Complex input. Must be two-dimensional.

    vmin,vmax : float
        Clip amplitude of input into this interval.

    Returns
    -------
    rgb : ndarray
        Three dimensional output.

    See also
    --------
    complex2rgb
    hsv2rgb
    hsv2complex
    """
    # HSV channels
    h = .5*np.angle(cin)/np.pi + .5
    s = np.ones(cin.shape)

    v = abs(cin)
    if vmin is None: vmin = 0.
    if vmax is None: vmax = v.max()
    #print vmin, vmax
    assert vmin < vmax
    v = (v.clip(vmin,vmax)-vmin)/(vmax-vmin)

    return np.asarray((h,s,v))

def complex2rgb(r,i):
    """
    Executes `complex2hsv` and then `hsv2rgb`

    See also
    --------
    complex2hsv
    hsv2rgb
    rgb2complex
    """
    cin = r + 1j * i
    return hsv2rgb(complex2hsv(cin))

def hsv2rgb(hsv):
    """\
    HSV (Hue,Saturation,Value) to RGB (Red,Green,Blue) transformation.

    Parameters
    ----------
    hsv : array-like
        Input must be two-dimensional. **First** axis is interpreted
        as hue,saturation,value channels.

    Returns
    -------
    rgb : ndarray
        Three dimensional output. **First** axis is interpreted as
        red, green, blue channels.

    See also
    --------
    complex2rgb
    complex2hsv
    rgb2hsv
    """
    # HSV channels
    h,s,v = hsv

    i = (6.*h).astype(int)
    f = (6.*h) - i
    p = v*(1. - s)
    q = v*(1. - s*f)
    t = v*(1. - s*(1.-f))
    i0 = (i%6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    rgb = np.zeros((3,) + h.shape , dtype=h.dtype)
    rgb[0,:,:] = 255*(i0*v + i1*q + i2*p + i3*p + i4*t + i5*v)
    rgb[1,:,:] = 255*(i0*t + i1*v + i2*v + i3*q + i4*p + i5*p)
    rgb[2,:,:] = 255*(i0*p + i1*p + i2*t + i3*v + i4*v + i5*q)

    return rgb.astype(np.float32)

def rgb2complex1(rgb):
    """
    Reverse to :any:`complex2rgb`
    """
    cout = hsv2complex1(rgb2hsv1(rgb))
    return (cout.real.astype(np.float32),cout.imag.astype(np.float32))
]=])


function m.read_file(path)
    local file = io.open(path, "rb") -- r read mode and b binary mode
    if not file then return nil end
    local content = file:read "*a" -- *a or *all reads the whole file
    file:close()
    return content
end

function m.script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function m.DTF2D(N)
  local WR,WI = py.eval('DTF2D(N)',{N=N})
  local f = torch.FloatTensor(N,N,2)
  z[{{},{},{1}}]:copy(WR)
  z[{{},{},{2}}]:copy(WI)
  local z = torch.ZFloatTensor(N,N):copy(f)
  return z
end

function m.unwrap(x)
  return py.eval('np.unwrap(x)',{x=x})
end

function m.percentile(a,perc)
  return py.eval('np.percentile(a,q)',{a=a,q=perc})
end

function m.gf(a,sigma)
  return py.eval('f.gaussian_filter(a,sigma)',{a=a,sigma=sigma})
end

function m.load_sim_and_allocate(file)
  local ret = {}
  local d = dataloader()
  local data = d:loadHDF5(file)
--  pprint(data)
  ret.prop = data.propagator:zcuda()
  ret.bwprop = ret.prop:clone():conj()
  ret.probe = data.probe:zcuda()
  ret.positions = data.positions
  ret.atompot = {}
  ret.inv_atompot = {}
  for Z, pot in pairs(data.atompot) do
  --  print(Z)
    ret.atompot[Z] = pot:mul(pot:nElement()):zcuda()
    ret.inv_atompot[Z] = ret.atompot[Z]:pow(-1)
    -- plt:plot(inv_pot[Z]:zfloat())
  end
  ret.deltas = {}
  ret.gradWeights = {}
  for Z, delta in pairs(data.deltas) do
    ret.deltas[Z] = delta:re():cuda()
    ret.gradWeights[Z] = delta:clone():zero()
  end
  ret.nslices = data.deltas[data.Znums[1]]:size():totable()[1]
  return ret
end

function m.r(N)
  local x = torch.repeatTensor(torch.linspace(-N/2,N/2,N),N,1)
  local y = x:clone():t()
  local r2 = (x:pow(2) + y:pow(2))
  return r2:sqrt()
end

function m.r2(N)
  local x = torch.repeatTensor(torch.linspace(-N/2,N/2,N),N,1)
  local y = x:clone():t()
  local r2 = (x:pow(2) + y:pow(2))
  return r2
end

m.qq = argcheck{
   nonamed=true,
   {name="n", type='number'},
   {name="dx", type='number'},
   call = function(n,dx)
     local r = m.r(n)
     return r:div(n*dx)
   end
}

m.qq2 = argcheck{
   nonamed=true,
   {name="n", type='number'},
   {name="dx", type='number'},
   call = function(n,dx)
     local qq = m.qq(n,dx)
     return qq:pow(2)
   end
}

function m.load_sim_and_allocate_stacked(path,file,ptycho)
  local ret = {}
  local d = dataloader()
  local data = d:loadHDF5(path,file,ptycho)
--  pprint(data)
  ret.nslices = data.deltas[1]:size():totable()[1]
  ret.prop = data.propagator:zcuda()
  ret.bwprop = ret.prop:clone():conj()
  ret.probe = data.probe:zcuda()
  ret.positions = data.positions
  ret.Znums = data.Znums
  ret.a_k = data.a_k:cuda()--:sqrt()

  local ps = ret.probe[1]:size():totable()
  local s = data.atompot[1]:size():totable()
  local offset = math.floor((s[1] - ps[1])/2)
  local mi = {offset,offset+ps[1]-1}
  local middle = {mi,mi}
  -- pprint(middle)
--  pprint(s)
  local potsize = {#data.atompot,ps[1],ps[2]}
  ret.Wsize = potsize
--  pprint(potsize)
  ret.atompot = torch.ZCudaTensor().new(potsize)
  ret.inv_atompot = torch.ZCudaTensor().new(potsize)
--  pprint(ret.atompot:size())
--  pprint(ret.inv_atompot:size())
  local i = 1
  for _, pot in pairs(data.atompot) do
--    print(Z)
    -- local s = ret.atompot[i]
--    print('after s')
--    pprint(s)
    -- pprint(pot)
    local p = pot:mul(pot:nElement()):fftshift()
    -- plt:plot(p,'middle of atompot 1')
    -- pprint(p)
    p = p[middle]
    -- pprint(p)
    p:fftshift()
    -- plt:plot(p,'middle of atompot 2')/

    ret.atompot[i]:copy(p)
--    pprint(ret.atompot)
    -- local pow = ret.atompot[i]:clone()
--    pprint(pow)
    ret.inv_atompot[i]:copy(ret.atompot[i]):pow(-1)
    i = i + 1
    -- plt:plot(inv_pot[Z]:zfloat())
  end
--  print('wp 1')
  ret.deltas = torch.CudaTensor(#data.atompot,ret.nslices,s[1],s[2])
--  pprint(ret.deltas)
--  print('wp 2')
  ret.gradWeights = torch.CudaTensor(#data.atompot,ret.nslices,s[1],s[2]):zero()
--  print('wp 3')
  for j =1, ret.nslices do
    for k, delta in ipairs(data.deltas) do
      -- local s = ret.deltas[j]
--      pprint(s)
--      pprint(data.deltas[Z])
      ret.deltas[k][j]:copy(data.deltas[k][j]:re())
    end
  end
  return ret
end

function m.printMem()
  local freeMemory, totalMemory = cutorch.getMemoryUsage(cutorch.getDevice())
  freeMemory = freeMemory / (1024*1024)
  totalMemory = totalMemory / (1024*1024)
  print(string.format('free: %d MB, total: %d MB, used: %d MB',freeMemory,totalMemory,totalMemory -freeMemory))
end

function m.printram(str)
  -- print('================ MEMORY REPORT ================')
  -- print(str)
  -- -- free -m
  -- local handle = io.popen('free -m | cat')
  -- if handle then
  --   local result = handle:read("*a")
  --   print(result)
  --   handle:close()
  -- end
  -- print('===============================================')
end

function m.initial_probe(size1,support_ratio)
  local ratio = support_ratio or 0.5
  local size = m.copytable(size1)
  size[#size+1] = 2
  -- pprint(size)
  local r = torch.randn(unpack(size))
  -- pprint(r)
  local res = torch.ZFloatTensor(unpack(size1)):copy(r)
  -- pprint(res)
  local mod = znn.SupportMask(size1,size1[#size1]*ratio)
  res = mod:forward(res)
    -- pprint(res)

  plt:plot(res,'probe')
  return res
end

function m.copytable(obj, seen)
  if type(obj) ~= 'table' then return obj end
  if seen and seen[obj] then return seen[obj] end
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do
    if k ~= '_classAttributes' then
      -- print(k)
      -- pprint(v)
      res[m.copytable(k, s)] = m.copytable(v, s)
    end
  end
  return res
end

function m.meshgrid(x,y)

end

function m.complex2rgb(cx)
  local r = cx:re()
  local i  = cx:im()
  local rgb = py.eval('complex2rgb(r,i)',{r=r,i=i})
  return rgb
end

function m.rgb2complex(rgb1)
  local x = py.eval('rgb2complex1(rgb1)',{rgb1=rgb1})
  local r,i = table.unpack(x)
  -- local s = r:size():totable()
  --
  -- local f = torch.FloaTensor(s[1],s[2],2)
  -- f[{{},{},{1}}]:copy(r)
  -- f[{{},{},{2}}]:copy(i)

  local cx = torch.ZFloatTensor.new(r,i)
  return cx
end

function m.printf(s,...)
  return io.write(s:format(...)..'\n')
end

function m.debug(s,...)
  if DEBUG then
    print(string.format(s,...))
  end
end

return m
