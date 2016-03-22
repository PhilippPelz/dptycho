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

local path = '/home/philipp/projects/dptycho/'
local file = 'po.h5'



local par = {}
par.i = 50
par.DM_smooth_amplitude = 1
par.probe_change_start = 1

local f = hdf5.open(path..file,'r')

local a = f:read('/data_unshift'):all():cuda()
local pos = f:read('/scan_info/positions_int'):all()
pos:add(1)
local dx_spec = f:read('/scan_info/dx_spec')
local w = f:read('/fmask'):all():cuda()
local o_r = f:read('/or'):all():cuda()
local o_i = f:read('/oi'):all():cuda()
local solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)
o_r = nil
o_i = nil
collectgarbage()

-- frames
local K = a:size(1)
local M = a:size(2)
local MM = M*M
local nmodes_probe = 1
local nmodes_object = 1


local engine = classic.class('engine')

function engine:_init(pos,a,nmodes_probe,nmodes_object,solution)
  self.solution = solution
  self.pos = pos
  self.a = a
  self.nmodes_probe = nmodes_probe
  self.nmodes_object = nmodes_object
  self.object_size = pos:max(1):add(torch.Tensor({a:size(2),a:size(3)})):squeeze()
  self.O = torch.ZFloatTensor.new(nmodes_object,self.object_size[1]-1,self.object_size[2]-1):zcuda():fillRe(1):fillIm(0)
  -- plt:plot(self.O[1]:zfloat(),'object')
  self.O_denom = torch.CudaTensor(self.O:size()):zero()
  self.P = torch.ZFloatTensor.new(nmodes_probe,a:size(2),a:size(3)):random():zcuda()
  local probe_size = self.P:size():totable()
  local support = znn.SupportMask(probe_size,probe_size[#probe_size]/3)
  self.P = support:forward(self.P)
  self.z = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))
  self.P_Qz = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))
  self.P_Fz = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))

  self.z_tmp = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))
  self.P_tmp = torch.ZCudaTensor.new(nmodes_probe,a:size(2),a:size(3))
  self.O_tmp = torch.ZCudaTensor.new(nmodes_object,self.object_size[1],self.object_size[2])

  pprint(self.z_tmp)

  self.K = a:size(1)
  self.M = a:size(2)
  self.MM = M*M

  self.O_views = {}
  self.O_denom_views = {}
  self.err_hist = {}

  self.beta = 1
  self.norm_a = self.a:sum()

  for i=1,K do
    local slice = {{},{pos[i][1],pos[i][1]+M-1},{pos[i][2],pos[i][2]+M-1}}
    -- pprint(slice)
    self.O_views[i] = self.O[slice]
    self.O_denom_views[i] = self.O_denom[slice]
    self.z[i]:cmul(self.P,self.O_views[i]:expandAs(self.P))
    -- plt:plot(z[i][1]:zfloat(),'z_'..i)
  end
  self:calculateO_denom()

  u.printMem()
end

-- recalculate (Q*Q)^-1
function engine:calculateO_denom()
  self.O_denom:fill(1e-4)
  local norm_P =  self.P_tmp:norm(self.P):sum(1):re()
  plt:plot(norm_P[1]:float(),'norm_P - calculateO_denom')
  for k=1,K do
    self.O_denom_views[k]:add(norm_P)
  end
end

function engine:P_Q(z_in,z_out)
  -- print('engine:P_Q')
  -- pprint(self.z_tmp)
  -- pprint(self.P_tmp)
  -- pprint(self.O_tmp)
  -- pprint(z_in)
  -- pprint(z_out)
  local tmp = self.z_tmp[1]
  self.O:zero()
  for k=1,K do
    self.O_views[k]:add(tmp:conj(self.P):cmul(z_in[k]):sum(1))
  end
  self.O:cdiv(self.O_denom)
  plt:plot(self.O[1]:zfloat(),'object - engine:P_Q')
  for k=1,K do
    -- print('here')
    -- pprint(z_out[k])
    local ov = self.O_views[k]:repeatTensor(self.nmodes_probe,1,1)
    -- pprint(ov)
    z_out[k]:copy(ov):cmul(self.P)
    -- plt:plot(z_out[k][1]:zfloat(),'z_out[k] - engine:P_Q')
  end
end

function engine:P_F(z_in,z_out)
  local norm = self.P_tmp
  for k=1,K do
    -- plt:plot(z_in[k][1]:zfloat(),'z_in[k] - engine:P_F')
    z_out[k]:fftBatched(z_in[k])
    -- plt:plot(z_out[k][1]:zfloat(),'z_in[k] fft - engine:P_F')
    -- sum over probe intensities --> K x 1 x M x M -expand-> K x mprobe x M x M
    norm:norm(z_out[k]):sum(2):sqrt():expandAs(z_out[k])
    -- plt:plot(norm[1]:zfloat(),'norm - engine:P_F')
    -- z_in[k]:fftBatched()
    -- plt:plot(a[k]:float():log(),'a[k] - engine:P_F')
    z_out[k]:cmul(a[k]:view(1,self.M,self.M):expandAs(z_out[k]))
    -- plt:plot(z_out[k][1]:zfloat(),'z_out[k] 1 - engine:P_F')
    z_out[k]:cdiv(norm)
    -- plt:plot(z_out[k][1]:zfloat(),'z_out[k] 2 - engine:P_F')
    z_out[k]:ifftBatched()
    -- plt:plot(z_out[k][1]:zfloat(),'z_out[k] 3 - engine:P_F')
  end
end

function engine:simpleRefineProbe()
  local oview_conj = self.P_tmp[1]
  local sub = self.P_tmp
  local denom = self.P_tmp
  for k=1,K do
    local ovk = self.O_views[k]:repeatTensor(self.nmodes_probe,1,1)
    oview_conj:conj(ovk)
    self.z_tmp[k]:cmul(self.P_Fz[k],oview_conj)
    -- plt:plot(self.z_tmp[k][1]:zfloat(),'z_tmp[k] 1 - simpleRefineProbe')
    sub:norm(ovk):cmul(self.P)
    self.z_tmp[k]:add(-1,sub)
    -- plt:plot(self.z_tmp[k][1]:zfloat(),'z_tmp[k] 2 - simpleRefineProbe')
    denom:norm(ovk)
    denom:add(denom:mean()*1e-4)
    self.z_tmp[k]:cdiv(denom)
  end
  self.z_tmp = self.z_tmp:sum(1)
  plt:plot(self.z_tmp[1][1]:zfloat(),'self.z_tmp - simpleRefineProbe')
  self.P:add(self.z_tmp)
  plt:plot(self.P[1]:zfloat(),'new probe - simpleRefineProbe')
  self:calculateO_denom()
  plt:plot(self.O_denom[1]:float(),'new O_denom - simpleRefineProbe')
  print(self.O_denom:max(),self.O_denom:min(),self.O_denom:mean())
end

function engine:module_error(z)
  -- K x 1 x M x M
  local znorm = self.z_tmp:norm(z):sum(2):sqrt():re()
  return znorm:add(-1,self.a):sum()/self.norm_a
end

function engine:overlap_error(z_in,z_out)
  return self.z_tmp:add(z_in,-1,z_out):norm():sum() / z_out:norm():sum()
end

function engine:image_error()
  local norm = self.O_denom
  if self.solution then
    local c = self.O[1]:dot(self.solution)
    local phase_diff = c/zt.abs(c)
    self.O:mul(phase_diff)
    norm:normZ(self.O_tmp:add(self.O,-1,self.solution))
    return norm:sum()/self.solution:norm():sum()
  end
end

function engine:iterate(steps)
  local mod_error, overlap_error, image_error = -1,-1,-1
  for i=1,steps do
    self:P_Q(self.z,self.P_Qz)
    self:P_F(self.z,self.P_Fz)
    mod_error = self:module_error(self.P_Fz)
    -- (1-2beta)P_F z
    self.P_Fz:mul(1-2*self.beta)
    -- beta(I-P_Q) z
    self.z:add(-1,self.P_Qz)
    self.z:mul(self.beta)
    -- beta(I-P_Q) z + (1-2beta)P_F z
    self.z:add(self.P_Fz)
    self.P_Fz:mul(1/(1-2*self.beta))
    -- 2 beta P_Q P_F
    self:P_Q(self.P_Fz,self.P_Qz)
    overlap_error = self:overlap_error(self.P_Fz,self.P_Qz)
    plt:plot(self.O[1]:zfloat(),'object - it '..i)
    self.P_Qz:mul(2*self.beta)
    -- 2 beta P_Q P_F + beta(I-P_Q) z + (1-2beta)P_F z
    self.z:add(self.P_Qz)
    self:simpleRefineProbe()
    image_error = self:image_error()

    u.printf('iteration %03d: e_mod = %g    e_overlap = %g    e_image = %g',i,mod_error,overlap_error,image_error)
  end
end


local ngin = engine(pos,a,nmodes_probe,nmodes_object,solution)
print('outside')
pprint(ngin.z_tmp)
ngin:iterate(10)













local x= 1
