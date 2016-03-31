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

local path = '/home/philipp/drop/Public/'
local file = 'po.h5'



local par = {}
par.i = 50
par.DM_smooth_amplitude = 1
par.probe_change_start = 1

local f = hdf5.open(path..file,'r')

local a = f:read('/data_unshift'):all():cuda()
local pos = f:read('/scan_info/positions_int'):all():int()
pos:add(1)
-- local dx_spec = f:read('/scan_info/dx_spec')
-- local w = f:read('/fmask'):all():cuda()
local o_r = f:read('/or'):all():cuda()
local o_i = f:read('/oi'):all():cuda()
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
local probe = torch.ZCudaTensor.new(pr:size()):copyIm(pi):copyRe(pr)
local solution = torch.ZCudaTensor.new(o_r:size()):copyIm(o_i):copyRe(o_r)



o_r = nil
o_i = nil
pr = nil
pi = nil
collectgarbage()

-- frames
local K = a:size(1)
local M = a:size(2)
local MM = M*M
local nmodes_probe = 1
local nmodes_object = 1

function printMinMax(x,label)
  u.printf('%s re max,min = %g,%g   im max,min = %g,%g',label,x:re():max(),x:re():min(),x:im():max(),x:im():min())
end
local engine = classic.class('engine')



function engine:_init(pos,a,nmodes_probe,nmodes_object,solution,probe)
  self.solution = solution
  self.pos = pos
  self.a = a
  self.nmodes_probe = nmodes_probe
  self.nmodes_object = nmodes_object
  self.object_size = pos:max(1):add(torch.IntTensor({a:size(2),a:size(3)})):squeeze()
  self.O = torch.ZFloatTensor.new(nmodes_object,self.object_size[1]-1,self.object_size[2]-1):zcuda():fillRe(1)
  -- plt:plot(self.O[1]:zfloat(),'object')
  self.O_denom = torch.CudaTensor(self.O:size()):zero()
  self.P = torch.ZFloatTensor.new(nmodes_probe,a:size(2),a:size(3)):random():zcuda()

  if probe then
    self.P[1]:copy(probe)
  end

  local probe_size = self.P:size():totable()
  local support = znn.SupportMask(probe_size,probe_size[#probe_size]/3)
  self.P = support:forward(self.P)
  self.z = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))
  self.P_Qz = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))
  self.P_Fz = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))

  self.z_tmp = torch.ZCudaTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3))
  self.P_tmp = torch.ZCudaTensor.new(nmodes_probe,a:size(2),a:size(3))
  self.O_tmp = torch.ZCudaTensor.new(nmodes_object,self.object_size[1],self.object_size[2])
  self.P_real = torch.CudaTensor.new(nmodes_probe,a:size(2),a:size(3))
  self.a_tmp = torch.CudaTensor.new(1,a:size(2),a:size(3))
  self.fdev = torch.CudaTensor.new(1,a:size(2),a:size(3))
  self.O_real_tmp = torch.CudaTensor.new(1,self.object_size[1],self.object_size[2])
  -- plt:plot(self.P[1]:zfloat(),'self.P - init')
  -- pprint(self.z_tmp)

  self.K = a:size(1)
  self.M = a:size(2)
  self.MM = M*M

  self.O_views = {}
  self.O_denom_views = {}
  self.err_hist = {}

  self.beta = 1
  self.pbound = 0.25 * 0.5^2
  self.clip_max = 1e20
  self.clip_min = -1e20
  self.norm_a = self.a:sum()
  self.P_Mod =  function(x,abs,measured_abs)
                  x.THNN.P_Mod(x:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
                end
  self.InvSigma = function(x,sigma)
                    x.THNN.InvSigma(x:cdata(),x:cdata(),sigma)
                  end
  self.ClipMinMax = function(x,min,max)
                    x.THNN.ClipMinMax(x:cdata(),x:cdata(),min,max)
                  end

  self.update_probe = false

  for i=1,K do
    local slice = {{},{pos[i][1],pos[i][1]+M-1},{pos[i][2],pos[i][2]+M-1}}
    -- pprint(slice)
    self.O_views[i] = self.O[slice]
    self.O_denom_views[i] = self.O_denom[slice]
    local ov = self.O_views[i]:repeatTensor(self.nmodes_probe,1,1)
    self.z[i]:cmul(self.P,ov)
    -- plt:plot(z[i][1]:zfloat(),'z_'..i)
  end
  self:calculateO_denom()

  -- self.O_mask = self.O_denom:lt(1e-16)
  -- plt:plot(self.O_mask[1]:float(),'O_mask')
  plt:plot(self.O[1]:abs():float(),'self.O')
  printMinMax(self.O,'O')
  u.printMem()
  printMinMax(self.P,'probe')
end

-- recalculate (Q*Q)^-1
function engine:calculateO_denom()
  self.O_denom:fill(1e-3)
  local norm_P =  self.P_tmp:norm(self.P):sum(1):re()
  local tmp = self.P_real
  -- plt:plot(norm_P[1]:float(),'norm_P - calculateO_denom')
  for k=1,K do
    self.O_denom_views[k]:add(norm_P)
  end

  local abs_max = tmp:absZ(self.P):max()
  local sigma = abs_max * abs_max * 1e-4
  -- local sigma = 1e-6
  print('sigma = '..sigma)
  plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
  self.InvSigma(self.O_denom,sigma)

  -- local mask = self.O_denom:lt(1)
  -- self.O_denom:maskedFill(mask,1)
  plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
  -- printMinMax(self.O_denom,'calculateO_denom O_denom')
end

function engine:P_Q(z_in,z_out)
  local tmp = self.z_tmp[1]
  self.O:fillRe(1e-4):fillIm(0)
  -- printMinMax(z_in,'engine:P_Q z_in  ')
  for k=1,K do
    self.O_views[k]:add(tmp:conj(self.P):cmul(z_in[k]):sum(1))
  end
  -- printMinMax(self.O,   'self.O         before cmul')
  -- plt:plot(self.O[1]:abs():float(),'self.O')
  -- plt:plot(self.O_denom[1]:float():log(),'self.O_denom')
  self.O:cmul(self.O_denom)
  -- self.O:mul(1/self.O:max())
  -- plt:plot(self.O[1]:abs():float(),'self.O')
  -- printMinMax(self.O,   'self.O                    ')
  -- self.ClipMinMax(self.O,self.clip_min,self.clip_max)
  for k=1,K do
    local ov = self.O_views[k]:repeatTensor(self.nmodes_probe,1,1)
    z_out[k]:copy(ov):cmul(self.P)
  end
end

function engine:P_F(z_in,z_out)
  local abs = self.P_real
  local da = self.a_tmp
  local module_error = 0

  for k=1,K do
    z_out[k]:fftBatched(z_in[k])
    abs = abs:normZ(z_out[k]):sum(1)
    abs:sqrt()

    module_error = module_error + da:add(abs,-1,self.a[k]):sum()

    -- self.fdev:add(abs,-1,self.a[k]):pow(2)
    -- local power = self.fdev:sum()/self.fdev:nElement()
    --
    -- if power > self.pbound then
    abs:expandAs(z_out[k])
      -- local renorm = math.sqrt(self.pbound/power)
      -- u.printf('%i: power=%g, pbound = %g, renorm = %g',k,power,self.pbound, renorm)
      -- self.fdev:sqrt():mul(renorm):add(self.a[k])
    -- printMinMax(z_out[k],'engine:P_F abs  ')
    -- plt:plot(abs[1]:re():float():log(),'norm - engine:P_F')
    -- z_in[k]:fftBatched()
    -- plt:plot(a[k]:float():log(),'a[k] - engine:P_F')
    -- printMinMax(a[k],'engine:P_F a[k]    ')
    -- plt:plot(z_out[k][1]:abs():float():log(),'z_out[k] - before P_Mod')
    -- plt:plot(abs[1]:float():log(),'  abs[k] - before P_Mod')
    self.P_Mod(z_out[k],abs,self.a[k])
    z_out[k]:ifftBatched()
  end

  module_error = module_error / self.norm_a
  return math.abs(module_error)
end

function engine:simpleRefineProbe()
  -- simple ePIE update
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

end

function engine:overlap_error(z_in,z_out)
  return self.z_tmp:add(z_in,-1,z_out):norm():sum() / z_out:norm():sum()
end

function engine:image_error()
  local norm = self.O_real_tmp
  if self.solution then
    local c = self.O[1]:dot(self.solution)
    local phase_diff = c/zt.abs(c)
    print('phase difference: ',phase_diff)
    self.O:mul(phase_diff)
    norm:normZ(self.O_tmp:add(self.O,-1,self.solution))
    return norm:sum()/self.solution:norm():sum()
  end
end

function engine:iterate(steps)
  local mod_error, overlap_error, image_error
  printMinMax(self.solution,'solution      ')
  for i=1,steps do
    -- printMinMax(self.z,   'z                             ')
    self:P_Q(self.z,self.P_Qz)
    mod_error = self:P_F(self.z,self.P_Fz)
    -- printMinMax(self.P_Qz,'P_Q z                         ')
    -- printMinMax(self.P_Fz,'P_F z                         ')
    if i % 5 == 0 then
      plt:plot(self.O[1]:zfloat(),'object - it '..i)
    end
    -- (1-2beta)P_F z
    self.P_Fz:mul(1-2*self.beta)
    -- printMinMax(self.P_Fz, (1-2*self.beta)..' *P_F z                 ')
    -- beta(P_Q-I) z
    self.z:add(-1,self.P_Qz)
    self.z:mul(self.beta)
    -- printMinMax(self.z, self.beta..' *(P_Q-I) z                 ')
    -- beta(P_Q-I) z + (1-2beta)P_F z
    self.z:add(self.P_Fz)
    self.P_Fz:mul(1/(1-2*self.beta))
    -- printMinMax(self.z,   self.beta..' *(I-P_Q) z + '.. (1-2*self.beta) ..' P_F z')
    -- 2 beta P_Q P_F
    self:P_Q(self.P_Fz,self.P_Qz)
    overlap_error = self:overlap_error(self.P_Fz,self.P_Qz)

    self.P_Qz:mul(2*self.beta)
    -- printMinMax(self.P_Qz,   (2*self.beta)..' *P_Q P_F z              ')
    -- 2 beta P_Q P_F + beta(P_Q-I) z + (1-2beta)P_F z
    self.z:add(self.P_Qz)
    -- printMinMax(self.z,   'z_{i+1}                       ')
    if self.update_probe then
      self:simpleRefineProbe()
    end
    image_error = self:image_error()

    u.printf('iteration %03d: e_mod = %g    e_overlap = %g    e_image = %g',i,mod_error,overlap_error,image_error)
    print('--------------------------------------------------------------------------------')
  end
end


local ngin = engine(pos,a,nmodes_probe,nmodes_object,solution,probe)
-- print('outside')
-- pprint(ngin.z_tmp)
ngin:iterate(100)













local x= 1
