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

-- frames
local K = a:size(1)
local M = a:size(2)
local MM = M*M
local nmodes_probe = 1
local nmodes_object = 1

local function fw(x)
  return x:fftBatched():mul(1/math.sqrt(MM))
end
local function bw(x)
  return x:ifftBatched():mul(math.sqrt(MM))
end

pprint(a)
pprint(w)
pprint(pos)

local max_power = a:clone():pow(2):sum(2):sum(3):max()
local total_measurements = w:clone():sum(1):sum(2):sum(3):squeeze()
pprint(total_measurements)
local pbound = .25 * .5 * .5
u.printf('Computed pbound is %g',pbound)
local osize = pos:max(1):add(torch.Tensor({a:size(2),a:size(3)})):squeeze()


local O = torch.ZFloatTensor.new(nmodes_object,osize[1],osize[2]):zcuda():fillRe(1)
local P = torch.ZFloatTensor.new(nmodes_probe,a:size(2),a:size(3)):random():zcuda()
local z = torch.ZFloatTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3)):zcuda()
local new_z = z:clone()
local tmp_z = z:clone()
local tmp_p = P:clone()
local tmp_p_re = torch.CudaTensor(P:size())

local probe_size = P:size():totable()
local support = znn.SupportMask(probe_size,probe_size[#probe_size]/3)
P = support:forward(P)

-- plt:plot(P[1]:zfloat(),probe)

pprint(O)
pprint(P)

local nO = O:clone():zero()
local O_denom = O:clone():zero()
local nP = P:clone():zero()
local P_denom = torch.CudaTensor(P:size()):zero()

-- create views
local O_views = {}
local O_denom_views = {}
local err_hist = {}

for i=1,K do
  local slice = {{},{pos[i][1],pos[i][1]+M-1},{pos[i][2],pos[i][2]+M-1}}
  -- pprint(slice)
  O_views[i] = O[slice]
  O_denom_views[i] = O_denom[slice]
  z[i]:cmul(P,O_views[i]:expandAs(P))
  -- plt:plot(z[i][1]:zfloat(),'z_'..i)
end

u.printMem()

local engine = classic.class('engine')

function engine:_init(pos,a,nmodes_probe,nmodes_object)
  self.pos = pos
  self.a = a
  self.nmodes_probe = nmodes_probe
  self.nmodes_object = nmodes_object
  self.object_size = pos:max(1):add(torch.Tensor({a:size(2),a:size(3)})):squeeze()
  self.O = torch.ZFloatTensor.new(nmodes_object,self.object_size[1],self.object_size[2]):zcuda():fillRe(1)
  self.O_denom = torch.CudaTensor(O:size()):zero()
  self.P = torch.ZFloatTensor.new(nmodes_probe,a:size(2),a:size(3)):random():zcuda()
  self.z = torch.ZFloatTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3)):zcuda()
  self.P_Qz = torch.ZFloatTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3)):zcuda()
  self.P_Fz = torch.ZFloatTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3)):zcuda()

  self.z_tmp = torch.ZFloatTensor.new(a:size(1),nmodes_probe,a:size(2),a:size(3)):zcuda()
  self.P_tmp = torch.ZFloatTensor.new(nmodes_probe,a:size(2),a:size(3)):zcuda()

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
    self.O_views[i] = O[slice]
    self.O_denom_views[i] = O_denom[slice]
    self.z[i]:cmul(P,O_views[i]:expandAs(P))
    -- plt:plot(z[i][1]:zfloat(),'z_'..i)
  end
  self:calculateO_denom()
end

-- recalculate (Q*Q)^-1
function engine:calculateO_denom()
  self.O_denom:fill(1e-4)
  local norm_P =  self.z_tmp[2]:norm(self.P):sum(1)
  for k=1,K do
    self.O_denom_views[k]:add(norm_P)
  end
end

function engine:P_Q(z_in,z_out)
  local tmp = self.z_tmp[1]
  self.O:zero()
  for k=1,K do
    self.O_views[k]:add(tmp:conj(P):cmul(z_in[k]):sum(1))
  end
  self.O:cdiv(self.O_denom)
  for k=1,K do
    z_out[k]:copy(self.O_views[k]):cmul(self,P)
  end
end

function engine:P_F(z_in,z_out)
  -- sum over probe intensities --> K x 1 x M x M -expand-> K x mprobe x M x M
  self.z_tmp:norm(z_in):sum(2):sqrt():expandAs(z_out)
  for k=1,K do
    z_out[k]:fftBatched(z_in[k])
    z_out[k]:cmul(a[k]:view(1,self.M,self.M):expandAs(z_out[k]))
    z_out[k]:cdiv(self.z_tmp[k])
    z_out[k]:ifftBatched()
  end
end

function engine:simpleRefineProbe()
  local oview_conj = self.P_tmp[1]
  local sub = self.P_tmp
  local denom = self.P_tmp
  for k=1,K do
    oview_conj:conj(O_views[k]):view(1,self.M,self.M):expandAs(self.z_tmp[k])
    self.z_tmp[k]:cmul(self.P_Fz[k],oview_conj)
    sub:norm(O_views[k]:view(1,self.M,self.M):expandAs(sub)):cmul(self.P)
    self.z_tmp[k]:add(-1,sub)
    denom:norm(O_views[k]:view(1,self.M,self.M):expandAs(sub))
    denom:add(denom:mean()*1e-4)
    self.z_tmp[k]:cdiv(denom)
  end
  self.z_tmp = self.z_tmp:sum(1)
  self.P:add(self.z_tmp)

  self:calculateO_denom()
end

function engine:module_error(z)
  -- K x 1 x M x M
  self.z_tmp:norm(z):sum(2):sqrt()
  return self.z_tmp:add(-1,self.a):sum()/self.norm_a
end

function engine:overlap_error(z_in,z_out)
  return self.z_tmp:add(z_in,-1,z_out):norm():sum() / z_out:norm():sum()
end

function engine:image_error()

end

function engine:iterate(steps)
  local mod_error, overlap_error
  for i=1,steps do
    self.P_Q(self.z,self.P_Qz)
    self.P_F(self.z,self.P_Fz)
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
    self.P_Q(self.P_Fz,self.P_Qz)
    overlap_error = self:overlap_error(self.P_Fz,self.P_Qz)
    self.P_Qz:mul(2*self.beta)
    -- 2 beta P_Q P_F + beta(I-P_Q) z + (1-2beta)P_F z
    self.z:add(self.P_Qz)
    self.simpleRefineProbe()

  end
end

-- for i=1,par.i do
--   u.printf('iteration # %d of %d',i,par.i)
--   print(' - projection 1: overlap constraint - ')
--
--   for j = 1,10 do
--     local tmp, tmp_re = tmp_z[1], tmp_p_re[1]
--     -- object update
--     O:zero()
--     O_denom:fillRe(.001):fillIm(0)
--
--     for k=1,K do
--       O_views[k]:add(tmp:conj(P):cmul(z[k]):sum(1))
--       O_denom_views[k]:add(tmp:abs(P):pow(2))
--     end
--     O:cdiv(O_denom)
--     if j == 9 then plt:plot(O[1]:zfloat(),'object after update') end
--
--     -- probe update
--     if i > par.probe_change_start then
--       for k=1,K do
--         nP:add(tmp:conj(O_views[k]:expandAs(P)):cmul(z[k]))
--         P_denom:add(tmp_re:normZ(O_views[k]:expandAs(P)))
--       end
--       nP:cdiv(P_denom)
--
--       local probe_change = tmp:add(P,-1,nP):abs():sum()
--       u.printf('Change in probe: %f',probe_change)
--
--       P:copy(nP)
--     end
--   end
--
--   print(' - projection 2: Fourier modulus constraint - ')
--   -- update z[i]
--   local update_counter, er2 = 0, 0
--   for i=1,K do
--     local tmp, tmp_re = tmp_p[1], tmp_p_re[1]
--
--     new_z[i]:cmul(P,O_views[i]:expandAs(P))
--     tmp_z[i]:copy(new_z[i])
--     local f = fw(tmp_z[i]:mul(2):add(-1,z[i]))
--     local af = f:norm():sum(1):sqrt()
--     -- plt:plot(af[1]:float():log(),'af')
--     -- plt:plot(a[i]:float():log(),'a[i]')
--     local fdev = tmp_re:add(af,-1,a[i]:viewAs(af))
--     -- plt:plot(fdev[1]:float():log(),'fdev')
--     local fdev2 = torch.cmul(fdev,w[i]):pow(2)
--     local power = fdev2:mean()/w[i]:mean()
--     u.printf('power , pbound = %g , %g',power,pbound)
--     local df
--     if power > pbound then
--       update_counter = update_counter + 1
--       local renorm = math.sqrt(pbound / power)
--       -- fm = (1-w) + w*(a[i]+fdev*renorm)/(af+1e-10)
--       local fm = w[i]:ne(1):add(tmp_re:mul(fdev,renorm):add(a[i]):cmul(w[i]):cdiv(af:add(1e-10)))
--       -- plt:plot(fm:float(),'fm')
--       local update_z = bw(f:cmul(fm))
--       df = update_z:add(-1,new_z[i])
--     else
--       df = tmp_z[i]:add(new_z[i],-1,z[i])
--     end
--     -- plt:plot(df[1]:zfloat(),'df')
--     z[i]:add(df)
--     -- plt:plot(z[i][1]:zfloat(),'z[i]')
--     er2 = er2 + df:norm():sum()
--   end
--   if i % 1 == 0 then
--     plt:plot(P[1]:zfloat(),'probe')
--     plt:plot(O[1]:zfloat(),'object')
--   end
--   u.printf('%d updates from fourier modulus constraint. Fraction of %.2f',update_counter,update_counter/K)
--
--   err_hist[#err_hist+1] = math.sqrt(er2/max_power/(K*nmodes_probe*nmodes_object))
--   u.printf('Error: %12.3f',err_hist[#err_hist])
-- end















local x= 1
