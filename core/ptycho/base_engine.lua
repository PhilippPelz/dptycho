local classic = require 'classic'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local zt = require "ztorch.fcomplex"
local pprint = require "pprint"
local engine = classic.class(...)

function printMinMax(x,label)
  u.printf('%s re max,min = %g,%g   im max,min = %g,%g',label,x:re():max(),x:re():min(),x:im():max(),x:im():min())
end

function engine:allocateBuffers(K,No,Np,M,Nx,Ny)
  self.O = torch.ZFloatTensor.new(No,Nx,Ny)
  self.O_denom = torch.CudaTensor(self.O:size())
  self.P = torch.ZCudaTensor.new(Np,M,M)

  self.z = torch.ZCudaTensor.new(K,Np,M,M)
  self.P_Qz = torch.ZCudaTensor.new(K,Np,M,M)
  self.P_Fz = torch.ZCudaTensor.new(K,Np,M,M)

  local P_Qz_storage = self.P_Qz:storage()
  local P_Fz_storage = self.P_Fz:storage()

  local z2_pointer = tonumber(torch.data(P_Qz_storage,true))
  local z3_pointer = tonumber(torch.data(P_Fz_storage,true))

  local P_Qz_storage_real = torch.CudaStorage(P_Qz_storage:size()*2,z2_pointer)
  local P_Fz_storage_real = torch.CudaStorage(P_Fz_storage:size()*2,z3_pointer)

  -- buffers for refine_probe, P_Qz and P_Fz available
  self.P_tmp = torch.ZCudaTensor.new(P_Qz_storage,1,{Np,M,M})
  self.P_tmp2 = torch.ZCudaTensor.new(P_Qz_storage,Np*M*M+1,{Np,M,M})
  self.a_tmp2 = torch.CudaTensor.new(P_Fz_storage_real,1,torch.LongStorage{1,M,M})
  self.a_tmp3 = torch.CudaTensor.new(P_Qz_storage_real,((2*Np*M*M*2)+1),torch.LongStorage{1,M,M})
  self.P_real2 = torch.CudaTensor.new(P_Qz_storage_real,((2*Np*M*M*2)+1*M*M+1),torch.LongStorage{Np,M,M})

  -- buffers used in P_F, points to the not used P_Qz
  self.P_real = torch.CudaTensor.new(P_Qz_storage_real,1,torch.LongStorage{Np,M,M})
  self.a_tmp = torch.CudaTensor.new(P_Qz_storage_real,Np*M*M+1,torch.LongStorage{1,M,M})

  -- not used atm
  self.fdev = torch.CudaTensor.new(P_Fz_storage_real,1,torch.LongStorage{1,M,M})

  -- used in image_error
  self.O_real_tmp = torch.CudaTensor.new(P_Qz_storage_real,1,torch.LongStorage{1,Nx,Ny})
  self.O_tmp = torch.ZCudaTensor.new(P_Fz_storage,1,{No,Nx,Ny})

  self.O = self.O:normal():zcuda()
  self.O_denom:zero()
  self.P:add(1)
end

function engine:_init(pos,a,nmodes_probe,nmodes_object,solution,probe,dpos)

  self.solution = solution
  self.pos = pos
  self.a = a

  self.K = a:size(1)
  self.M = a:size(2)
  self.MM = self.M*self.M

  self.Np = nmodes_probe
  self.No = nmodes_object

  local object_size = pos:max(1):add(torch.IntTensor({a:size(2),a:size(3)})):squeeze()
  self.Nx = object_size[1] - 1
  self.Ny = object_size[2] - 1

  self:allocateBuffers(self.K,self.No,self.Np,self.M,self.Nx,self.Ny)

  if probe then
    self.P[1]:copy(probe)
  end

  local probe_size = self.P:size():totable()
  local support = znn.SupportMask(probe_size,probe_size[#probe_size]/3)
  self.P = support:forward(self.P)
  -- self.P:mul(self.mean_counts*3)
  self.support = znn.SupportMask(probe_size,probe_size[#probe_size]*5/12)
  self.O_views = {}
  self.O_tmp_views = {}
  self.O_denom_views = {}
  self.err_hist = {}

  self.beta = 1
  self.pbound = 0.25 * 0.5^2

  self.P_Mod =  function(x,abs,measured_abs)
                  x.THNN.P_Mod(x:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
                end
  self.InvSigma = function(x,sigma)
                    x.THNN.InvSigma(x:cdata(),x:cdata(),sigma)
                  end
  self.ClipMinMax = function(x,min,max)
                    x.THNN.ClipMinMax(x:cdata(),x:cdata(),min,max)
                  end

  self.probe_update_start = 5
  self.update_probe = true
  pprint(self.O)
  for i=1,self.K do
    local slice = {{},{pos[i][1],pos[i][1]+self.M-1},{pos[i][2],pos[i][2]+self.M-1}}

    self.O_views[i] = self.O[slice]
    self.O_tmp_views[i] = self.O[slice]
    self.O_denom_views[i] = self.O_denom[slice]
  end

  self:calculateO_denom()
  self:update_z_from_O(self.z)
  self.norm_a = self.a:sum()
  self.mean_counts = self.norm_a/self.K/self.MM
  self.O_mask = self.O_denom:lt(1e-8)
  -- plt:plot(self.O_mask[1]:float(),'O_mask')
  -- plt:plot(self.O[1]:abs():float(),'self.O')
  -- printMinMax(self.O,'O')
  u.printMem()
  -- plt:plot(self.P[1]:zfloat(),'self.P - init')
  -- printMinMax(self.P,'probe')
end

-- recalculate (Q*Q)^-1
function engine:calculateO_denom()
  self.O_denom:fill(1e-3)
  local norm_P =  self.P_real:normZ(self.P):sum(1)
  local tmp = self.P_real
  -- plt:plot(norm_P[1]:float(),'norm_P - calculateO_denom')
  for k=1,self.K do
    self.O_denom_views[k]:add(norm_P)
  end

  local abs_max = tmp:absZ(self.P):max()
  local sigma = abs_max * abs_max * 1e-4
  -- local sigma = 1e-6
  -- print('sigma = '..sigma)
  -- plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
  self.InvSigma(self.O_denom,sigma)
end

function engine:merge_frames(z, mul_merge, merge_memory, merge_memory_views)
  local tmp = self.P_tmp
  merge_memory:fillRe(1e-4):fillIm(0)
  for k, view in ipairs(merge_memory_views) do
    view:add(tmp:conj(mul_merge):cmul(z[k]):sum(1))
  end
  self.O:cmul(self.O_denom)
end

function engine:update_frames(z,mul_split,merge_memory_views)
  for k, view in ipairs(merge_memory_views) do
    local ov = view:repeatTensor(self.Np,1,1)
    z[k]:copy(ov):cmul(mul_split)
  end
end

function engine:merge_and_split_pair(i,j,mul_merge,mul_split,result)
  local tmp = self.P_tmp
  self.O_tmp:zero()
  self.O_tmp_views[i]:add(tmp:conj(mul_merge):cmul(self.z[i]):sum(1))
  self.O_tmp:cmul(self.O_denom)
  local ov = self.O_tmp_views[j]:repeatTensor(self.Np,1,1)
  result:cmul(mul_split,ov)
  return result
end

function engine:split_single(i,P_out,mul_split)
    local ov = self.O_views[i]:repeatTensor(self.Np,1,1)
    P_out:copy(ov):cmul(mul_split)
    return P_out
end

function engine:update_O(z_in)
  self:merge_frames(z_in,self.P,self.O,self.O_views)
end

function engine:update_z_from_O(z_out)
  self:update_frames(z_out,self.P,self.O_views)
end

function engine:P_Q(z_in,z_out)
  self:merge_frames(z_in,self.P,self.O,self.O_views)
  self:update_frames(z_out,self.P,self.O_views)
end

function engine:P_F(z_in,z_out)
  local abs = self.P_real
  local da = self.a_tmp
  local module_error = 0

  for k=1,self.K do
    z_out[k]:fftBatched(z_in[k])
    abs = abs:normZ(z_out[k]):sum(1)
    abs:sqrt()
    module_error = module_error + da:add(abs,-1,self.a[k]):sum()
    abs:expandAs(z_out[k])
    self.P_Mod(z_out[k],abs,self.a[k])
    z_out[k]:ifftBatched()
  end

  module_error = module_error / self.norm_a
  return math.abs(module_error)
end

function engine:refine_probe()
  local new_probe = self.P_tmp
  local oview_conj = self.P_tmp2
  local dP = self.P_tmp2
  local dP_abs = self.P_real2
  local denom = self.a_tmp3
  local tmp = self.a_tmp2

  new_probe:zero()
  denom:zero()

  for k, view in ipairs(self.O_views) do
    local ovk = view:repeatTensor(self.Np,1,1)
    oview_conj:conj(ovk)
    new_probe:add(oview_conj:cmul(self.z[k]))
    denom:add(tmp:normZ(view))
  end

  new_probe:cdiv(denom)
  new_probe = self.support:forward(new_probe)

  local probe_change = dP_abs:absZ(dP:add(new_probe,-1,self.P)):sum()
  self.P:copy(new_probe)

  self:calculateO_denom()
  return probe_change
end

function engine:overlap_error(z_in,z_out)
  local result = self.P_Fz
  return result:add(z_in,-1,z_out):norm():sum() / z_out:norm():sum()
end

function engine:image_error()
  local norm = self.O_real_tmp
  if self.solution then
    local c = self.O[1]:dot(self.solution)
    -- print(c)
    local phase_diff = c/zt.abs(c)
    print('phase difference: ',phase_diff)
    self.O_tmp:mul(self.O,phase_diff)
    norm:normZ(self.O_tmp:add(-1,self.solution)):cmul(self.O_mask)
    return norm:sum()/self.solution:norm():sum()
  end
end

engine:mustHave("iterate")

return engine
