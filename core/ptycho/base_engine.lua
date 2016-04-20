local classic = require 'classic'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local zt = require "ztorch.fcomplex"
local pprint = require "pprint"
local stats = require 'dptycho.util.stats'
local params = require "dptycho.core.ptycho.params"


-- _init
-- allocateBuffers
-- update_views
-- replace_default_params
-- update_iteration_dependent_parameters
-- generate_data
-- calculateO_denom
-- merge_frames
-- update_O
-- update_frames
-- update_z_from_O
-- P_Q
-- P_F
-- refine_probe
-- overlap_error
-- image_error
-- probe_error
-- mustHave("iterate")

require 'hdf5'
local engine = classic.class(...)
--pos,a,nmodes_probe,nmodes_object,solution,probe,dpos,fmask
function engine:_init(par)

  local par = self:replace_default_params(par)
  -- print(par.pos)
  local min = par.pos:min(1)
  -- pprint(min)
  self.pos = par.pos:add(-1,min:expandAs(par.pos)):add(1)
  -- pprint(self.pos:min(1))
  self.dpos = par.dpos
  self.a = par.a
  self.fm = par.fmask:expandAs(self.a)

  self.K = par.a:size(1)
  self.M = par.a:size(2)
  self.MM = self.M*self.M

  self.Np = par.nmodes_probe
  self.No = par.nmodes_object
  self.fourier_relax_factor = par.fourier_relax_factor
  self.plot_every = par.plot_every
  self.plot_start = par.plot_start
  self.probe_update_start = par.probe_update_start
  self.position_refinement_start = par.position_refinement_start
  self.position_refinement_every = par.position_refinement_every
  self.P_Q_iterations = par.P_Q_iterations
  self.beta = par.beta

  self.dpos_solution = par.dpos_solution

  local object_size = par.pos:max(1):add(torch.IntTensor({par.a:size(2),par.a:size(3)})):squeeze()
  self.Nx = object_size[1] - 1
  self.Ny = object_size[2] - 1
  -- pprint( pos:max(1))
  self.solution = par.solution[{{1,self.Nx},{1,self.Ny}}]
  -- pprint(object_size)
  -- pprint(self.solution)

  self:allocateBuffers(self.K,self.No,self.Np,self.M,self.Nx,self.Ny)

  if par.probe then
    self.probe_solution = par.probe
    -- self.P[1]:copy(par.probe)
    -- self.O[1]:copy(self.solution)
  end

  if par.copy_solution then
    self.O[1]:copy(self.solution)
    self.P[1]:copy(self.probe_solution)
  end

  local probe_size = self.P:size():totable()
  local support = znn.SupportMask(probe_size,probe_size[#probe_size]/4)
  self.P = support:forward(self.P)
  -- plt:plotReIm(self.P[1]:zfloat(),'probe - it '..0)
  -- self.P:mul(self.mean_counts*3)
  self.support = znn.SupportMask(probe_size,probe_size[#probe_size]*5/12)
  self.O_views = {}
  self.O_tmp_PF_views = {}
  self.O_tmp_PQ_views = {}
  self.O_denom_views = {}
  self.err_hist = {}

  self.max_power = self.a_tmp_real_PQstore:cmul(self.a,self.fm):pow(2):max()
  self.power_threshold = 0.25 * self.fourier_relax_factor^2 * self.max_power / self.MM

  self.P_Mod =  function(x,abs,measured_abs)
                  x.THNN.P_Mod(x:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
                end
  self.P_Mod_renorm =  function(x,fm,fdev,a,af,renorm)
                  x.THNN.P_Mod_renorm(x:cdata(),fm:cdata(),fdev:cdata(),a:cdata(),af:cdata(),renorm)
                end
  self.InvSigma = function(x,sigma)
                    x.THNN.InvSigma(x:cdata(),x:cdata(),sigma)
                  end
  self.ClipMinMax = function(x,min,max)
                    x.THNN.ClipMinMax(x:cdata(),x:cdata(),min,max)
                  end

  self.update_positions = false
  self.update_probe = false
  self.object_inertia = par.object_inertia * self.K
  self.probe_inertia = par.probe_inertia * self.Np * self.No * self.K

  self.i = 0
  self.j = 0
  self.iterations = 1

  self:update_views()
  self:calculateO_denom()
  self:update_z_from_O(self.z)

  self.norm_a = self.a_tmp_real_PQstore:pow(self.a,2):cmul(self.fm):sum()
  local beam_fluence = self.norm_a/self.K
  --
  local P_norm = self.P_tmp1_real_PQstore:normZ(self.P)
  local P_fluence = P_norm:sum()

  -- local P_extent = P_norm:gt(1e-1):sum()
  --
  -- self.P:mul(beam_fluence/P_fluence/self.MM)
  -- print(P_fluence,P_extent,beam_fluence,beam_fluence/P_fluence/P_extent)
  u.printf('power_threshold is %g',self.power_threshold)
  u.printMem()
end

function engine:update_views()
  for i=1,self.K do
    local slice = {{},{self.pos[i][1],self.pos[i][1]+self.M-1},{self.pos[i][2],self.pos[i][2]+self.M-1}}
    -- pprint(slice)
    self.O_views[i] = self.O[slice]
    self.O_tmp_PF_views[i] = self.O_tmp_PFstore[slice]
    self.O_tmp_PQ_views[i] = self.O_tmp_PQstore[slice]
    self.O_denom_views[i] = self.O_denom[slice]
  end
end

function engine:replace_default_params(p)
  local par = params.DEFAULT_PARAMS()
  for k,v in pairs(p) do
    -- print(k,v)
    par[k] = v
  end
  return par
end

-- total memory requirements:
-- 1 x object complex
-- 1 x object real
-- 1 x probe real
-- 3 x z
function engine:allocateBuffers(K,No,Np,M,Nx,Ny)
  self.O = torch.ZCudaTensor.new(No,Nx,Ny)
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

  local P_Qstorage_offset, P_Fstorage_offset = 1,1

  -- buffers in P_Qz_storage

  self.P_tmp1_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_PQstore:nElement() + 1
  self.P_tmp2_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp2_PQstore:nElement() + 1
  self.P_tmp3_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_PQstore:nElement() + 1
  self.P_tmp4_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp4_PQstore:nElement() + 1
  self.P_tmp5_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp5_PQstore:nElement() + 1
  self.P_tmp6_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp6_PQstore:nElement() + 1
  self.P_tmp7_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp7_PQstore:nElement() + 1

  self.O_tmp_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{No,Nx,Ny})
  P_Qstorage_offset = P_Qstorage_offset + self.O_tmp_PQstore:nElement()

  -- offset for real arrays
  P_Qstorage_offset = P_Qstorage_offset*2 + 1
  self.P_tmp1_real_PQstore = torch.CudaTensor.new(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_real_PQstore:nElement() + 1
  self.P_tmp2_real_PQstore = torch.CudaTensor.new(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp2_real_PQstore:nElement() + 1
  self.P_tmp3_real_PQstore = torch.CudaTensor.new(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_real_PQstore:nElement() + 1
  self.a_tmp_real_PQstore = torch.CudaTensor.new(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp_real_PQstore:nElement() + 1
  self.a_tmp2_real_PQstore = torch.CudaTensor.new(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp2_real_PQstore:nElement() + 1

  -- buffers in P_Fz_storage

  self.P_tmp1_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_PFstore:nElement() + 1
  self.P_tmp2_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_PFstore:nElement() + 1
  self.P_tmp3_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_PFstore:nElement() + 1
  self.P_tmp4_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp4_PFstore:nElement() + 1
  self.P_tmp5_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp5_PFstore:nElement() + 1
  self.P_tmp6_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp6_PFstore:nElement() + 1
  self.P_tmp7_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp7_PFstore:nElement() + 1
  self.P_tmp8_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp8_PFstore:nElement() + 1
  self.P_tmp9_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp9_PFstore:nElement() + 1
  self.P_tmp10_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp10_PFstore:nElement() + 1
  self.P_tmp11_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp11_PFstore:nElement() + 1

  self.O_tmp_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_PFstore:nElement()

  -- offset for real arrays
  P_Fstorage_offset = P_Fstorage_offset*2 + 1
  self.P_tmp1_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_real_PFstore:nElement() + 1
  self.P_tmp2_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_real_PFstore:nElement() + 1
  self.P_tmp3_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_real_PFstore:nElement() + 1
  self.a_tmp_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{1,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.a_tmp_real_PFstore:nElement() + 1
  self.O_tmp_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{No,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_real_PFstore:nElement() + 1

  local re = stats.truncnorm({No,Nx,Ny},0,1,0.1,0.1):cuda()
  local im = stats.truncnorm({No,Nx,Ny},0,1,0.1,0.1):cuda()

  -- plt:plot(re[1]:float())
  -- plt:plot(im[1]:float())

  self.O = self.O:zero():copyRe(re):add(1+0i)
  self.O_denom:zero()
  self.P:zero():add(300+0i)
end

function engine:allocateBuffers2(K,No,Np,M,Nx,Ny)
  self.O = torch.ZCudaTensor.new(No,Nx,Ny)
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

  local P_Qstorage_offset, P_Fstorage_offset = 0,0

  -- buffers for refine_probe, P_Qz and P_Fz available
  -- self.P_tmp_PQstore = torch.ZCudaTensor.new(P_Qz_storage,1,{Np,M,M})
  -- self.P_tmp2_PQstore = torch.ZCudaTensor.new(P_Qz_storage,Np*M*M+1,{Np,M,M})
  -- self.P_tmp2_PFstore = torch.CudaTensor.new(P_Fz_storage_real,1,torch.LongStorage{1,M,M})
  -- self.P_tmp3_PFstore = torch.CudaTensor.new(P_Qz_storage_real,((2*Np*M*M*2)+1),torch.LongStorage{1,M,M})
  -- self.P_tmp_real_PQstore = torch.CudaTensor.new(P_Qz_storage_real,((2*Np*M*M*2)+1*M*M+1),torch.LongStorage{Np,M,M})
  -- pprint(self.P_tmp3_PFstore)
  self.P_tmp_PQstore = torch.ZCudaTensor.new(Np,M,M)
  self.P_tmp2_PQstore = torch.ZCudaTensor.new(Np,M,M)
  self.P_tmp3_PQstore = torch.ZCudaTensor.new(Np,M,M)

  self.P_tmp2_PFstore = torch.CudaTensor.new(Np,M,M)
  self.P_tmp3_PFstore = torch.CudaTensor.new(Np,M,M)
  self.a_tmp4_PQstore = torch.CudaTensor.new(1,M,M)
  self.P_tmp_real_PQstore = torch.CudaTensor.new(Np,M,M)

  -- update_frames_shifted buffer
  self.P_tmp_PF_store = torch.ZCudaTensor.new(P_Fz_storage,1,{Np,M,M})

  -- buffers used in P_F, points to the not used P_Qz
  -- self.P_real = torch.CudaTensor.new(P_Qz_storage_real,1,torch.LongStorage{Np,M,M})
  -- self.a_tmp = torch.CudaTensor.new(P_Qz_storage_real,Np*M*M+1,torch.LongStorage{1,M,M})

  self.P_real = torch.CudaTensor.new(Np,M,M)
  self.a_tmp = torch.CudaTensor.new(1,M,M)
  -- not used atm
  self.fdev = torch.CudaTensor.new(1,M,M)

  -- used in image_error
  -- self.O_real_tmp = torch.CudaTensor.new(P_Qz_storage_real,1,torch.LongStorage{1,Nx,Ny})
  -- self.O_tmp = torch.ZCudaTensor.new(P_Fz_storage,1,{No,Nx,Ny})

  self.O_real_tmp = torch.CudaTensor.new(1,Nx,Ny)
  self.O_tmp = torch.ZCudaTensor.new(No,Nx,Ny)

  local re = stats.truncnorm({No,Nx,Ny},0,1,0.1,0.1):cuda()
  local im = stats.truncnorm({No,Nx,Ny},0,1,0.1,0.1):cuda()

  -- plt:plot(re[1]:float())
  -- plt:plot(im[1]:float())

  self.O = self.O:zero():copyRe(re):add(1+0i)
  self.O_denom:zero()
  self.P:zero():add(300+0i)
end

function engine:update_iteration_dependent_parameters(it)
  self.i = it
  self.update_probe = it >= self.probe_update_start
  self.update_positions = it >= self.position_refinement_start and it % self.position_refinement_every == 0
  self.do_plot = it % self.plot_every == 0 and it > self.plot_start
end

function engine:generate_data(filename)
  self.P[1]:copy(self.probe_solution)
  self.O[1]:copy(self.solution)
  -- self:calculateO_denom2()
  self:update_frames(self.z,self.P,self.O_views)
  for k=1,self.K do
    -- plt:plotcompare({self.z[k][1]:re():float(),self.z[k][1]:im():float()},'self.z[k]')
    local a_m = self.z[k]:fftBatched():sum(1)[1]:abs()
    -- plt:plot(a_m:float():log(),'diff')
    self.a[k]:copy(a_m)
  end
  local f = hdf5.open(filename,'w')
  f:write('/pr',self.P[1]:re():float())
  f:write('/pi',self.P[1]:im():float())
  f:write('/o_r',self.solution:re():float())
  f:write('/o_i',self.solution:im():float())
  plt:plot(self.P[1]:re():float(),'P')
  plt:plot(self.O[1]:re():float(),'O')
  f:write('/scan_info/positions_int',self.pos)
  -- :add(self.dpos)
  f:write('/scan_info/positions',self.pos:float())
  f:write('/scan_info/dpos',self.dpos)
  f:write('/data_unshift',self.a:float())
  f:close()
end

-- recalculate (Q*Q)^-1
function engine:calculateO_denom()
  self.O_denom:fill(self.object_inertia*self.K)
  local norm_P =  self.P_tmp2_real_PQstore:normZ(self.P):sum(1)
  local tmp = self.P_tmp2_real_PQstore
  -- plt:plot(norm_P[1]:float(),'norm_P - calculateO_denom')
  for k=1,self.K do
    self.O_denom_views[k]:add(norm_P)
  end
  local fact_start, fact_end = 1e-3, 1e-6

  local fact = fact_start-(self.i/self.iterations)*(fact_start-fact_end)
  local abs_max = tmp:absZ(self.P):max()
  local sigma = abs_max * abs_max * fact
  -- local sigma = 1e-8
  -- print('sigma = '..sigma)
  -- plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
  self.InvSigma(self.O_denom,sigma)
  self.O_mask = self.O_denom:lt(1e-3)
  -- plt:plot(self.O_denom[1]:float():log(),'O_denom')
  -- plt:plot(self.O_mask[1]:float(),'O_mask')
end

-- P_Qz, P_Fz free to use
function engine:merge_frames(z, mul_merge, merge_memory, merge_memory_views)
  local tmp = self.P_tmp1_PQstore
  merge_memory:mul(self.object_inertia)
  for k, view in ipairs(merge_memory_views) do
    view:add(tmp:conj(mul_merge):cmul(z[k]):sum(1))
  end
  self.O:cmul(self.O_denom)
end


function engine:update_O(z_in)
  self:merge_frames(z_in,self.P,self.O,self.O_views)
end

-- P_Fz free
function engine:update_frames(z,mul_split,merge_memory_views)
  for k, view in ipairs(merge_memory_views) do
    local ov = view:repeatTensor(self.Np,1,1)
    z[k]:copy(ov):cmul(mul_split)
  end
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
  local abs = self.P_tmp1_real_PQstore
  local da = self.a_tmp_real_PQstore
  local fdev = self.a_tmp2_real_PQstore
  local module_error, err_fmag, renorm  = 0, 0, 0
  local mod_updates = 0
  for k=1,self.K do
    -- plt:plot(z_in[k][1]:abs():float():log(),'z_out[k][1]')
    z_out[k]:fftBatched(z_in[k])
    abs = abs:normZ(z_out[k]):sum(1)
    abs:sqrt()
    -- plt:plot(abs[1]:float():log(),'abs')
    fdev:add(abs,-1,self.a[k])
    da:abs(fdev):pow(2):cmul(self.fm[k])
    err_fmag = da:cmul(self.fm[k]):sum()
    module_error = module_error + err_fmag
    -- plt:plot(da[1]:float(),'da')
    if err_fmag > self.power_threshold then
      abs:expandAs(z_out[k])
      fdev:expandAs(z_out[k])
      renorm = math.sqrt(self.power_threshold/err_fmag)
      self.P_Mod_renorm(z_out[k],self.fm[k],fdev,self.a[k],abs,renorm)
      -- self.P_Mod(z_out[k],abs,self.a[k])
      mod_updates = mod_updates + 1
    end
    z_out[k]:ifftBatched()
  end

  -- u.printf('%d/%d modulus updates', mod_updates, self.K)

  module_error = module_error / self.norm_a
  return module_error, mod_updates
end

function engine:refine_probe()
  local new_probe = self.P_tmp1_PQstore
  local oview_conj = self.P_tmp2_PQstore

  local dP = self.P_tmp3_PQstore
  local dP_abs = self.P_tmp3_real_PQstore

  local denom = self.P_tmp1_real_PQstore
  local tmp = self.P_tmp3_real_PQstore
  new_probe:mul(self.P,self.probe_inertia)
  denom:fill(self.probe_inertia)

  for k, view in ipairs(self.O_views) do
    local ovk = view:repeatTensor(self.Np,1,1)
    oview_conj:conj(ovk)
    new_probe:add(oview_conj:cmul(self.z[k]))
    denom:add(tmp:normZ(view))
  end

  new_probe:cdiv(denom)
  new_probe = self.support:forward(new_probe)

  local probe_change = dP_abs:normZ(dP:add(new_probe,-1,self.P)):sum()
  local P_norm = dP_abs:normZ(self.P):sum()
  self.P:copy(new_probe)

  self:calculateO_denom()
  return math.sqrt(probe_change/P_norm/self.Np)
end

function engine:overlap_error(z_in,z_out)
  local result = self.P_Fz
  return result:add(z_in,-1,z_out):norm():sum() / z_out:norm():sum()
end

function engine:image_error()
  local norm = self.O_tmp_real_PFstore
  local O_res = self.O_tmp_PFstore
  if self.solution then
    -- pprint(self.O[1])
    -- pprint(self.solution)
    local c = self.O[1]:dot(self.solution)
    -- print(c)
    local phase_diff = c/zt.abs(c)
    --:cmul(self.O_mask)
    -- print('phase difference: ',phase_diff)
    O_res:mul(self.O,phase_diff)
    norm:normZ(O_res:add(-1,self.solution):cmul(self.O_mask))
    local norm1 = norm:sum()/self.solution:norm():sum()
    O_res:mul(self.O,phase_diff)
    norm:normZ(O_res:add(self.solution):cmul(self.O_mask))
    local norm2 = norm:sum()/self.solution:norm():sum()
    return math.min(norm1,norm2)
  end
end

-- free buffer: P_Fz
function engine:probe_error()
  local norm = self.P_tmp1_real_PFstore
  local P_corr = self.P_tmp2_PFstore
  if self.solution then
    -- pprint(self.O[1])
    -- pprint(self.solution)
    local c = self.P[1]:dot(self.probe_solution)
    -- print(c)
    local phase_diff = c/zt.abs(c)
    --:cmul(self.O_mask)
    -- print('phase difference: ',phase_diff)
    P_corr:mul(self.P,phase_diff)
    norm:normZ(P_corr:add(-1,self.probe_solution))
    local norm1 = norm:sum()/self.probe_solution:norm():sum()
    P_corr:mul(self.P,phase_diff)
    norm:normZ(P_corr:add(self.probe_solution))
    local norm2 = norm:sum()/self.probe_solution:norm():sum()
    return math.min(norm1,norm2)
  end
end

engine:mustHave("iterate")

return engine
