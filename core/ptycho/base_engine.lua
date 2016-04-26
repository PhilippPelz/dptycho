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

  self.a_exp = self.a:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
  self.fm_exp = self.fm:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)

  self.dpos_solution = par.dpos_solution

  -- dimensions used for probe and object modes in z[k]
  self.O_dim = 1
  self.P_dim = 2

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
  end

  local probe_size = self.P:size():totable()
  local support = znn.SupportMask(probe_size,probe_size[#probe_size]/4)

  local re = stats.truncnorm({self.No,self.Nx,self.Ny},0,1,0.1,0.1):cuda()
  local Pre = stats.truncnorm({self.M,self.M},0,1,0.1,0.1):cuda()
  self.O = self.O:zero():copyRe(re):add(1+0i)
  self.O_denom:zero()
  self.P:zero()
  self.P[1]:add(300+0i)
  for i=2,self.Np do
    self.P[i]:copyRe(Pre)
  end
  self.P = support:forward(self.P)

  self.O_views = {}
  self.O_tmp_PF_views = {}
  self.O_tmp_PQ_views = {}
  self.O_denom_views = {}
  self.err_hist = {}

  self.max_power = self.a_tmp2_real_PQstore:cmul(self.a,self.fm):pow(2):max()
  self.power_threshold = 0.25 * self.fourier_relax_factor^2 * self.max_power / self.MM
  self.update_positions = false
  self.update_probe = false
  self.object_inertia = par.object_inertia * self.K
  self.probe_inertia = par.probe_inertia * self.Np * self.No * self.K

  self.i = 0
  self.j = 0
  self.iterations = 1

  if par.copy_solution then
    self.O[1]:copy(self.solution)
    self.P[1]:copy(self.probe_solution)
  end

  self:update_views()
  self:calculateO_denom()
  self:update_frames(self.P,self.O_views)

  self.norm_a = self.a_tmp2_real_PQstore:cmul(self.a,self.fm):pow(2):sum()
  local max_measured_I = self.a_tmp2_real_PQstore:cmul(self.a,self.fm):pow(2):sum(2):sum(3):max()
  -- pprint(max_measured_I)
  local P_fluence = self.P_tmp1_real_PQstore:normZ(self.P):sum()

  self.support = znn.SupportMask(probe_size,probe_size[#probe_size]*5/12)
  --
  -- self.P:mul(math.sqrt(max_measured_I/P_fluence))
  -- print(P_fluence,max_measured_I,self.P_tmp1_real_PQstore:normZ(self.P):sum())
  -- plt:plot(self.P[1][1]:zfloat(),'probe - it '..0)

  print(   '----------------------------------------------------')
  u.printf('K = %d',self.K)
  u.printf('N = (%d,%d)',self.Nx,self.Ny)
  u.printf('M = %d',self.M)
  u.printf('power_threshold is %g',self.power_threshold)
  print(   '----------------------------------------------------')
  -- u.printMem()
end

function engine.P_Mod(x,abs,measured_abs)
  x.THNN.P_Mod(x:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
end

function engine.P_Mod_renorm(x,fm,fdev,a,af,renorm)
  x.THNN.P_Mod_renorm(x:cdata(),fm:cdata(),fdev:cdata(),a:cdata(),af:cdata(),renorm)
end

function engine.InvSigma(x,sigma)
  x.THNN.InvSigma(x:cdata(),x:cdata(),sigma)
end

function engine.ClipMinMax(x,min,max)
  x.THNN.ClipMinMax(x:cdata(),x:cdata(),min,max)
end

function engine:update_views()
  for i=1,self.K do
    local slice = {{},{},{self.pos[i][1],self.pos[i][1]+self.M-1},{self.pos[i][2],self.pos[i][2]+self.M-1}}
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

function engine:maybe_copy_new_batch(z,z_h,k)
  if (k-1) % self.batch_size == 0 then
    local batch = math.floor(k/self.batch_size) + 1
    print('batch '..batch)
    local old_batch_start, old_batch_end, old_batch_size = table.unpack(self.old_batch_params)
    local batch_start, batch_end, batch_size = table.unpack(self.batch_params[batch])
    self.old_batch_params = self.batch_params[batch]
    local last_z, last_zh = table.unpack(self.last_z_zh)
    self.last_z_zh = {z,z_h}
    u.debug('old_batch_start, old_batch_end, old_batch_size = (%d,%d,%d)',old_batch_start, old_batch_end, old_batch_size)
    u.debug('batch_start, batch_end, batch_size             = (%d,%d,%d)',batch_start, batch_end, batch_size)
    last_zh[{{old_batch_start,old_batch_end},{},{},{},{}}]:copy(last_z[{{1,old_batch_size},{},{},{},{}}])
    z[{{1,batch_size},{},{},{},{}}]:copy(z_h[{{batch_start,batch_end},{},{},{},{}}])
  end
end

-- total memory requirements:
-- 1 x object complex
-- 1 x object real
-- 1 x probe real
-- 3 x z
function engine:allocateBuffers(K,No,Np,M,Nx,Ny)

  local frames_memory = 0
  local Fsize_bytes ,Zsize_bytes = 4,8

  self.O = torch.ZCudaTensor.new(No,1,Nx,Ny)
  self.O_denom = torch.CudaTensor(self.O:size())
  self.P = torch.ZCudaTensor.new(1,Np,M,M)

  local free_memory, total_memory = cutorch.getMemoryUsage(cutorch.getDevice())
  local used_memory = total_memory - free_memory
  local batches = 1
  for n_batches = 1,50 do
    frames_memory = math.ceil(K/n_batches)*No*Np*M*M * Zsize_bytes
    if used_memory + frames_memory * 3 < total_memory * 0.98 then
      batches = n_batches
      print(   '----------------------------------------------------')
      u.printf('Using %d batches for the reconstruction.',batches )
      u.printf('Total memory requirements:')
      u.printf('-- used  :    %-5.2f MB' , used_memory * 1.0 / 2^20)
      u.printf('-- frames  : 3x %-5.2f MB' , frames_memory / 2^20)
      print(   '====================================================')
      u.printf('-- Total   :    %-5.2f MB (%2.1f percent)' , (used_memory + frames_memory * 3.0) / 2^20,(used_memory + frames_memory * 3.0)/ total_memory * 100)
      u.printf('-- Avail   :    %-5.2f MB' , total_memory * 1.0 / 2^20)
      break
    end
  end

  batches = 2

  self.batch_params = {}
  local batch_size = math.ceil(self.K / batches)
  for i = 0,batches do
    local relative_end  = self.K - (i+1)*batch_size
    local data_finished = relative_end > 0 and 0 or relative_end
    self.batch_params[#self.batch_params+1] = {i*batch_size+1,(i+1)*batch_size-data_finished,batch_size-data_finished}
  end

  self.k_to_batch_index = {}
  for k = 0, self.K do
    self.k_to_batch_index[k+1] = (k % batch_size) + 1
  end
  self.batch_size = batch_size
  self.batches = batches
  K = batch_size
  u.debug('batch_size = %d',batch_size)

  self.z = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
  self.P_Qz = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
  self.P_Fz = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})

  if self.batches > 1 then
    self.z_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    self.P_Qz_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    self.P_Fz_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})

    self.old_batch_params = self.batch_params[1]
    local old_batch_start, old_batch_end, old_batch_size = table.unpack(self.old_batch_params)
    u.debug('old_batch_start, old_batch_end, old_batch_size = (%d,%d,%d)',old_batch_start, old_batch_end, old_batch_size)
    -- self.z[{{1,old_batch_size},{},{},{},{}}]:copy(self.z_h[{{old_batch_start,old_batch_end},{},{},{},{}}])
    -- self.P_Qz[{{1,old_batch_size},{},{},{},{}}]:copy(self.P_Qz_h[{{old_batch_start,old_batch_end},{},{},{},{}}])
    -- self.P_Fz[{{1,old_batch_size},{},{},{},{}}]:copy(self.P_Fz_h[{{old_batch_start,old_batch_end},{},{},{},{}}])
  end

  local P_Qz_storage = self.P_Qz:storage()
  local P_Fz_storage = self.P_Fz:storage()

  local z2_pointer = tonumber(torch.data(P_Qz_storage,true))
  local z3_pointer = tonumber(torch.data(P_Fz_storage,true))

  local P_Qz_storage_real = torch.CudaStorage(P_Qz_storage:size()*2,z2_pointer)
  local P_Fz_storage_real = torch.CudaStorage(P_Fz_storage:size()*2,z3_pointer)

  local P_Qstorage_offset, P_Fstorage_offset = 1,1

  -- buffers in P_Qz_storage

  self.P_tmp1_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_PQstore:nElement() + 1
  self.P_tmp3_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_PQstore:nElement() + 1

  self.zk_tmp1_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_PQstore:nElement() + 1
  self.zk_tmp2_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_PQstore:nElement() + 1

  self.O_tmp_PQstore = torch.ZCudaTensor.new(P_Qz_storage,P_Qstorage_offset,{No,1,Nx,Ny})
  P_Qstorage_offset = P_Qstorage_offset + self.O_tmp_PQstore:nElement()

  -- offset for real arrays
  P_Qstorage_offset = P_Qstorage_offset*2 + 1

  self.P_tmp1_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_real_PQstore:nElement() + 1
  self.P_tmp2_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp2_real_PQstore:nElement() + 1
  self.P_tmp3_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_real_PQstore:nElement() + 1
  self.a_tmp_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,1,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp_real_PQstore:nElement() + 1

  self.zk_tmp1_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_real_PQstore:nElement() + 1
  self.zk_tmp2_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_real_PQstore:nElement() + 1

  self.a_tmp2_real_PQstore = torch.CudaTensor(P_Qz_storage_real,1,torch.LongStorage{self.K,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp2_real_PQstore:nElement() + 1

  -- buffers in P_Fz_storage

  self.P_tmp1_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_PFstore:nElement() + 1
  self.P_tmp2_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_PFstore:nElement() + 1
  self.P_tmp3_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_PFstore:nElement() + 1
  self.P_tmp4_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp4_PFstore:nElement() + 1
  self.P_tmp5_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp5_PFstore:nElement() + 1
  self.P_tmp6_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp6_PFstore:nElement() + 1
  self.P_tmp7_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp7_PFstore:nElement() + 1
  self.P_tmp8_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp8_PFstore:nElement() + 1
  self.P_tmp9_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp9_PFstore:nElement() + 1
  self.P_tmp10_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp10_PFstore:nElement() + 1
  self.P_tmp11_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp11_PFstore:nElement() + 1

  self.zk_tmp1_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp1_PFstore:nElement() + 1
  self.zk_tmp2_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp2_PFstore:nElement() + 1
  self.zk_tmp3_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp3_PFstore:nElement() + 1
  self.zk_tmp4_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp4_PFstore:nElement() + 1
  self.zk_tmp5_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp5_PFstore:nElement() + 1
  self.zk_tmp6_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp6_PFstore:nElement() + 1
  self.zk_tmp7_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp7_PFstore:nElement() + 1
  self.zk_tmp8_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp8_PFstore:nElement() + 1

  self.O_tmp_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,1,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_PFstore:nElement()

  -- offset for real arrays
  P_Fstorage_offset = P_Fstorage_offset*2 + 1
  self.P_tmp1_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_real_PFstore:nElement() + 1
  self.P_tmp2_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_real_PFstore:nElement() + 1
  self.P_tmp3_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_real_PFstore:nElement() + 1
  self.a_tmp_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{1,1,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.a_tmp_real_PFstore:nElement() + 1
  self.O_tmp_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{No,1,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_real_PFstore:nElement() + 1

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

  local a = self.a_tmp_real_PQstore

  for k=1,self.K do
    a:zero()
    for o = 1,self.No do
      -- plt:plotcompare({self.z[k][1]:re():float(),self.z[k][1]:im():float()},'self.z[k]')
      -- plt:plot(a_m:float():log(),'diff')
      a:add(self.z[k][o]:fftBatched():sum(1)[1]:abs())
    end
    self.a[k]:copy(a)
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
  local norm_P =  self.P_tmp2_real_PQstore:normZ(self.P):sum(self.P_dim)
  local tmp = self.P_tmp2_real_PQstore
  norm_P = norm_P:expandAs(self.O_denom_views[1])
  for _, view in ipairs(self.O_denom_views) do
    view:add(norm_P)
  end
  local fact_start, fact_end = 1e-3, 1e-6

  local fact = fact_start-(self.i/self.iterations)*(fact_start-fact_end)
  local abs_max = tmp:absZ(self.P):max()
  local sigma = abs_max * abs_max * fact
  -- local sigma = 1e-8
  -- print('sigma = '..sigma)
  -- plt:plot(self.O_denom[1][1]:float():log(),'calculateO_denom  self.O_denom')
  self.InvSigma(self.O_denom,sigma)
  self.O_mask = self.O_denom:lt(1e-3)
  -- plt:plot(self.O_denom[1][1]:float():log(),'O_denom')
  -- plt:plot(self.O_mask[1]:float(),'O_mask')
end

-- P_Qz, P_Fz free to use
function engine:merge_frames(mul_merge, merge_memory, merge_memory_views)
  local mul_merge_repeated = self.zk_tmp1_PQstore
  merge_memory:mul(self.object_inertia)
  for k, view in ipairs(merge_memory_views) do
    self:maybe_copy_new_batch(self.z,self.z_h,k)
    local ind = self.k_to_batch_index[k]
    mul_merge_repeated:conj(mul_merge:expandAs(self.z[ind]))
    view:add(mul_merge_repeated:cmul(self.z[ind]):sum(self.P_dim))
  end
  merge_memory:cmul(self.O_denom)
end

-- P_Fz free
function engine:update_frames(mul_split,merge_memory_views)
  local z = self.P_Qz
  for k, view in ipairs(merge_memory_views) do
    self:maybe_copy_new_batch(self.P_Qz,self.P_Qz_h,k)
    local ind = self.k_to_batch_index[k]
    z[ind]:cmul(view:expandAs(z[ind]),mul_split:expandAs(z[ind]))
  end
end

function engine:P_Q_plain()
  self:merge_frames(self.P,self.O,self.O_views)
  self:update_frames(self.P,self.O_views)
end

function engine:P_Q()
  if self.update_probe then
    for _ = 1,self.P_Q_iterations do
      self.j = _
      probe_change = self:refine_probe()
      self:merge_frames(self.P,self.O,self.O_views)
      -- or probe_change > 0.97 * last_probe_change
      if not probe_change_0 then probe_change_0 = probe_change end
      if probe_change < .1 * probe_change_0  then break end
      -- last_probe_change = probe_change
      u.printf('            probe change : %g',probe_change)
    end
    self:update_frames(self.P,self.O_views)
  else
    self:P_Q_plain()
  end
end

function engine:P_F()
  local z = self.P_Fz
  local abs = self.zk_tmp1_real_PQstore
  local da = self.a_tmp_real_PQstore
  local fdev = self.zk_tmp2_real_PQstore
  local module_error, err_fmag, renorm  = 0, 0, 0
  local mod_updates = 0
  for k=1,self.K do
    self:maybe_copy_new_batch(self.P_Fz,self.P_Fz_h,k)
    local ind = self.k_to_batch_index[k]
    for o = 1, self.No do
      z[ind][o]:fftBatched()
    end

    -- sum over probe and object modes
    abs = abs:normZ(z[ind]):sum(self.O_dim):sum(self.P_dim)
    abs:sqrt()
    -- plt:plot(abs[1]:float():log(),'abs')
    fdev:add(abs:expandAs(fdev),-1,self.a_exp[ind])
    da:pow(fdev,2):cmul(self.fm_exp[ind])
    err_fmag = da:sum()
    module_error = module_error + err_fmag
    if err_fmag > self.power_threshold then
      renorm = math.sqrt(self.power_threshold/err_fmag)
      -- plt:plot(z_out[ind][1][1]:abs():float():log(),'z_out[ind][1] before')
      self.P_Mod_renorm(z[ind],self.fm_exp[k],fdev,self.a_exp[k],abs:expandAs(z[ind]),renorm)
      -- self.P_Mod(z_out[ind],abs,self.a[k])
      -- plt:plot(z_out[k][1][1]:abs():float():log(),'z_out[k][1] after')
      mod_updates = mod_updates + 1
    end

    for o = 1, self.No do
      z[ind][o]:ifftBatched()
    end
  end

  -- u.printf('%d/%d modulus updates', mod_updates, self.K)

  module_error = module_error / self.norm_a
  return module_error, mod_updates
end

function engine:maybe_refine_positions()
  if self.update_positions then
    self:refine_positions()
  end
end

function engine:maybe_plot()
  if self.do_plot then
    -- :cmul(self.O_mask)
    plt:plot(self.O_tmp_PFstore:copy(self.O):cmul(self.O_mask)[1][1]:zfloat(),'object - it '..self.i)
    plt:plot(self.P[1][1]:zfloat(),'new probe')
  end
end

function engine:refine_positions()
end

function engine:refine_probe()
  local new_P = self.P_tmp1_PQstore
  local oview_conj = self.zk_tmp1_PQstore

  local dP = self.P_tmp3_PQstore
  local dP_abs = self.P_tmp3_real_PQstore

  local new_P_denom = self.P_tmp1_real_PQstore
  local denom_tmp = self.zk_tmp1_real_PQstore

  new_P:mul(self.P,self.probe_inertia)
  new_P_denom:fill(self.probe_inertia)

  for k, view in ipairs(self.O_views) do
    local view_exp = view:expandAs(self.z[k])
    oview_conj:conj(view_exp)
    new_P:add(oview_conj:cmul(self.z[k]):sum(self.O_dim))
    new_P_denom:add(denom_tmp:normZ(view_exp):sum(self.O_dim))
  end

  new_P:cdiv(new_P_denom)
  -- new_probe = self.support:forward(new_probe)

  local probe_change = dP_abs:normZ(dP:add(new_P,-1,self.P)):sum()
  local P_norm = dP_abs:normZ(self.P):sum()
  self.P:copy(new_P)

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
