local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local engine, super = classic.class(...,base_engine)
require 'cutorch'
function engine:_init(par)
  super._init(self,par)
end

-- total memory requirements:
-- 1 x object complex
-- 1 x object real
-- 1 x probe real
-- 3 x z
function engine:allocate_buffers(K,No,Np,M,Nx,Ny)

  local frames_memory = 0
  local Fsize_bytes ,Zsize_bytes = 4,8

  self:allocate_object(No,Nx,Ny)
  self:allocate_probe(Np,M)

  self.O_denom0 = torch.CudaTensor(1,1,Nx,Ny)
  self.O_denom = self.O_denom0:expandAs(self.O)
  self.O_mask0 = torch.CudaTensor(1,1,Nx,Ny)
  self.O_mask = self.O_mask0:expandAs(self.O)

  if self.background_correction_start > 0 then
    self.eta = torch.CudaTensor(M,M)
    self.bg = torch.CudaTensor(M,M)
    self.eta_like_a = self.eta:view(1,self.M,self.M):expand(self.K,self.M,self.M)
    self.bg_like_a = self.bg:view(1,self.M,self.M):expand(self.K,self.M,self.M)
    self.eta_exp = self.eta:view(1,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
    self.bg_exp = self.bg:view(1,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
    self.bg:zero()
    self.eta:fill(1)
  end

  local free_memory, total_memory = cutorch.getMemoryUsage(cutorch.getDevice())
  local used_memory = total_memory - free_memory
  local batches = 1
  for n_batches = 1,50 do
    frames_memory = math.ceil(K/n_batches)*No*Np*M*M * Zsize_bytes
    if used_memory + frames_memory * 3 < total_memory * 0.85 then
      batches = n_batches
      print(   '----------------------------------------------------')
      u.printf('Using %d batches for the reconstruction. ',batches )
      u.printf('Total memory requirements:')
      u.printf('-- used  :    %-5.2f MB' , used_memory * 1.0 / 2^20)
      u.printf('-- frames  : 3x %-5.2f MB' , frames_memory / 2^20)
      print(   '====================================================')
      u.printf('-- Total   :    %-5.2f MB (%2.1f percent)' , (used_memory + frames_memory * 3.0) / 2^20,(used_memory + frames_memory * 3.0)/ total_memory * 100)
      u.printf('-- Avail   :    %-5.2f MB' , total_memory * 1.0 / 2^20)
      break
    end
  end

  -- if batches == 1 then batches = 2 end

  self.batch_params = {}
  local batch_size = math.ceil(self.K / batches)
  for i = 0,batches-1 do
    local relative_end  = self.K - (i+1)*batch_size
    -- print('relative_end ',relative_end)
    local data_finished = relative_end > 0 and 0 or relative_end
    -- print('data_finished ',data_finished)
    self.batch_params[#self.batch_params+1] = {i*batch_size+1,(i+1)*batch_size+data_finished,batch_size+data_finished}
    -- pprint({i*batch_size+1,(i+1)*batch_size+data_finished,batch_size+data_finished})
  end

  self.k_to_batch_index = {}
  for k = 1, self.K do
    self.k_to_batch_index[k] = ((k-1) % batch_size) + 1
  end
  -- pprint(self.k_to_batch_index)
  self.batch_size = batch_size
  self.batches = batches
  K = batch_size
  u.debug('batch_size = %d',batch_size)

  self.z = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
  self.P_Qz = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
  self.P_Fz = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})

  self.z1 = self.P_Qz
  self.z2 = self.P_Fz

  if self.batches > 1 then
    self.z_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    if self.has_solution then
      -- allocate for relative error calculation
      self.z1_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    end
    self.P_Qz_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    self.P_Fz_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
  end

  self.old_batch_params = {}
  self.old_batch_params['z'] = self.batch_params[1]
  self.old_batch_params['z1'] = self.batch_params[1]
  self.old_batch_params['P_Qz'] = self.batch_params[1]
  self.old_batch_params['P_Fz'] = self.batch_params[1]

  local P_Qz_storage = self.P_Qz:storage()
  local P_Fz_storage = self.P_Fz:storage()

  local z2_pointer = tonumber(torch.data(P_Qz_storage,true))
  local z3_pointer = tonumber(torch.data(P_Fz_storage,true))

  local P_Qz_storage_real = torch.CudaStorage(P_Qz_storage:size()*2,z2_pointer)
  local P_Fz_storage_real = torch.CudaStorage(P_Fz_storage:size()*2,z3_pointer)

  local P_Qstorage_offset, P_Fstorage_offset = 1,1

  -- buffers in P_Qz_storage

  -- self.P_tmp1_PQstore = torch.ZCudaTensor.new( {1,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_PQstore:nElement() + 1
  -- self.P_tmp3_PQstore = torch.ZCudaTensor.new( {1,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_PQstore:nElement() + 1
  --
  -- self.zk_tmp1_PQstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_PQstore:nElement() + 1
  -- self.zk_tmp2_PQstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_PQstore:nElement() + 1
  --
  -- self.O_tmp_PQstore = torch.ZCudaTensor.new( {No,1,Nx,Ny})
  -- P_Qstorage_offset = P_Qstorage_offset + self.O_tmp_PQstore:nElement()
  --
  -- -- offset for real arrays
  -- P_Qstorage_offset = P_Qstorage_offset*2 + 1
  --
  -- self.P_tmp1_real_PQstore = torch.CudaTensor( torch.LongStorage{1,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_real_PQstore:nElement() + 1
  -- self.P_tmp2_real_PQstore = torch.CudaTensor( torch.LongStorage{1,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.P_tmp2_real_PQstore:nElement() + 1
  -- self.P_tmp3_real_PQstore = torch.CudaTensor( torch.LongStorage{1,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_real_PQstore:nElement() + 1
  -- self.a_tmp_real_PQstore = torch.CudaTensor( torch.LongStorage{1,1,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.a_tmp_real_PQstore:nElement() + 1
  -- self.a_tmp3_real_PQstore = torch.CudaTensor( torch.LongStorage{1,1,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.a_tmp3_real_PQstore:nElement() + 1
  --
  -- self.zk_tmp1_real_PQstore = torch.CudaTensor( torch.LongStorage{No,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_real_PQstore:nElement() + 1
  -- self.zk_tmp2_real_PQstore = torch.CudaTensor( torch.LongStorage{No,Np,M,M})
  -- P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_real_PQstore:nElement() + 1
  --
  -- if P_Qstorage_offset + self.K*M*M + 1 > self.P_Qz:nElement() * 2 then
  --   self.d = torch.CudaTensor(torch.LongStorage{self.K,M,M})
  -- else
  --   self.d = torch.CudaTensor( torch.LongStorage{self.K,M,M})
  --   P_Qstorage_offset = P_Qstorage_offset + self.d:nElement() + 1
  -- end
  --
  -- self.a_buffer1 = torch.CudaTensor(P_Qz_storage_real,1,torch.LongStorage{self.K,M,M})
  --
  -- -- buffers in P_Fz_storage
  --
  -- self.P_tmp1_PFstore = torch.ZCudaTensor.new( {1,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_PFstore:nElement() + 1
  -- self.P_tmp2_PFstore = torch.ZCudaTensor.new( {1,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_PFstore:nElement() + 1
  -- self.P_tmp3_PFstore = torch.ZCudaTensor.new( {1,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_PFstore:nElement() + 1
  --
  -- self.zk_tmp1_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp1_PFstore:nElement() + 1
  -- self.zk_tmp2_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp2_PFstore:nElement() + 1
  -- self.zk_tmp5_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp5_PFstore:nElement() + 1
  -- self.zk_tmp6_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp6_PFstore:nElement() + 1
  -- self.zk_tmp7_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp7_PFstore:nElement() + 1
  -- self.zk_tmp8_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp8_PFstore:nElement() + 1
  --
  -- self.O_tmp_PFstore = torch.ZCudaTensor.new( {No,1,Nx,Ny})
  -- P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_PFstore:nElement()
  --
  -- -- offset for real arrays
  -- P_Fstorage_offset = P_Fstorage_offset*2 + 1
  -- self.P_tmp1_real_PFstore = torch.CudaTensor.new( torch.LongStorage{1,Np,M,M})
  -- P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_real_PFstore:nElement() + 1
  -- self.O_tmp_real_PFstore = torch.CudaTensor.new( torch.LongStorage{No,1,Nx,Ny})
  -- P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_real_PFstore:nElement() + 1

  self.P_tmp1_PQstore = torch.ZCudaTensor.new( {1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_PQstore:nElement() + 1
  self.P_tmp2_PQstore = torch.ZCudaTensor.new( {1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp2_PQstore:nElement() + 1

  self.zk_tmp1_PQstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_PQstore:nElement() + 1
  self.zk_tmp2_PQstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_PQstore:nElement() + 1

  self.O_tmp_PQstore = torch.ZCudaTensor.new( {No,1,Nx,Ny})
  P_Qstorage_offset = P_Qstorage_offset + self.O_tmp_PQstore:nElement()

  -- offset for real arrays
  P_Qstorage_offset = P_Qstorage_offset*2 + 1

  self.P_tmp1_real_PQstore = torch.CudaTensor( torch.LongStorage{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp1_real_PQstore:nElement() + 1
  self.P_tmp2_real_PQstore = torch.CudaTensor( torch.LongStorage{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp2_real_PQstore:nElement() + 1
  self.P_tmp3_real_PQstore = torch.CudaTensor( torch.LongStorage{1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_real_PQstore:nElement() + 1
  self.a_tmp_real_PQstore = torch.CudaTensor( torch.LongStorage{1,1,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp_real_PQstore:nElement() + 1
  self.a_tmp3_real_PQstore = torch.CudaTensor( torch.LongStorage{1,1,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp3_real_PQstore:nElement() + 1

  self.zk_tmp1_real_PQstore = torch.CudaTensor( torch.LongStorage{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_real_PQstore:nElement() + 1
  self.zk_tmp2_real_PQstore = torch.CudaTensor( torch.LongStorage{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_real_PQstore:nElement() + 1

  if P_Qstorage_offset + self.K*M*M + 1 > self.P_Qz:nElement() * 2 then
    self.d = torch.CudaTensor(torch.LongStorage{self.K,M,M})
  else
    self.d = torch.CudaTensor( torch.LongStorage{self.K,M,M})
    P_Qstorage_offset = P_Qstorage_offset + self.d:nElement() + 1
  end

  self.a_buffer1 = torch.CudaTensor(P_Qz_storage_real,1,torch.LongStorage{self.K,M,M})

  -- buffers in P_Fz_storage

  self.zk_tmp1_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp1_PFstore:nElement() + 1
  local P_Fstorage_offset0 = P_Fstorage_offset

  self.a_buffer2 = torch.CudaTensor(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{self.K,M,M})

  self.P_tmp1_PFstore = torch.ZCudaTensor.new( {1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset0 + self.P_tmp1_PFstore:nElement() + 1
  self.P_tmp2_PFstore = torch.ZCudaTensor.new( {1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_PFstore:nElement() + 1
  self.P_tmp3_PFstore = torch.ZCudaTensor.new( {1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_PFstore:nElement() + 1
  self.zk_tmp2_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp2_PFstore:nElement() + 1
  self.zk_tmp5_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp5_PFstore:nElement() + 1
  self.zk_tmp6_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp6_PFstore:nElement() + 1
  self.zk_tmp7_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp7_PFstore:nElement() + 1
  self.zk_tmp8_PFstore = torch.ZCudaTensor.new( {No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp8_PFstore:nElement() + 1

  self.O_tmp_PFstore = torch.ZCudaTensor.new( {No,1,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_PFstore:nElement()

  -- offset for real arrays
  P_Fstorage_offset = P_Fstorage_offset*2 + 1
  self.P_tmp1_real_PFstore = torch.CudaTensor.new( torch.LongStorage{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_real_PFstore:nElement() + 1
  self.O_tmp_real_PFstore = torch.CudaTensor.new( torch.LongStorage{No,1,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_real_PFstore:nElement() + 1

  if self.position_refinement_method == 'annealing' then
    self.z_buffer_trials = torch.ZCudaTensor.new( torch.LongStorage{self.position_refinement_trials,No,Np,M,M})
    self.z_buffer_trials_real = torch.CudaTensor.new( torch.LongStorage{self.position_refinement_trials,No,Np,M,M})
  end

  self.P_buffer = self.P_tmp1_PQstore
  self.P_buffer2 = self.P_tmp2_PQstore
  self.P_buffer_real = self.P_tmp1_real_PQstore
  self.zk_buffer_update_frames = self.zk_tmp1_PFstore
  self.zk_buffer_merge_frames = self.zk_tmp1_PQstore
  self.Pk_buffer_real = self.a_tmp_real_PQstore
  self.O_buffer_real = self.O_tmp_real_PFstore
  self.O_buffer = self.O_tmp_PFstore
  self.shift_buffer = self.P_tmp1_PFstore[1][1]
end

function engine:DM_update()
  if self.i >= self.background_correction_start then
    -- print('DM_update_with_background')
    return self:DM_update_with_background()
  else
    return self:DM_update_without_background()
  end
end

-- P_Qz, P_Fz free to use
function engine:merge_frames(z,mul_merge, merge_memory, merge_memory_views, do_normalize_merge_memory)
  self:merge_frames_internal(z,mul_merge, merge_memory, merge_memory_views, self.zk_buffer_merge_frames, self.P_buffer, do_normalize_merge_memory)
end

function engine:DM_update_without_background()
  -- print('DM_update')
  local mod_error, mod_updates
  -- local overlap_error = self:overlap_error(self.z,self.P_Qz)
  self.mod_updates, self.module_error = 0, 0
  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch_all(k)
  -- self.P_Fz = (2P_Q - I)z
    -- plt:plot(self.P_Qz[1][1][1],'P_Qz '..1)
    -- plt:plot(self.z[1][1][1],'z '..1)
    self.P_Fz:mul(self.P_Qz,1+self.beta):add(-self.beta,self.z)
    -- plt:plot(self.P_Fz[1][1][1],'z '..1)
    -- z_{i+1} = z_i - P_Q z_i , P_Qz can be used as buffer now
    self.z:add(-1,self.P_Qz)
    -- self.P_Fz = P_F((2P_Q - I))z
    self.plots = 0
    mod_error, mod_updates = self:P_F()

    --  z_{i+1} = z_i - P_Q z_i + P_F(((1+beta))P_Q - beta I))z_i
    self.z:add(self.P_Fz)
  end
  -- self:maybe_copy_new_batch_z(1)
  return mod_error, mod_updates
end

function engine:DM_update_with_background()
  -- print('DM_update')
  local mod_error, mod_updates
  -- local overlap_error = self:overlap_error(self.z,self.P_Qz)
  self.mod_updates, self.module_error = 0, 0
  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch_all(k)
  -- self.P_Fz = (2P_Q - I)z
    self.P_Fz:mul(self.P_Qz,1+self.beta):add(-self.beta,self.z)
    -- z_{i+1} = z_i - P_Q z_i , P_Qz can be used as buffer now
    self.z:add(-1,self.P_Qz)
  end
  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch_z(k)
    self:maybe_copy_new_batch_P_F(k)
    -- self.P_Fz = P_F((2P_Q - I))z
    self.plots = 0
    mod_error, mod_updates = self:P_F()
    --  z_{i+1} = z_i - P_Q z_i + P_F((2P_Q - I))z_i
    self.z:add(self.P_Fz)
  end
  -- self:maybe_copy_new_batch_z(1)
  return self.module_error, self.mod_updates
end

function engine:get_errors()
  return {self.mod_errors:narrow(1,1,self.i), self.overlap_errors:narrow(1,1,self.i),self.img_errors:narrow(1,1,self.i),self.rel_errors:narrow(1,1,self.i)}
end
function engine:get_error_labels()
  return {'RFE','ROE','RMSE_img','RMSE_z'}
end
function engine:allocate_error_history()
  self.mod_errors = torch.FloatTensor(self.iterations):fill(1)
  self.overlap_errors = torch.FloatTensor(self.iterations):fill(1)
  self.img_errors = torch.FloatTensor(self.iterations):fill(1)
  self.rel_errors = torch.FloatTensor(self.iterations):fill(1)
end

function engine:save_error_history(hdfile)
  hdfile:write('/results/err_img_final',torch.FloatTensor({self.img_errors[self.i-1]}))
  hdfile:write('/results/err_rel_final',torch.FloatTensor({self.rel_errors[self.i-1]}))
  hdfile:write('/results/err_overlap_final',torch.FloatTensor({self.rel_errors[self.i-1]}))
  hdfile:write('/results/err_mod_final',torch.FloatTensor({self.rel_errors[self.i-1]}))

  hdfile:write('/results/err_img',self.img_errors:narrow(1,1,self.i))
  hdfile:write('/results/err_rel',self.rel_errors:narrow(1,1,self.i))
  hdfile:write('/results/err_overlap',self.rel_errors:narrow(1,1,self.i))
  hdfile:write('/results/err_mod',self.mod_errors:narrow(1,1,self.i))
end

function engine:iterate(steps)
  self:before_iterate()
  if self.has_solution then
    u.printf('rel error : %g',self:relative_error())
  end
  self.iterations = steps
  self:initialize_plotting()
  local mod_error, overlap_error, relative_error, probe_error, mod_updates, im_error = -1,-1,nil, nil, 0
  local probe_change_0, last_probe_change, probe_change = nil, 1e10, 0
  u.printf('%-10s%-15s%-15s%-15s%-15s%-15s%-15s','iteration','e_mod','e_over','e_rel','e_img','e_probe','modulus updates %')
  print('----------------------------------------------------------------------------------------------')
  -- self:update_frames(self.P_Qz,self.P,self.O_views,self.maybe_copy_new_batch_P_Q)
  for i=1,steps do
    self:update_iteration_dependent_parameters(i)
    print('before P_Q')
    self:P_Q()

    plt:plotReIm(self.O[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'O after PQ',path .. '/object/'.. string.format('%d_Oreim',i),false)
    plt:plot(self.O[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'O after PQ',path .. '/object1/'.. string.format('%d_O',i),false)
    if self.has_solution then
      self.img_errors[{i}] = self:image_error()
      probe_error = self:probe_error()
    end
    -- u.printf('rel error P_Qz: %g',self:relative_error(self.P_Qz))
    self.overlap_errors[i] = self:overlap_error(self.z,self.P_Qz)
    self:maybe_refine_positions()

    print('before DM_update')
    self.mod_errors[i], mod_updates = self:DM_update()

    if self.has_solution then
      self.img_errors[{i}] = self:image_error()
      self.rel_errors[{i}] = self:relative_error()
      probe_error = self:probe_error()
    end

    u.printf('%-10d%-15g%-15g%-15g%-15g%-15g%-15g',i,self.mod_errors[i] or -1,self.overlap_errors[i] or -1 , self.rel_errors[i] or -1,self.img_errors[i] or -1, probe_error or -1, mod_updates/self.K*100.0)

    self:maybe_plot()
    self:maybe_save_data()

    collectgarbage()
  end
  self:save_data(self.save_path .. self.run_label .. '_DM_' .. (steps+1))
  self:maybe_plot()
  plt:shutdown_reconstruction_plot()
  -- plt:plot(self.O[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'object - it '..steps)
  -- plt:plot(self.P[1]:zfloat(),'new probe')
end

return engine
