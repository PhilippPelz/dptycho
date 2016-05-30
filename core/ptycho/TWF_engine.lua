local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine_shifted"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local TruncatedPoissonLikelihood = require 'dptycho.znn.TruncatedPoissonLikelihood'
local plt = plot()
local optim = require 'optim'
local engine, super = classic.class(...,base_engine)

function engine:_init(par, a_h)
  super._init(self,par)
  self.a_h = a_h

  self.mu_max = 0.01
  self.tau0 = 330

  self.P_optim = self.P_optim(opfunc, x, config, state)
  self.O_optim = self.O_optim(opfunc, x, config, state)
end

function engine:allocateBuffers(K,No,Np,M,Nx,Ny)

    local frames_memory = 0
    local Fsize_bytes ,Zsize_bytes = 4,8

    self.O = torch.ZCudaTensor.new(No,1,Nx,Ny)
    self.O_denom = torch.CudaTensor(self.O:size())

    self.dL_dO = torch.ZCudaTensor.new(self.O:size())
    self.dL_dO_tmp1 = torch.ZCudaTensor.new(self.O:size())
    self.dL_dO_tmp2 = torch.ZCudaTensor.new(self.O:size())

    if not self.P then
      self.P = torch.ZCudaTensor.new(1,Np,M,M)
    end

    self.dL_dP = torch.ZCudaTensor.new(self.P:size())
    self.dL_dP_tmp1 = torch.ZCudaTensor.new(self.P:size())
    self.dL_dP_tmp2 = torch.ZCudaTensor.new(self.P:size())
    -- if self.background_correction_start > 0 then
    --   self.eta = torch.CudaTensor(M,M)
    --   self.bg = torch.CudaTensor(M,M)
    --   self.eta_like_a = self.eta:view(1,self.M,self.M):expand(self.K,self.M,self.M)
    --   self.bg_like_a = self.bg:view(1,self.M,self.M):expand(self.K,self.M,self.M)
    --   self.eta_exp = self.eta:view(1,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
    --   self.bg_exp = self.bg:view(1,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
    --   self.bg:zero()
    --   self.eta:fill(1)
    -- end

    if self.has_solution then
      self.solution = torch.ZCudaTensor.new(No,1,Nx,Ny)
    end

    local free_memory, total_memory = cutorch.getMemoryUsage(cutorch.getDevice())
    local used_memory = total_memory - free_memory
    local probe_grad_memory = Np * M * M * Zsize_bytes * 3
    local object_grad_memory = No * M * M * Zsize_bytes * 3
    local batches = 1
    for n_batches = 1,50 do
      frames_memory = math.ceil(K/n_batches)*No*Np*M*M * Zsize_bytes
      local needed_memory = frames_memory + probe_grad_memory + object_grad_memory
      if used_memory + needed_memory < total_memory * 0.85 then
        batches = n_batches
        print(   '----------------------------------------------------')
        u.printf('Using %d batches for the reconstruction. ',batches )
        u.printf('Total memory requirements:')
        u.printf('-- used  :    %-5.2f MB' , used_memory * 1.0 / 2^20)
        u.printf('-- needed: 3x %-5.2f MB' , needed_memory / 2^20)
        print(   '====================================================')
        u.printf('-- Total   :    %-5.2f MB (%2.1f percent)' , (used_memory + needed_memory) / 2^20,(used_memory + needed_memory)/ total_memory * 100)
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
    self.dL_dz = z

    self.buffer1 = torch.CudaTensor.new(torch.LongStorage{K,M,M})
    self.buffer2 = torch.CudaTensor.new(torch.LongStorage{K,M,M})

    if self.batches > 1 then
      self.z_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    end

    self.old_batch_params = {}
    self.old_batch_params['z'] = self.batch_params[1]

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

    -- offset for real arrays
    P_Qstorage_offset = P_Qstorage_offset*2 + 1

    self.P_tmp1_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,Np,M,M})

    self.L = TruncatedPoissonLikelihood(self.a_h, self.z, self.fm, self.buffer1, self.buffer2, self.par, K, No, Np)
end

function engine:initialize_views()
  self.O_views = {}
  self.dL_dO_views = {}
end

function engine:calculate_dL_dP(dL_dP)
  self:refine_probe_internal(dL_dP,self.zk_tmp1_PQstore,self.zk_tmp2_PQstore,self.P_tmp3_PQstore,self.P_tmp3_real_PQstore)
end

-- buffers:
--  3 x sizeof(z[k]) el C
--  2 x sizeof(P) el C
--  1 x sizeof(P) el R
function engine:refine_probe_internal(P_update_buffer, z_buffer1, z_buffer2, P_buffer, P_real_buffer)
  -- print('refine_probe')
  local new_P = P_update_buffer
  local oview_conj = z_buffer1
  local oview_conj_shifted = z_buffer2

  local dP = P_buffer
  local dP_abs = P_real_buffer

  new_P:zero()

  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(self.O_views) do
    self:maybe_copy_new_batch_z(k)
    local ind = self.k_to_batch_index[k]
    pos:fill(-1):cmul(self.dpos[k])
    oview_conj:conj(view:expandAs(self.z[ind]))

    oview_conj:cmul(self.z[ind])
    for o = 1, self.No do
      oview_conj_shifted[o]:shift(oview_conj[o],pos)
    end

    new_P:add(oview_conj_shifted:sum(self.O_dim))
  end
  local P_norm = dP_abs:normZ(self.P):sum()
  u.printf('total current: %g, max_pow/total_current = %g',P_norm,self.max_power/P_norm)
  -- plt:plot(new_P[1][1]:zfloat(),title1,self.save_path ..title1)

  -- if self.probe_support then self.P = self.support:forward(self.P) end

  if self.probe_regularization_amplitude(self.i) then self:regularize_probe() end
  if self.probe_lowpass_fwhm(self.i) then self:filter_probe() end

  u.printram('after refine_probe')
  return math.sqrt(P_norm/self.Np)
end

function engine:update_views()
  for i=1,self.K do
    local slice = {{},{},{self.pos[i][1],self.pos[i][1]+self.M-1},{self.pos[i][2],self.pos[i][2]+self.M-1}}
    -- pprint(slice)
    self.O_views[i] = self.O[slice]
    self.dL_dO_views[i] = self.dL_dO[slice]
  end
end

function engine:merge_frames(mul_merge, merge_memory, merge_memory_views)
  -- TODO buffers
  self:merge_frames_internal(self.z, mul_merge, merge_memory, merge_memory_views, self.zk_tmp1_PQstore, self.P_tmp1_PQstore, false)
end

function engine:mu(it)
  -- muf = @(t) min(1-exp(-t/Params.tau0), Params.muTWF)
  return math.min(1-math.exp(-it/self.tau0),self.mu_max)
end

function engine:iterate(steps)
  for i = 1, steps do
    self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)
    local L = self.L:updateOutput(self.z,self.a)
    self.dL_dz = self.L:updateGradInput(self.z,self.a)

    self:merge_frames(self.P,self.dL_dO, self.dL_dO_views)
    self:calculate_dL_dP(self.dL_dP)

    self.O:add(self:mu(i),self.dL_dO)
    self.P:add(self:mu(i),self.dL_dP)

    u.printf('iteration %-3d: L = %-02.02g',i,L)


    -- grad = fun_compute_grad_TPWFP_Real(z, y, Params, A, At, Masks, n1_LR, n2_LR, fmaskpro);
    -- z = z - muf(t) * grad;             % Gradient update

  end
end
