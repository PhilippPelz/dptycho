local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local znn = require 'dptycho.znn'
local plt = plot()
local optim = require 'optim'
local fn = require 'fn'
require 'sys'

local TWF_engine, super = classic.class(...,base_engine)

function TWF_engine:_init(par)
  u.printf('========================== TWF engine ==========================')
  super._init(self,par)
  self.L = znn.TruncatedPoissonLikelihood(self.twf.a_h,self.twf.a_lb,self.twf.a_ub, self.z, self.fm, self.a_buffer1, self.a_buffer2, self.z1_buffer_real, self.K, self.No, self.Np, self.M, self.Nx, self.Ny, self.twf.diagnostics,self.twf.do_truncate)
  if self.regularizer then
    self.R = self.regularizer(self.O_tmp,self.dR_dO,self.rescale_regul_amplitude*self.twf.nu,self.O_tmp_real1,self.O_tmp_real2)
  end
  -- we deal with intensities in the MAP framework
  self.a:pow(2)
end

function TWF_engine:allocateBuffers(K,No,Np,M,Nx,Ny)
    local frames_memory = 0
    local Fsize_bytes ,Zsize_bytes = 4,8

    self:allocate_object(No,Nx,Ny)
    self:allocate_probe(Np,M)

    self.O_denom0 = torch.CudaTensor(1,1,Nx,Ny)
    print('allocateBuffers')
    -- pprint(self.O)
    -- pprint(self.O_denom0)
    self.O_denom = self.O_denom0:expandAs(self.O)
    self.O_mask0 = torch.CudaTensor(1,1,Nx,Ny)
    self.O_mask = self.O_mask0:expandAs(self.O)

    self.dL_dO = torch.ZCudaTensor.new(self.O:size())
    self.O_tmp = torch.ZCudaTensor.new(self.O:size())
    self.O_tmp_real1 = torch.CudaTensor.new(self.O:size())
    self.O_tmp_real2 = torch.CudaTensor.new(self.O:size())
    self.dR_dO = torch.ZCudaTensor.new(self.O:size())

    self.dL_dP = torch.ZCudaTensor.new(self.P:size()):zero()
    self.dL_dP_tmp1 = torch.ZCudaTensor.new(self.P:size())
    self.dL_dP_tmp1_real = torch.CudaTensor.new(self.P:size())

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

    local free_memory, total_memory = cutorch.getMemoryUsage(cutorch.getDevice())
    local used_memory = total_memory - free_memory
    local probe_grad_memory = Np * M * M * Zsize_bytes * 2.5
    local object_grad_memory = No * M * M * Zsize_bytes * 1.5
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
        u.printf('-- needed:    %-5.2f MB' , needed_memory / 2^20)
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
    end

    self.k_to_batch_index = {}
    for k = 1, self.K do
      self.k_to_batch_index[k] = ((k-1) % batch_size) + 1
    end
    self.batch_size = batch_size
    self.batches = batches
    K = batch_size
    u.debug('batch_size = %d',batch_size)

    self.z = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
    self.dL_dz = z
    self.z1 = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
    if self.has_solution then
      self.z2 = torch.ZCudaTensor.new(torch.LongStorage{K,No,Np,M,M})
    end

    if self.batches > 1 then
      self.z_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    end

    self.old_batch_params = {}
    self.old_batch_params['z'] = self.batch_params[1]

    local z1_storage = self.z1:storage()
    local z1_pointer = tonumber(torch.data(z1_storage,true))
    local z1_storage_real = torch.CudaStorage(z1_storage:size()*2,z1_pointer)

    local z1_storage_offset = 1
    self.a_buffer1 = torch.CudaTensor.new(z1_storage_real, z1_storage_offset, torch.LongStorage{K,M,M})
    z1_storage_offset = z1_storage_offset + self.a_buffer1:nElement() + 1

    if self:sufficient_space(z1_storage_real,z1_storage_offset,K*M*M) then
      -- there is enough room already allocated for the second buffer
      self.a_buffer2 = torch.CudaTensor.new(z1_storage_real, z1_storage_offset, torch.LongStorage{K,M,M})
      z1_storage_offset = z1_storage_offset + self.a_buffer2:nElement() + 1
    else
      self.a_buffer2 = torch.CudaTensor.new(torch.LongStorage{K,M,M})
    end

    if self:sufficient_space(z1_storage_real,z1_storage_offset,K*No*Np*M*M) then
      self.z1_buffer_real = torch.CudaTensor.new(z1_storage_real,z1_storage_offset,torch.LongStorage{K,No,Np,M,M})
      z1_storage_offset = z1_storage_offset + self.z1_buffer_real:nElement() + 1
    else
      self.z1_buffer_real = torch.CudaTensor.new(torch.LongStorage{K,No,Np,M,M})
    end

    self.zk_buffer_update_frames = self.z1[1]
    self.zk_buffer_merge_frames = self.z1[1]
    self.zk_buffer_2 = self.z1[2]
    self.P_buffer = self.dL_dP
    self.P_buffer_real = self.dL_dP_tmp1_real
    self.O_buffer_real = self.O_denom
    self.O_buffer = self.dR_dO
end

function TWF_engine:sufficient_space(storage,offset,elements)
  return elements < (#storage - offset)
end

function TWF_engine:initialize_views()
  self.O_views = {}
  self.O_denom_views = {}
  self.dL_dO_views = {}
  if self.has_solution then
    self.O_solution_views = {}
    self.O_mask_views = {}
  end
end

function TWF_engine:update_views()
  for i=1,self.K do
    local slice = {{},{},{self.pos[i][1],self.pos[i][1]+self.M-1},{self.pos[i][2],self.pos[i][2]+self.M-1}}
    -- pprint(slice)
    -- pprint(self.object_solution)
    self.O_views[i] = self.O[slice]
    self.O_denom_views[i] = self.O_denom[slice]
    self.dL_dO_views[i] = self.dL_dO[slice]
    if self.has_solution then
      self.O_solution_views[i] = self.object_solution[slice]
      self.O_mask_views[i] = self.O_mask[slice]
    end
  end
end

function TWF_engine:calculate_dL_dP(dL_dP)
  self:refine_probe_internal(dL_dP,self.zk_buffer_merge_frames,self.zk_buffer_2,self.P_buffer_real)
end

-- buffers:
--  3 x sizeof(z[k]) el C
--  2 x sizeof(P) el C
--  1 x sizeof(P) el R
function TWF_engine:refine_probe_internal(P_update_buffer, z_buffer1, z_buffer2, P_real_buffer)
  local new_P = P_update_buffer
  local oview_conj = z_buffer1
  local oview_conj_shifted = z_buffer2
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
  local P_norm = P_real_buffer:normZ(new_P):sum()
  -- u.printf('probe gradient norm: %g',P_norm)
  -- plt:plot(new_P[1][1]:zfloat(),title1,self.save_path ..title1)

  -- if self.probe_support then self.P = self.support:forward(self.P) end

  if self.probe_regularization_amplitude(self.i) then self:regularize_probe() end
  if self.probe_lowpass_fwhm(self.i) then self:filter_probe() end

  u.printram('after refine_probe')
  return math.sqrt(P_norm/self.Np)
end

function TWF_engine:mu(it)
  return math.min(1-math.exp(-it/self.twf.tau0),self.twf.mu_max) / self.a:nElement()
  -- return 1e-34
end

function TWF_engine:get_errors()
  return {self.rel_error:narrow(1,1,self.i), self.L_error:narrow(1,1,self.i),self.R_error:narrow(1,1,self.i)}
end

function TWF_engine:get_error_labels()
  return {'NRMSE','L','R'}
end

function TWF_engine:allocate_error_history()
  self.rel_error = torch.FloatTensor(self.iterations):fill(1)
  self.img_error = torch.FloatTensor(self.iterations):fill(1)
  self.R_error = torch.FloatTensor(self.iterations):fill(1)
  self.L_error = torch.FloatTensor(self.iterations):fill(1)
end

function TWF_engine.optim_func_object(self,O)
  self.counter = self.counter + 1
  -- u.printf('calls: %d',self.counter)
  -- u.printf('||O_old-O|| = %2.8g',self.old_O:clone():add(-1,self.O):normall(2)^2)
  -- u.printf('||O||       = %2.8g',self.O:normall(2)^2)
  self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)
  local L = self.L:updateOutput(self.z,self.a)
  self.dL_dz, valid_gradients = self.L:updateGradInput(self.z,self.a)
  -- plt:plot(self.dL_dz[20][1][1]:zfloat(),'dL_dz 20')
  -- plt:plot(self.dL_dz[40][1][1]:zfloat(),'dL_dz 40')
  -- plt:plot(self.dL_dz[60][1][1]:zfloat(),'dL_dz 60')

  -- calculate dL_dO
  self:merge_frames(self.dL_dz,self.P,self.dL_dO, self.dL_dO_views)
  -- plt:plot(self.dL_dO[1][1]:zfloat(),'O 1')
  if self.regularizer then
    self.R_error[self.i] = self.R:updateOutput(self.O)
    self.dR_dO = self.R:updateGradInput(self.O)
    self.dL_dO:add(self.dR_dO)
  end
  -- plt:plot(self.dR_dO[1][1]:zfloat(),'self.dR_dO')

  return L, self.dL_dO
end

function TWF_engine.optim_func_probe(self,P)
  local L = self.L:updateOutput(self.z,self.a)
  self.dL_dz, valid_gradients = self.L:updateGradInput(self.z,self.a)
  self:calculate_dL_dP(self.dL_dP)
  return L, self.dL_dP
end

function TWF_engine:iterate(steps)
  self:before_iterate()
  u.printf('rel error : %g',self:relative_error())
  self.iterations = steps
  self:initialize_plotting()
  local valid_gradients = 0
  local mod_error, overlap_error, relative_error, probe_error, mod_updates, im_error = -1,-1,nil, nil, 0
  u.printf('%-10s%-15s%-15s%-13s%-15s%-15s%-15s%-15s%-15s%-15s','iteration','L','R','R (%)','||dL/dO||','||dL/dP||','mu', 'e_rel', 'e_img','valid')
  print('---------------------------------------------------------------------------------------------------------------------------------')

  self.counter = 0
  local t = sys.clock()
  sys.tic()
  local it_no_progress = 0
  for i = 1, steps do
    self:update_iteration_dependent_parameters(i)

    self.L_error[i] = self.L:updateOutput(self.z,self.a)
    self.dL_dz, valid_gradients = self.L:updateGradInput(self.z,self.a)

    self.old_O = self.O:clone()
    local O,L,k = self.optimizer(fn.partial(self.optim_func_object,self),self.O,self.optim_config,self.optim_state)
    local dL_dO_1norm = self.dL_dO:normall(1)
    -- plt:plot(self.O[1][1]:zfloat(),'O 1')
    -- print()
    -- print('Number of function evals = ',i)
    -- print('L=')
    -- for j=1,#L do print(j,L[j]); end
    -- print()
    self.L_error[i] = L[#L]
    self.R_error[i] = 0
    -- plt:hist(self.dL_dO:abs():float():view(self.dL_dO:nElement()),'dl_do')
    -- plt:plot(self.O[1][1]:zfloat(),'O 1')
    -- plt:plot(self.dL_dO[1][1]:zfloat(),'self.dL_dO')
    -- self.O:add(self.dL_dO)
    -- plt:plot(self.O[1][1]:zfloat(),'O 2')
    if self.update_probe then
      self:calculate_dL_dP(self.dL_dP)
      -- plt:plot(self.dL_dP[1][1]:zfloat(),'self.dL_dP')
      self.P:add(- self:mu(i),self.dL_dP)
      self:calculateO_denom()
    end

    self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)

    if self.has_solution then
      self.rel_error[i] = self:relative_error()
    end
    self.img_error[i] = self:image_error()

    local rel = 100.0*self.R_error[i]/(self.L_error[i]+self.R_error[i])
    u.printf('%-10d%-15g%-15g%-10.2g%%  %-15g%-15g%-15g%-15g%-15g%-15g',i,self.L_error[i],self.R_error[i],rel,dL_dO_1norm,self.dL_dP:normall(1),self:mu(i),self.rel_error[i],self.img_error[i],valid_gradients/self.total_measurements*100.0)

    self:maybe_plot()
    self:maybe_save_data()

    if i>1 and math.abs(self.img_error[i] - self.img_error[i-1]) < self.stopping_threshold then
      it_no_progress = it_no_progress + 1
    end
    if it_no_progress == 3 then
      -- it_no_progress = it_no_progress + 1
      -- self.optim_config = {}
      -- -- self.optim_config.maxIter = 10
      -- -- self.optim_config.sig = 0.5
      -- -- self.optim_config.red = 1e8
      -- -- self.optim_config.break_on_success = true
      -- -- self.optim_config.verbose = false
      -- self.optim_state = {}
      -- self.optim_config.learningRate = 1e-3
      -- self.optim_config.learningRateDecay = 1e-2
      -- self.optim_config.weightDecay = 0
      -- self.optim_config.momentum = 0.9
      -- self.optim_config.dampening = nil
      -- self.optimizer = optim.nag
      break
    end
    -- grad = fun_compute_grad_TPWFP_Real(z, y, Params, A, At, Masks, n1_LR, n2_LR, fmaskpro);
    -- z = z - muf(t) * grad;             % Gradient update

  end
  t = sys.toc()
  self.time = t
  self:save_data(self.save_path .. self.run_label .. '_TWF_' .. (self.i+1))
  -- self.do_plot = true
  self:maybe_plot()
  plt:shutdown_reconstruction_plot()
  collectgarbage()
end

return TWF_engine
