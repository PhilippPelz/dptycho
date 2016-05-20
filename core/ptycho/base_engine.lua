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
local tablex = require "pl.tablex"
local paths = require "paths"
require 'hdf5'
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


local engine = classic.class(...)
--pos,a,nmodes_probe,nmodes_object,solution,probe,dpos,fmask
function engine:_init(par)
  local par = self:replace_default_params(par)
  tablex.update(self,par)

  self.par = par.par_save

  if not paths.dirp(self.save_path) then
    paths.mkdir(self.save_path)
  end

  local min = par.pos:min(1)
  -- pprint(min)
  self.pos = par.pos:add(-1,min:expandAs(par.pos)):add(1)

  self.fm = par.fmask:expandAs(self.a)

  self.K = par.a:size(1)
  self.M = par.a:size(2)
  self.MM = self.M*self.M

  local x = torch.repeatTensor(torch.linspace(-self.M/2,self.M/2,self.M),self.M,1)
  -- pprint(x)
  local y = x:clone():t()
  self.r2 = (x:pow(2) + y:pow(2))

  self.has_solution = par.solution and true
  self.a_exp = self.a:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
  self.fm_exp = self.fm:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)

  -- dimensions used for probe and object modes in z[k]
  self.O_dim = 1
  self.P_dim = 2

  self.margin = par.margin
  local object_size = par.pos:max(1):add(torch.IntTensor({par.a:size(2) + 2*self.margin,par.a:size(3)+ 2*self.margin})):squeeze()
  self.pos:add(self.margin)
  self.Nx = object_size[1] - 1
  self.Ny = object_size[2] - 1
  -- pprint( pos:max(1))
  -- pprint(object_size)
  -- pprint(self.solution)
  if par.probe then
    self.P = par.probe
  end

  u.printram('before allocateBuffers')

  self:allocateBuffers(self.K,self.No,self.Np,self.M,self.Nx,self.Ny)

  u.printram('after allocateBuffers')

  if self.has_solution then
    local startx, starty = 1,1
    local endx = self.Nx-2*self.margin
    local endy = self.Ny-2*self.margin
    -- {},{},
    local slice = {{startx,endx},{starty,endy}}
    pprint(slice)
    pprint(par.solution)
    local sv = par.solution[slice]:clone()
    pprint(sv)
    self.solution:zero()

    self.solution[{{},{},{startx+self.margin,endx+self.margin},{starty+self.margin,endy+self.margin}}]:copy(sv)
    -- plt:plot(self.solution[1][1]:zfloat(),'solution')
  end

  local probe_size = self.P:size():totable()
  local support = znn.SupportMask(probe_size,probe_size[#probe_size]/4)

  local re = stats.truncnorm({self.No,self.Nx,self.Ny},0,1,1e-1,1e-2):cuda()
  local Pre = stats.truncnorm({self.M,self.M},0,1,1e-1,1e-2):cuda()
  self.O = self.O:zero():copyRe(re):add(1+0i)
  self.O_denom:zero()
  self.P:zero()
  self.P[1]:add(300+0i)
  for i=2,self.Np do
    self.P[1][i]:copyRe(Pre):copyIm(Pre)
    -- self.P[1][i]:add(150+0i)
  end
  self.P = support:forward(self.P)

  self.O_views = {}
  self.O_tmp_PF_views = {}
  self.O_tmp_PQ_views = {}
  self.O_denom_views = {}
  self.err_hist = {}

  self.max_power = self.a_tmp2_real_PQstore:cmul(self.a,self.fm):pow(2):sum(2):sum(3):max()
  self.total_power = self.a_tmp2_real_PQstore:cmul(self.a,self.fm):pow(2):sum()
  self.total_measurements = self.fm:sum()
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
  self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)
  self.norm_a = self.a_tmp2_real_PQstore:cmul(self.a,self.fm):pow(2):sum()
  local P_fluence = self.P_tmp1_real_PQstore:normZ(self.P):sum()

  if self.probe_support then
    self.support = znn.SupportMask(probe_size,probe_size[#probe_size]*self.probe_support)
  else
    self.support = nil
  end
  -- self.bandwidth_limit = znn.SupportMask(probe_size,200)
  --
  -- self.P:mul(math.sqrt(self.max_power/P_fluence))
  -- print(P_fluence,max_measured_I,self.P_tmp1_real_PQstore:normZ(self.P):sum())
  -- plt:plot(self.P[1][1]:zfloat(),'1probe - it '..0)
  -- plt:plot(self.P[1][2]:zfloat(),'2probe - it '..0)
  -- plt:plot(self.P[1][3]:zfloat(),'3probe - it '..0)

--   obj_Npix = obj.size
-- expected_obj_var = obj_Npix / tot_power  # Poisson
-- reg_rescale  = tot_measpts / (8. * obj_Npix * expected_obj_var)
-- verbose(2, 'Rescaling regularization amplitude using the Poisson distribution assumption.')
-- verbose(2, 'Factor: %8.5g' % reg_rescale)
-- reg_del2_amplitude *= reg_rescale
  local expected_obj_var = self.O:nElement() / self.total_power
  self.rescale_regul_amplitude = self.total_measurements / (8*self.O:nElement()*expected_obj_var)

  print(   '----------------------------------------------------')
  u.printf('K = %d',self.K)
  u.printf('N = (%d,%d)',self.Nx,self.Ny)
  u.printf('M = %d',self.M)
  u.printf('power threshold is                      %g',self.power_threshold)
  u.printf('total measurements:                     %g',self.total_measurements)
  u.printf('total measurements/image_pixels:        %g',self.total_measurements/self.O:nElement())
  u.printf('total power:                            %g',self.total_power)
  u.printf('maximum power:                          %g',self.max_power)
  u.printf('rescale_regul_amplitude:                %g',self.rescale_regul_amplitude)
  print(   '----------------------------------------------------')
  -- u.printMem()
  u.printram('after init')
end

function engine.P_Mod(x,abs,measured_abs)
  x.THNN.P_Mod(x:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
end

function engine.P_Mod_bg(x,fm,bg,a,af)
  x.THNN.P_Mod_bg(x:cdata(),fm:cdata(),bg:cdata(),a:cdata(),af:cdata(),1)
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

function engine:maybe_copy_new_batch_z(k)
  u.printram('before maybe_copy_new_batch_z')
  self:maybe_copy_new_batch(self.z,self.z_h,'z',k)
  -- collectgarbage()
  -- u.printram('after maybe_copy_new_batch_z')
end

function engine:maybe_copy_new_batch_P_Q(k)
  self:maybe_copy_new_batch(self.P_Qz,self.P_Qz_h,'P_Qz',k)
end

function engine:maybe_copy_new_batch_P_F(k)
  -- u.printf('engine:maybe_copy_new_batch_P_F(%d)',k)
  self:maybe_copy_new_batch(self.P_Fz,self.P_Fz_h,'P_Fz',k)
end

function engine:maybe_copy_new_batch_all(k)
  self:maybe_copy_new_batch_z(k)
  self:maybe_copy_new_batch_P_Q(k)
  self:maybe_copy_new_batch_P_F(k)
end

function engine:same_batch(batch_params1,batch_params2)
  local batch_start1, batch_end1, batch_size1 = table.unpack(batch_params1)
  local batch_start2, batch_end2, batch_size2 = table.unpack(batch_params2)
  return batch_start1 == batch_start2 and batch_end1 == batch_end2 and batch_size1 == batch_size2
end

function engine:maybe_copy_new_batch(z,z_h,key,k)
  if (k-1) % self.batch_size == 0 and self.batches > 1 then
    local batch = math.floor(k/self.batch_size) + 1

    local oldparams = self.old_batch_params[key]
    self.old_batch_params[key] = self.batch_params[batch]
    local batch_start, batch_end, batch_size = table.unpack(self.batch_params[batch])
    u.debug('----------------------------------------------------------------------')
    -- u.debug('batch '..batch)
    u.debug('%s: s, e, size             = (%03d,%03d,%03d)',key,batch_start, batch_end, batch_size)

    if oldparams then
      local old_batch_start, old_batch_end, old_batch_size = table.unpack(oldparams)
      u.debug('%s: old_s, old_e, old_size = (%03d,%03d,%03d)',key,old_batch_start, old_batch_end, old_batch_size)
      if not self:same_batch(oldparams,self.batch_params[batch]) then
        z_h[{{old_batch_start,old_batch_end},{},{},{},{}}]:copy(z[{{1,old_batch_size},{},{},{},{}}])
        z[{{1,batch_size},{},{},{},{}}]:copy(z_h[{{batch_start,batch_end},{},{},{},{}}])
      end
    else
      z[{{1,batch_size},{},{},{},{}}]:copy(z_h[{{batch_start,batch_end},{},{},{},{}}])
    end
  end
  -- u.printram('after maybe_copy_new_batch')
end

function engine:prepare_plot_data()
  self.O_hZ:copy(self.O)
  self.P_hZ:copy(self.P)
  self.plot_pos:add(self.pos:float(),self.dpos)
  self.bg_h:copy(self.bg)
end

function engine:maybe_plot()
  if self.do_plot then
    -- :cmul(self.O_mask)
    self.O_tmp_PFstore:copy(self.O)--:cmul(self.O_mask)
    self.P_hZ:copy(self.P)
    self.O_hZ:copy(self.O_tmp_PFstore)
    for n = 1, self.No do
      local title = self.i..'_O_'..n
      plt:plot(self.O_hZ[n][1],title,self.save_path ..title,self.show_plots)
    end
    for n = 1, self.Np do
      local title = self.i..'_P_'..n
      plt:plot(self.P_hZ[1][n],title,self.save_path ..title,self.show_plots)
      -- ,self.show_plots
    end
    -- plt:plot(self.bg:float(),'bg')
    -- self:prepare_plot_data()
    -- plt:update_reconstruction_plot(self.plot_data)
  end
end

function engine:initialize_plotting()
  self.bg_h = torch.FloatTensor(self.bg:size())
  self.O_hZ = torch.ZFloatTensor(self.O:size())
  self.P_hZ = torch.ZFloatTensor(self.P:size())
  -- local O_hZ_store = self.O_hZ:storage()
  -- local P_hZ_store = self.P_hZ:storage()
  --
  -- local z2_pointer = tonumber(self.O_hZ:data())
  -- local z3_pointer = tonumber(self.P_hZ:data())
  --
  -- local O_hZ_store_real = torch.FloatStorage(#O_hZ_store*2,z2_pointer)
  -- local P_hZ_store_real = torch.FloatStorage(#P_hZ_store*2,z3_pointer)
  -- local os = self.O:size():totable()
  -- os[#os+1] = 2
  -- local ps = self.P:size():totable()
  -- ps[#ps+1] = 2
  -- self.O_h = torch.FloatTensor(O_hZ_store_real,1,torch.LongStorage(os))
  -- self.P_h = torch.FloatTensor(P_hZ_store_real,1,torch.LongStorage(ps))
  self.mod_errors = torch.FloatTensor(self.iterations)
  self.overlap_errors = torch.FloatTensor(self.iterations)
  self.plot_pos = self.dpos:clone()

  self:prepare_plot_data()
  -- self.O_h:zero()
  -- print(self.P_h:max(),self.P_h:min())
  -- print(self.O_h:max(),self.O_h:min())

  self.plot_data = {
    self.O_hZ:abs(),
    self.O_hZ:arg(),
    self.P_hZ:re(),
    self.P_hZ:im(),
    self.bg_h,
    self.plot_pos,
    self.mod_errors,
    self.overlap_errors
  }
  -- plt:init_reconstruction_plot(self.plot_data)
  -- print('here')
  u.printram('after initialize_plotting')
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
  if not self.P then
    self.P = torch.ZCudaTensor.new(1,Np,M,M)
  end

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

  if self.has_solution then
    self.solution = torch.ZCudaTensor.new(No,1,Nx,Ny)
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

  if self.batches > 1 then
    self.z_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    self.P_Qz_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
    self.P_Fz_h = torch.ZFloatTensor.new(torch.LongStorage{self.K,No,Np,M,M})
  end

  self.old_batch_params = {}
  self.old_batch_params['z'] = self.batch_params[1]
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
  self.a_tmp3_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{1,1,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.a_tmp3_real_PQstore:nElement() + 1

  self.zk_tmp1_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp1_real_PQstore:nElement() + 1
  self.zk_tmp2_real_PQstore = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{No,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.zk_tmp2_real_PQstore:nElement() + 1

  if P_Qstorage_offset + self.K*M*M + 1 > self.P_Qz:nElement() * 2 then
    self.d = torch.CudaTensor(torch.LongStorage{self.K,M,M})
  else
    self.d = torch.CudaTensor(P_Qz_storage_real,P_Qstorage_offset,torch.LongStorage{self.K,M,M})
    P_Qstorage_offset = P_Qstorage_offset + self.d:nElement() + 1
  end

  self.a_tmp2_real_PQstore = torch.CudaTensor(P_Qz_storage_real,1,torch.LongStorage{self.K,M,M})

  -- buffers in P_Fz_storage

  self.P_tmp1_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp1_PFstore:nElement() + 1
  self.P_tmp2_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp2_PFstore:nElement() + 1
  self.P_tmp3_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{1,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.P_tmp3_PFstore:nElement() + 1

  self.zk_tmp1_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp1_PFstore:nElement() + 1
  self.zk_tmp2_PFstore = torch.ZCudaTensor.new(P_Fz_storage,P_Fstorage_offset,{No,Np,M,M})
  P_Fstorage_offset = P_Fstorage_offset + self.zk_tmp2_PFstore:nElement() + 1
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
  self.O_tmp_real_PFstore = torch.CudaTensor.new(P_Fz_storage_real,P_Fstorage_offset,torch.LongStorage{No,1,Nx,Ny})
  P_Fstorage_offset = P_Fstorage_offset + self.O_tmp_real_PFstore:nElement() + 1
end

function engine:update_iteration_dependent_parameters(it)
  self.i = it
  self.update_probe = it >= self.probe_update_start
  self.update_positions = it >= self.position_refinement_start and it % self.position_refinement_every == 0
  self.do_plot = it % self.plot_every == 0 and it > self.plot_start
  self.calculate_new_background = self.i >= self.background_correction_start
  self.do_save_data = it % self.save_interval == 0
  if self.probe_lowpass_fwhm(it) then
    local conv = self.r2:clone():div(-2*(50/2.35482)^2):exp():cuda():fftshift()
    self.probe_lowpass = torch.ZCudaTensor.new(self.r2:size()):zero()
    self.probe_lowpass:copyRe(self.r2:clone():sqrt():lt(self.probe_lowpass_fwhm(it)):cuda()):fft()
    self.probe_lowpass:cmul(conv)
    self.probe_lowpass:ifft():fftshift()
    self.probe_lowpass:mul(1/self.probe_lowpass:max())
    -- :div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    -- self.probe_lowpass = self.r2:clone():div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    self.probe_lowpass = self.probe_lowpass:view(1,1,self.M,self.M):expand(1,self.Np,self.M,self.M)
    local title = self.i..' probe_lowpass'
    local savepath = self.save_path .. title
    plt:plot(self.probe_lowpass[1][1]:re():float(),title,savepath,self.show_plots)
  end
  if self.object_highpass_fwhm(it) then
    self.object_highpass = self.r2:clone():fill(1):add(-1,self.r2:clone():div(-2*(self.object_highpass_fwhm(it)/2.35482)^2):exp()):cuda():fftshift()
    self.object_highpass = self.object_highpass:view(1,1,self.M,self.M):expand(self.No,1,self.M,self.M)
  end
  if self.fm_support_radius(it) then
    self.fm_support = self.r2:lt(self.fm_support_radius(it)):cuda():fftshift()
    self.fm_support = self.fm_support:view(1,1,self.M,self.M):expand(self.No,self.Np,self.M,self.M)
  end
end

function engine:maybe_save_data()
  if self.do_save_data then
    self:save_data(self.save_path .. 'ptycho_' .. self.i)
  end
end

function engine:save_data(filename)
  u.printf('Saving at iteration %03d to file %s',self.i,filename .. '.h5')
  local f = hdf5.open(filename .. '.h5','w')
  f:write('/pr',self.P:zfloat():re())
  f:write('/pi',self.P:zfloat():im())
  if self.solution then
    f:write('/solr',self.solution:zfloat():re())
    f:write('/soli',self.solution:zfloat():im())
  end
  f:write('/or',self.O:zfloat():re())
  f:write('/oi',self.O:zfloat():im())
  if self.bg_solution then
    f:write('/bg',self.bg_solution:float())
  end
  f:write('/scan_info/positions_int',self.pos)
  f:write('/scan_info/positions',self.pos:clone():float():add(self.dpos))
  f:write('/scan_info/dpos',self.dpos)
  -- f:write('/parameters',self.par)
  f:write('/data_unshift',self.a:float())
  f:close()
end

function engine:generate_data(filename)
  self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)

  local a = self.a_tmp_real_PQstore

  for _, params1 in ipairs(self.batch_params) do
    local batch_start1, batch_end1, batch_size1 = table.unpack(params1)
    self:maybe_copy_new_batch_z(batch_start1)
    local first = true
    for k = 1, batch_size1 do
      local k_full = k + batch_start1 - 1
      print(k_full)
      a:zero()
      for o = 1,self.No do
        -- plt:plotcompare({self.z[k][1]:re():float(),self.z[k][1]:im():float()},'self.z[k]')
        if first then
          plt:plot(self.z[k][o][1]:abs():log():float(),'self.z[k][o][1]')
        end
        local abs = self.z[k][o]:fftBatched():sum(1)
        if first then
          plt:plot(self.z[k][o][1]:abs():log():float(),'self.z[k][o][1] fft')
        end
        abs = abs[1]:abs()
        abs:pow(2)
        a:add(abs)
      end
      if first then
        plt:plot(self.bg_solution:float(),'bg_solution')
      end
      if self.bg_solution then
        a:add(self.bg_solution)
      end
      a:sqrt()
      if first then
        plt:plot(a[1][1]:float():log(),'a[1][1]')
        first = false
      end
      self.a[k_full]:copy(a)
    end
  end
  plt:plot(self.P[1][1]:re():float(),'P')
  plt:plot(self.O[1][1]:re():float(),'O')
  self:save_data(filename)
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
    self:maybe_copy_new_batch_z(k)
    local ind = self.k_to_batch_index[k]
    mul_merge_repeated:conj(mul_merge:expandAs(self.z[ind]))
    view:add(mul_merge_repeated:cmul(self.z[ind]):sum(self.P_dim))
  end
  merge_memory:cmul(self.O_denom)
end

-- P_Fz free
function engine:update_frames(z,mul_split,merge_memory_views,batch_copy_func)
  for k, view in ipairs(merge_memory_views) do
    batch_copy_func(self,k)
    local ind = self.k_to_batch_index[k]
    z[ind]:cmul(view:expandAs(z[ind]),mul_split:expandAs(z[ind]))
  end
end

function engine:P_Q_plain()
  self:merge_frames(self.P,self.O,self.O_views)
  self:update_frames(self.P_Qz,self.P,self.O_views,self.maybe_copy_new_batch_P_Q)
end

function engine:Del_regularize(target,amplitude,tmp,result)
  result:zero()
  local d = tmp

  d:dx_bw(target)
  result:add(d)
  plt:plot(d[1][1]:zfloat(),'dx_bw',self.save_path ..'dx_bw')

  d:dy_bw(target)
  result:add(d)
  plt:plot(d[1][1]:zfloat(),'dy_bw',self.save_path ..'dx_bw')

  d:dx_fw(target)
  result:add(-1,d)
  plt:plot(d[1][1]:zfloat(),'dx_fw',self.save_path ..'dx_bw')

  d:dy_fw(target)
  result:add(-1,d)
  plt:plot(d[1][1]:zfloat(),'dy_fw',self.save_path ..'dx_bw')

  result:mul(2*amplitude*self.rescale_regul_amplitude)
  return result
end

function engine:P_Q()
  if self.update_probe then
    for _ = 1,self.P_Q_iterations do
      self.j = _
      self:merge_frames(self.P,self.O,self.O_views)
      probe_change = self:refine_probe()
      -- or probe_change > 0.97 * last_probe_change
      if not probe_change_0 then probe_change_0 = probe_change end
      if probe_change < .1 * probe_change_0  then break end
      -- last_probe_change = probe_change
      -- u.printf('            probe change : %g',probe_change)
    end
    self:update_frames(self.P_Qz,self.P,self.O_views,self.maybe_copy_new_batch_P_Q)
  else
    self:P_Q_plain()
  end
end

function engine:P_F_with_background()
  print('P_F_with_background ' .. self.i)
  local z = self.P_Fz
  local abs = self.zk_tmp1_real_PQstore
  local da = self.a_tmp_real_PQstore
  local sum_a = self.a_tmp_real_PQstore[1][1]
  local fdev = self.a_tmp3_real_PQstore
  local a_pow2 = self.a_tmp3_real_PQstore[1][1]
  local d = self.d
  local eta = self.eta
  local err_fmag, renorm  = 0, 0
  local batch_start, batch_end, batch_size = table.unpack(self.old_batch_params['P_Fz'])
  -- u.printf('batch_start = %d',batch_start)

  eta:zero()
  sum_a:zero()
  if self.calculate_new_background then
    -- print('calculate_new_background')
    for _, params1 in ipairs(self.batch_params) do
      local batch_start1, batch_end1, batch_size1 = table.unpack(params1)
      -- u.printf('batch_start1 = %d',batch_start1)
      self:maybe_copy_new_batch_P_F(batch_start1)
      local first = true
      for k = 1, batch_size1 do
        local k_full = k + batch_start1 - 1
        -- print(k_full)
        for o = 1, self.No do
          if first then
            -- plt:plot(z[k][o][1]:abs():log():float(),'z[k][o][1][1]')
          end
          z[k][o]:fftBatched()
          if first then
            -- plt:plot(z[k][o][1]:abs():log():float(),'z[k][o][1][1]')
          end
        end
        abs = abs:normZ(z[k]):sum(self.O_dim):sum(self.P_dim)
        if first then
          -- plt:plot(abs[1][1]:float():log(),'abs[1][1]')
          first = false
        end
        -- sum_a = sum_k |F z_k|^2
        sum_a:add(abs[1][1])
        -- d_k = sum_k (|F z_k|^2 - background)
        d[k_full]:add(a_pow2:pow(self.a[k_full],2),-1,self.bg)
        -- eta = sum_k d_k * |F z_k|^2
        eta:add(abs[1][1]:cmul(d[k_full]))
      end
    end
    -- local title = self.i..'_d'
    -- plt:plot(d[2]:float():log(),title,self.save_path..title,true)
    -- plt:plot(sum_a:float():log(),'sum_a 1')
    d:pow(2)
    -- local title = self.i..'_d^2'
    -- plt:plot(d[2]:float():log(),title,self.save_path..title,true)
    -- eta = sum_k d_k * |F z_k|^2 / sum_k d_k^2
    local dsum = d:sum(1)
    eta:cdiv(dsum):clamp(0.8,1)
    -- plt:plot(eta:float(),'eta')
    -- sum_a = (sum_k |F z_k|^2) / eta
    sum_a:cdiv(eta)
    local title = self.i..'_sum_a'
    plt:plot(sum_a:float():log(),title,self.save_path..title,true)
    -- plt:plot(sum_a:float():log(),'sum_a 2')
    d:sqrt()
    -- sum_a = (sum_k |F z_k|^2) / eta - (sum_k d_k)
    local dsum = d:sum(1)
    sum_a:add(-1,dsum)
    local title = self.i..'_sum_a_minus_dsum'
    plt:plot(sum_a:float():log(),title,self.save_path..title,true)
    -- sum_a = 1/K [ (sum_k d_k) - (sum_k |F z_k|^2) / eta ]
    sum_a:mul(-1/self.K)
    -- plt:plot(sum_a:float(),'sum_a 3')
    self.bg:add(sum_a)
    local title = self.i..'_bg_unclamped'
    plt:plot(self.bg:float():log(),title,self.save_path..title,true)
    self.bg:clamp(0,1e10)
    local title = self.i..'_bg_clamped'
    plt:plot(self.bg:float():log(),title,self.save_path..title,true)

    self.calculate_new_background = false

    self:maybe_copy_new_batch_P_F(batch_start)
  end
  local first = true
  for k=1,batch_size do
    local k_all = batch_start+k-1
    -- sum over probe and object modes - 1x1xMxM
    abs = abs:normZ(z[k]):sum(self.O_dim):sum(self.P_dim)
    abs:sqrt()
    fdev[1][1]:add(abs[1][1],-1,self.a[k_all])
    -- plt:plot(fdev[1][1]:float():log(),'fdev')
    da:abs(fdev):pow(2):cmul(self.fm[k_all])
    err_fmag = da:sum()
    -- u.printf('err_fmag = %g',err_fmag)
    self.module_error = self.module_error + err_fmag
    if err_fmag > self.power_threshold then
      renorm = math.sqrt(self.power_threshold/err_fmag)
      -- plt:plot(z[ind][1][1]:abs():float():log(),'z_out[ind][1] before')
      if first then
        -- plt:plot(self.bg_exp[k_all][1][1]:float(),'self.bg_exp[k_all]')
        -- plt:plot(self.a_exp[k_all][1][1]:abs():float():log(),'self.a_exp[k_all]')
        -- plt:plot(abs[1][1]:float():log(),'abs')
      end
      self.P_Mod_bg(z[k],self.fm_exp[k_all],self.bg_exp[k_all],self.a_exp[k_all],abs:expandAs(z[k]))
      -- self.P_Mod_renorm(z[k],self.fm_exp[k_all],fdev,self.a_exp[k_all],abs:expandAs(z[k]),renorm)
      if first then
        -- plt:plot(z[k][1][1]:abs():float():log(),'z_out[k][1] after')
        first = false
      end
      -- self.P_Mod_renorm(z[k],self.fm_exp[k_all],fdev,self.a_exp[k_all],abs:expandAs(z[k]),renorm)
      -- self.P_Mod(z_out[ind],abs,self.a[k])

      self.mod_updates = self.mod_updates + 1
    end
  end

  for k=1,batch_size do
    for o = 1, self.No do
      z[k][o]:ifftBatched()
    end
  end
  self.module_error = self.module_error / self.norm_a
  return self.module_error, self.mod_updates
end

function engine:P_F()
  if self.i >= self.background_correction_start then
    self:P_F_with_background()
  else
    self:P_F_without_background()
  end
end

function engine:P_F_without_background()
  print('P_F_without_background ' .. self.i)
  local z = self.P_Fz
  local abs = self.zk_tmp1_real_PQstore
  local da = self.a_tmp_real_PQstore
  local fdev = self.a_tmp3_real_PQstore
  local err_fmag, renorm  = 0, 0
  local batch_start, batch_end, batch_size = table.unpack(self.old_batch_params['P_Fz'])
  -- pprint(da)
  -- pprint(fdev)
  -- print('loop')
  local plots = 0
  local f = hdf5.open(self.save_path..self.i..'_abs.h5','w')
  for k=1,batch_size do
    for o = 1, self.No do
      -- print('P_F_without_background before fft')
      z[k][o]:fftBatched()
    end
    local k_all = batch_start+k-1
    -- sum over probe and object modes - 1x1xMxM
    abs = abs:normZ(z[k]):sum(self.O_dim):sum(self.P_dim)
    abs:sqrt()

    if plots < 5 and k % 10 == 0 then

      -- {name="self", type='table'},
      -- {name="imgs", type='table'},
      -- {name="suptitle", default='Image', type='string'},
      -- {name="savepath", default=py.None, type='string'},
      -- {name="show", default=true, type='boolean'},
      -- {name="cmap", default={'hot','hot'}, type='table'},
      -- {name="title", default={'Img1','Img2'}, type='table'},
      local title = self.i..'_abs_'..plots
      local ab = abs[1][1]:clone():fftshift():float():log()
      local ak = self.a[k]:clone():fftshift():float():log()
      plt:plotcompare({ab,ak},title,self.save_path ..title,self.show_plots)
      f:write('/ab_'..plots,ab)
      f:write('/ak_'..plots,ak)

      plots = plots + 1
    end

    fdev[1][1]:add(abs[1][1],-1,self.a[k_all])
    -- plt:plot(fdev[1][1]:float():log(),'fdev')
    da:abs(fdev):pow(2)
    da:cmul(self.fm[k_all])
    err_fmag = da:sum()
    -- u.printf('err_fmag = %g',err_fmag)
    self.module_error = self.module_error + err_fmag
    if err_fmag > self.power_threshold then
      renorm = math.sqrt(self.power_threshold/err_fmag)
      -- plt:plot(z[ind][1][1]:abs():float():log(),'z_out[ind][1] before')
      if self.fm_support_radius(self.i) then
        self.fm_support:cmul(self.fm_exp[k_all])
      else
        self.fm_support = self.fm_exp[k_all]
      end
      -- [k_all]
      -- self.P_Mod_renorm(z[k],self.fm_support,fdev:expandAs(z[k]),self.a_exp[k_all],abs:expandAs(z[k]),renorm)
      self.P_Mod(z[k],abs:expandAs(z[k]),self.a_exp[k_all])
      -- plt:plot(z[ind][1][1]:abs():float():log(),'z_out[k][1] after')
      self.mod_updates = self.mod_updates + 1
    end
    for o = 1, self.No do
      -- print('P_F_without_background before ifft')
      z[k][o]:ifftBatched()
    end
  end
  f:close()
  -- u.printMem()
  -- plt:plot(self.P_Fz[1][1][1]:zfloat(),'self.z')
  -- plt:plot(self.P_Fz[2][1][1]:zfloat(),'self.z2')
  -- plt:plot(self.P_Fz[3][1][1]:zfloat(),'self.z3')
  -- plt:plot(self.P_Fz[4][1][1]:zfloat(),'self.z4')
  -- u.printf('%d/%d modulus updates', mod_updates, self.K)

  self.module_error = self.module_error / self.norm_a
  return self.module_error, self.mod_updates
end

function engine:maybe_refine_positions()
  if self.update_positions then
    self:refine_positions()
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
    self:maybe_copy_new_batch_z(k)
    local ind = self.k_to_batch_index[k]
    local view_exp = view:expandAs(self.z[ind])
    oview_conj:conj(view_exp)
    new_P:add(oview_conj:cmul(self.z[ind]):sum(self.O_dim))
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
  -- print('overlap_error')
  local result = self.P_Fz
  local res, res_denom = 0, 0

  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch_P_Q(k)
    self:maybe_copy_new_batch_z(k)
    res = res + result:add(z_in,-1,z_out):norm():sum()
    res_denom = res_denom + result:norm(z_out):sum()
  end
  return res/res_denom
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
  if self.probe_solution then
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
