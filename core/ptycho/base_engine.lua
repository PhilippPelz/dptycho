local classic = require 'classic'
local nn = require 'nn'
local zt = require "ztorch.fcomplex"
local pprint = require "pprint"

local znn = require 'dptycho.znn'
local stats = require 'dptycho.util.stats'
local params = require "dptycho.core.ptycho.params"
local ptycho = require "dptycho.core.ptycho"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()


local tablex = require "pl.tablex"
local paths = require "paths"

local fn = require 'fn'
local ztorch = require 'ztorch'


require 'parallel'
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
-- relative_error
-- probe_error
-- mustHave("iterate")


local engine = classic.class(...)
--pos,a,nmodes_probe,nmodes_object,object_solution,probe,dpos,fmask
function engine:_init(par1)
  tablex.update(self,par1)
  self.dx = u.physics(self.experiment.E):real_space_resolution(self.experiment.z,self.experiment.det_pix,self.experiment.N_det_pix)
  if not paths.dirp(self.save_path) then
    paths.mkdir(self.save_path)
    u.printf('Created path %s',self.save_path)
  end

  local min = self.pos:min(1)
  -- pprint(min)
  self.pos = self.pos:add(-1,min:expandAs(self.pos)):add(1)

  self.fm = self.fmask:expandAs(self.a)

  self.K = self.a:size(1)
  self.M = self.a:size(2)
  self.MM = self.M*self.M
  self.err_hist = {}
  self.object_inertia = self.object_inertia * self.K
  self.probe_inertia = self.probe_inertia * self.Np * self.No * self.K
  self.i = 1
  self.iterations = 1
  self.update_positions = false
  self.update_probe = false

  self.r2 = u.r2(self.M)

  self.has_solution = self.object_solution and self.probe_solution
  -- plt:plotReIm(self.object_solution[1][1]:zfloat(),'object_solution 1')
  --expanded views of a and fm
  self.a_exp = self.a:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
  self.fm_exp = self.fm:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)

  -- dimensions used for probe and object modes in z[k]
  self.O_dim = 1
  self.P_dim = 2

  local object_size = self.pos:max(1):add(torch.IntTensor({self.a:size(2) + 2*self.margin,self.a:size(3)+ 2*self.margin})):squeeze()

  -- pprint(object_size)
  self.pos:add(self.margin)
  self.Nx = object_size[1] - 1
  self.Ny = object_size[2] - 1

  self:allocateBuffers(self.K,self.No,self.Np,self.M,self.Nx,self.Ny)

  -- save reference in params for future use
  par1.O = self.O
  par1.P = self.P

  self.O_mask_initialized = false
  self.P_initialized = false

  local Ox, Oy = self.O:size(3), self.O:size(4)
  -- print(Ox,Oy)
  local x = torch.FloatTensor(Ox,1)
  x:copy(torch.linspace(-Ox/2,Ox/2,Ox))
  local x = torch.repeatTensor(x,1,Oy)
  local y = torch.repeatTensor(torch.linspace(-Oy/2,Oy/2,Oy),Ox,1):float()
  self.r2_object = (x:pow(2) + y:pow(2))

  self:initialize_views()
  self.par = nil
  self:calculate_statistics()

  u.printram('after init')
  collectgarbage()
end

function engine:calculate_statistics()
  if not self.O_mask_initialized then
    if not self.P_initialized then
      self:initialize_probe()
    end
    self:initialize_object_solution()
    if #(self.O_denom_views) < 2 then
      self:update_views()
    end
    self:calculateO_denom()
  end

  self.a_buffer1:cmul(self.a,self.fm):pow(2)
  if self.calculate_dose_from_probe then
    self.I_total = self.K * self.P_buffer:copy(self.P):norm():re():sum()
    self.I_diff_total = self.a_buffer1:sum()
    self.elastic_percentage = self.I_diff_total / self.I_total
  else
    self.I_total = self.a_buffer1:sum()
  end
  self.I_max = self.a_buffer1:sum(2):sum(3):max()
  self.total_measurements = self.fm:sum()
  self.total_nonzero_measurements = torch.gt(self.a_buffer1:cmul(self.a,self.fm),0):sum()
  self.power_threshold = 0.25 * self.fourier_relax_factor^2 * self.I_max / self.MM
  self.norm_a = self.a_buffer1:cmul(self.a,self.fm):norm(2)^2
  local expected_obj_var = self.O:nElement() / self.I_total
  self.rescale_regul_amplitude = self.total_measurements / (8*self.O:nElement()*expected_obj_var)
  self.I_mean = self.I_total/self.a:size(1)
  self.P_norm = self.P:normall(2)^2
  self.pixels_with_sufficient_exposure = self:calculate_pixels_with_sufficient_measurements()
  self.counts_per_valid_pixel = self.I_total /self.pixels_with_sufficient_exposure
  self.counts_per_pixel = self.I_total / (self.O:nElement()/self.No)
  local dx2_in_Angstrom2 = self.dx^2 / 1e-20
  self.electrons_per_angstrom2 = self.counts_per_valid_pixel / dx2_in_Angstrom2
end

function engine:print_report()
  print(   '----------------------------------------------------')
  u.printf('K (number of diff. patterns)           : %d',self.K)
  u.printf('N (object dimensions)                  : %d x %d  (%2.1f x %2.1f Angstrom)',self.Nx,self.Ny,self.Nx*self.dx*1e10,self.Ny*self.dx*1e10)
  u.printf('M (probe size)                         : %d',self.M)
  u.printf('resolution (Angstrom)                  : %g',self.dx*1e10)
  print(   '----------------------------------------------------')
  u.printf('power threshold is                     : %g',self.power_threshold)
  u.printf('total measurements                     : %g',self.total_measurements)
  u.printf('# unknowns object                      : %2.3g',self.O:nElement())
  u.printf('# unknowns probe                       : %2.3g',self.P:nElement())
  u.printf('# unknowns total                       : %2.3g',self.O:nElement() + self.P:nElement())
  u.printf('')
  u.printf('total measurements/image_pixels        : %g',
  self.total_measurements/self.pixels_with_sufficient_exposure)
  u.printf('nonzero measurements/image_pixels      : %g',
  self.total_nonzero_measurements/self.pixels_with_sufficient_exposure)
  u.printf('nonzero measurements/# unknowns        : %g',
  self.total_nonzero_measurements/(self.pixels_with_sufficient_exposure + self.P:nElement()))
  u.printf('')
  u.printf('max counts in pattern                  : %g',self.I_max)
  u.printf('rescale_regul_amplitude                : %g',self.rescale_regul_amplitude)
  u.printf('probe intensity                        : %g',self.P_norm)
  u.printf('total counts in this scan              : %g',self.I_total)
  u.printf('mean counts                            : %g',self.I_mean)
  u.printf('counts per       pixel                 : %g',self.counts_per_pixel)
  u.printf('counts per valid pixel                 : %g',self.counts_per_valid_pixel)
  if self.calculate_dose_from_probe then
    u.printf('percentage of elastically scattered e- : %g %%',self.elastic_percentage * 100)
  end
  u.printf('')
  u.printf('e- / Angstrom^2                        : %g',self.electrons_per_angstrom2)
  print(   '----------------------------------------------------')
end

-- P_Fz free
function engine:update_frames(z,mul_split,merge_memory_views,batch_copy_func)
  self.ops.Q(z,mul_split,merge_memory_views,self.zk_buffer_update_frames,self.k_to_batch_index, fn.partial(batch_copy_func,self),self.batches,self.K,self.dpos)
end

function engine:calculateO_denom()
  self.ops.calculateO_denom(self.O_denom,self.O_mask,self.O_denom_views,self.P,self.P_buffer_real,self.Pk_buffer_real,self.object_inertia,self.K,self.O_denom_regul_factor_start,self.O_denom_regul_factor_end,self.i,self.iterations,self.dpos)
  self.O_mask_initialized = true
  -- plt:plot(self.O_denom[1][1]:float():log(),'O_denom')
end

function engine:P_Q()
  if self.update_probe then
    for _ = 1,self.P_Q_iterations do
      self:merge_frames(self.z,self.P,self.O,self.O_views)
      local probe_change = self:refine_probe()
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

function engine:refine_probe()
  local probe_change = self.ops.refine_probe(self.z,self.P,self.O_views,self.P_tmp1_PQstore,self.P_tmp1_real_PQstore,self.P_tmp2_real_PQstore,self.zk_tmp1_PQstore,self.zk_tmp2_PQstore, self.zk_tmp1_real_PQstore,self.k_to_batch_index, fn.partial(self.maybe_copy_new_batch_z,self),self.dpos,self.support,self.probe_inertia)
  self:calculateO_denom()
  return probe_change/self.Np
end

function engine:P_Q_plain()
  self:merge_frames(self.z,self.P,self.O,self.O_views,true)
  plt:plot(self.O[1][1]:zfloat(),'O after merge')
  plt:plot(self.P[1][1]:zfloat(),'P after merge')
  self.ops.Q(self.P_Qz,self.P,self.O_views,self.zk_buffer_update_frames,self.k_to_batch_index,fn.partial(self.maybe_copy_new_batch_P_Q,self),self.batches,self.K,self.dpos)
end

-- P_Qz, P_Fz free to use
function engine:merge_frames(z,mul_merge, merge_memory, merge_memory_views, do_normalize_merge_memory)
  self.ops.Q_star(z, mul_merge, merge_memory, merge_memory_views, self.zk_buffer_merge_frames, self.P_buffer,self.object_inertia, self.k_to_batch_index,fn.partial(self.maybe_copy_new_batch_z,self), self.batches, self.K,self.dpos)
  -- plt:plotReIm(self.O[1][1]:zfloat(),'O after merge 0')
  if do_normalize_merge_memory then
    merge_memory:cmul(self.O_denom)
  end
  -- plt:plotReIm(self.O[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'O after merge 1')
end

function engine:before_iterate()
  self:initialize_object_solution()
  self:update_views()
  self:initialize_probe()
  self:calculateO_denom()
  self:initialize_object()
  -- plt:plot(self.O[1][1],'object after init')
  self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)
  self:print_report()
end

function engine:initialize_views()
  self.O_views = {}
  self.O_tmp_PF_views = {}
  self.O_tmp_PQ_views = {}
  self.O_denom_views = {}
  if self.has_solution then
    self.O_solution_views = {}
    self.O_mask_views = {}
  end
end

function engine:update_views()
  for i=1,self.K do
    local slice = {{},{},{self.pos[i][1],self.pos[i][1]+self.M-1},{self.pos[i][2],self.pos[i][2]+self.M-1}}
    -- pprint(slice)
    self.O_views[i] = self.O[slice]
    self.O_tmp_PF_views[i] = self.O_tmp_PFstore[slice]
    self.O_tmp_PQ_views[i] = self.O_tmp_PQstore[slice]
    self.O_denom_views[i] = self.O_denom[slice]
    if self.has_solution then
      self.O_solution_views[i] = self.object_solution[slice]
      self.O_mask_views[i] = self.O_mask[slice]
    end
  end
end

function engine:create_views(O)
  local views = {}
  for i=1,self.K do
    local slice = {{},{},{self.pos[i][1],self.pos[i][1]+self.M-1},{self.pos[i][2],self.pos[i][2]+self.M-1}}
    -- pprint(slice)
    views[i] = O[slice]
  end
  return views
end

function engine.P_Mod_bg(x,fm,bg,a,af)
  x.THNN.P_Mod_bg(x:cdata(),fm:cdata(),bg:cdata(),a:cdata(),af:cdata(),1)
end

function engine:replace_default_params(p)
  local par = params.DEFAULT_PARAMS()
  for k,v in pairs(p) do
    -- print(k,v)
    par[k] = v
  end
  return par
end

function engine.maybe_copy_new_batch_z(self,k)
  u.printram('before maybe_copy_new_batch_z')
  self:maybe_copy_new_batch(self.z,self.z_h,'z',k)
  -- collectgarbage()
  -- u.printram('after maybe_copy_new_batch_z')
end

function engine.maybe_copy_new_batch_z1(self,k)
  u.printram('before maybe_copy_new_batch_z')
  self:maybe_copy_new_batch(self.z1,self.z1_h,'z1',k)
  -- collectgarbage()
  -- u.printram('after maybe_copy_new_batch_z')
end

function engine.maybe_copy_new_batch_P_Q(self,k)
  self:maybe_copy_new_batch(self.P_Qz,self.P_Qz_h,'P_Qz',k)
end

function engine.maybe_copy_new_batch_P_F(self,k)
  -- u.printf('engine:maybe_copy_new_batch_P_F(%d)',k)
  self:maybe_copy_new_batch(self.P_Fz,self.P_Fz_h,'P_Fz',k)
end

function engine.maybe_copy_new_batch_all(self,k)
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
  if self.O_mask then
    self.O_buffer:cmul(self.O,self.O_mask)
    -- self.O_buffer:copy(self.O)
  else
    self.O_buffer:copy(self.O)
  end

  self.P_hZ:copy(self.P)
  self.O_hZ:copy(self.O_buffer)
  self.plot_pos:add(self.pos:float(),self.dpos):add(self.M/2)
  if self.bg then
    self.bg_h:copy(self.bg)
  else
    self.bg_h = torch.FloatTensor(self.M,self.M)
  end
  self.plot_data = {}
  self.plot_data[1] = self.O_hZ:abs()
  self.plot_data[2] = self.O_hZ:arg()
  self.plot_data[3] = self.P_hZ:re()
  self.plot_data[4] = self.P_hZ:im()
  self.plot_data[5] = self.bg_h
  self.plot_data[6] = self.plot_pos
  self.plot_data[7] = self:get_errors()
end

function engine:get_error_labels()
  return {'RFE','ROE','RMSE','L'}
end

function engine:maybe_plot()
  if self.do_plot then
    -- :cmul(self.O_mask)

    -- for n = 1, self.No do
    --   local title = self.i..'_O_'..n
    --   plt:plot(self.O_hZ[n][1],title,self.save_path ..title,self.show_plots,{'hot','hsv'})
    -- end
    -- for n = 1, self.Np do
    --   local title = self.i..'_P_'..n
    --   plt:plot(self.P_hZ[1][n],title,self.save_path ..title,self.show_plots,{'hot','hsv'})
    --   -- ,self.show_plots
    -- end
    -- plt:plot(self.bg:float(),'bg')
    self:prepare_plot_data()
    local title = self.run_label .. '_' .. self.i
    plt:update_reconstruction_plot(self.plot_data,self.save_path..title)
  end
end

function engine:initialize_plotting()
  if self.bg then
    self.bg_h = torch.FloatTensor(self.bg:size())
  end

  self.O_hZ = torch.ZFloatTensor(self.O:size())
  self.P_hZ = torch.ZFloatTensor(self.P:size())
  self.plot_pos = self.dpos:clone()
  self:allocate_error_history()
  self:prepare_plot_data()
  -- pprint(self.plot_data)
  if self.plot_start < self.iterations then
    plt:init_reconstruction_plot(self.plot_data,'Reconstruction',self:get_error_labels())
  end
  -- print('here')
  u.printram('after initialize_plotting')
end


function engine:initialize_probe()
  local Pre = stats.truncnorm({self.M,self.M},0,1,1e-1,1e-2):cuda()
  for i=2,self.Np do
    self.P[1][i]:copyRe(Pre):copyIm(Pre)
    self.P[1][i]:cmul(self.P[1][1]:abs())
  end

  if self.copy_probe then
    if self.probe_solution:dim() == 2 then
      self.P[1][1]:copy(self.probe_solution)
    else
      self.P:copy(self.probe_solution)
    end
  else
    self.P:zero()
    self.P[1][1]:add(300+0i)
  end

  if self.probe_support ~= 0 then
    local probe_size = self.P:size():totable()
    self.support = znn.SupportMask(probe_size,probe_size[#probe_size]*self.probe_support)
    self.P = self.support:forward(self.P)
  else
    self.support = nil
  end
end

function engine:initialize_object()
  if self.object_init == 'const' then
    self.O:zero():add(1+0i)
  elseif self.object_init == 'rand' then
    local ph = u.stats.truncnorm(self.O:size():totable(),0,0.2,0.05,0.01)
    self.O:polar(1,ph:cuda())
  elseif self.object_init == 'trunc' then
    -- local z = ptycho.initialization.truncated_spectral_estimate(self.z,self.P,self.O_denom,self.object_init_truncation_threshold,self.ops,self.a,self.z1,self.a_buffer2,self.zk_buffer_update_frames,self.P_buffer,self.O_buffer,self.batch_params,self.old_batch_params,self.k_to_batch_index,self.batches,self.batch_size,self.K,self.M,self.No,self.Np,self.pos,self.dpos,self.O_mask)
    -- -- self.O:copy(O)
    -- z:view_3D():ifftBatched()
    -- self:merge_frames(z,self.P,self.O,self.O_views,true)
    -- plt:plot(self.O[1][1]:zfloat())

    local O = ptycho.initialization.truncated_spectral_estimate_power_it(self.z,self.P,self.O_denom,self.object_init_truncation_threshold,self.ops,self.a,self.z1,self.a_buffer2,self.zk_buffer_update_frames,self.P_buffer,self.O_buffer,self.batch_params,self.old_batch_params,self.k_to_batch_index,self.batches,self.batch_size,self.K,self.M,self.No,self.Np,self.pos,self.dpos,self.O_mask)
    local angle = O:arg():float()
    -- angle = u.gf(angle,2)
    self.O:polar(torch.CudaTensor(O:size()):fill(1),angle:cuda())
    -- z:view_3D():ifftBatched()
    -- self:merge_frames(z,self.P,self.O,self.O_views,true)
    self.O_init = self.O:zfloat()
    -- plt:plot(self.O[1][1]:zfloat(),'O initialization')
  elseif self.object_init == 'gcl' then
      u.printf('gcl initialization is not implement yet')
      self.O:zero():add(1+0i)
  elseif self.object_init == 'copy' then
      self.O:copy(self.object_initial)
  end
end

function engine:initialize_object_solution()
  if self.object_solution then
    local startx, starty = 1,1
    local endx = self.Nx-2*self.margin
    local endy = self.Ny-2*self.margin
    local slice = nil
    if self.object_solution:dim() == 2 then
      slice = {{startx,endx},{starty,endy}}
      self.object_solution = self.object_solution[slice]:clone()
      self.object_solution = self.object_solution:view(1,1,self.Nx,self.Ny):expand(self.No,self.Np,self.Nx,self.Ny)
    end

    slice = {{},{},{startx,endx},{starty,endy}}
    -- pprint(slice)
    local sv = self.object_solution[slice]:clone()
    -- pprint(self.object_solution:size())
    if self.object_solution:size(2) ~= self.Nx+2*self.margin or self.object_solution:size(3) ~= self.Ny+2*self.margin then
      local os = self.object_solution
      self.object_solution = torch.ZCudaTensor().new({self.No,1,self.Nx+2*self.margin,self.Ny+2*self.margin})
      os:storage():free()
    end

    self.object_solution:zero()

    self.object_solution[{{},{},{startx+self.margin,endx+self.margin},{starty+self.margin,endy+self.margin}}]:copy(sv)
    self.slice = {{},{},{startx+self.margin,endx+self.margin},{starty+self.margin,endy+self.margin}}
    -- pprint(self.object_solution)
  end
end

function engine:allocate_probe(Np,M)
  if not self.P then
    self.P = torch.ZCudaTensor.new(1,Np,M,M)
    local Pre = stats.truncnorm({self.M,self.M},0,1,1e-1,1e-2):cuda()
    for i=2,self.Np do
      self.P[1][i]:copyRe(Pre):copyIm(Pre)
      self.P[1][i]:cmul(self.P[1][1]:abs())
    end
  elseif self.P:dim() ~= 4 then
    local tmp = self.P
    self.P = torch.ZCudaTensor.new(1,Np,M,M)
    self.P[1][1]:copy(tmp)
    local Pre = stats.truncnorm({self.M,self.M},0,1,1e-1,1e-2):cuda()
    for i=2,self.Np do
      self.P[1][i]:copyRe(Pre):copyIm(Pre)
      self.P[1][i]:cmul(self.P[1][1]:abs())
    end
  end
end

function engine:allocate_object(No,Nx,Ny)
  -- if self.O == nil then
  self.O = torch.ZCudaTensor.new(No,1,Nx,Ny)
  -- end
end
-- total memory requirements:
-- 1 x object complex
-- 1 x object real
-- 1 x probe real
-- 3 x z
function engine:allocateBuffers(K,No,Np,M,Nx,Ny)

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
  self.P_tmp3_PQstore = torch.ZCudaTensor.new( {1,Np,M,M})
  P_Qstorage_offset = P_Qstorage_offset + self.P_tmp3_PQstore:nElement() + 1

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



  self.P_buffer = self.P_tmp1_PQstore
  self.P_buffer_real = self.P_tmp1_real_PQstore
  self.zk_buffer_update_frames = self.zk_tmp1_PFstore
  self.zk_buffer_merge_frames = self.zk_tmp1_PQstore
  self.Pk_buffer_real = self.a_tmp_real_PQstore
  self.O_buffer_real = self.O_tmp_real_PFstore
  self.O_buffer = self.O_tmp_PFstore
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
    self.object_highpass = self.r2_object:clone():div(-2*(self.object_highpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    -- self.r2:clone():fill(1):add(-1,self.r2:clone():div(-2*(self.object_highpass_fwhm(it)/2.35482)^2):exp()):cuda():fftshift()
    -- pprint(self.object_highpass)
    self.object_highpass = self.object_highpass:view(1,1,self.Nx,self.Ny):expand(self.No,1,self.Nx,self.Ny)
  end
  if self.fm_support_radius(it) then
    self.fm_support = self.r2:lt(self.fm_support_radius(it)):cuda():fftshift()
    self.fm_support = self.fm_support:view(1,1,self.M,self.M):expand(self.No,self.Np,self.M,self.M)
  else
    self.fm_support = nil
  end
  if self.fm_mask_radius(it) then
    -- local conv = self.r2:clone():div(-2*(50/2.35482)^2):exp():cuda():fftshift()
    self.fm_mask = torch.CudaTensor.new(self.r2:size()):zero()
    self.fm_mask:copy(self.r2:clone():sqrt():gt(self.fm_mask_radius(it))):fftshift()--:fft()
    -- self.fm_mask:cmul(conv)
    -- self.fm_mask:ifft():fftshift()
    -- self.fm_mask:mul(1/self.fm_mask:max())
    -- :div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    -- self.probe_lowpass = self.r2:clone():div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    self.fm_mask = self.fm_mask:view(1,1,self.M,self.M):expand(self.No,self.Np,self.M,self.M)
    local title = self.i..' fm_mask'
    local savepath = self.save_path .. title
    plt:plot(self.fm_mask[1][1]:float(),title,savepath,self.show_plots)
  end
  if it >= self.regularization_params.start_denoising then
    self.denoise = true
  else
    self.denoise = false
  end
  self.do_regularize = self.denoise
end

function engine:maybe_save_data()
  if self.do_save_data then
    self:save_data(self.save_path .. '_it_' .. self.i)
  end
end

function engine:save_data(filename)
  u.printf('Saving at iteration %03d to file %s',self.i,filename .. '.h5')
  local f = hdf5.open(filename .. '.h5')
  local options = hdf5.DataSetOptions()
  options:setChunked(1,1,500, 500)
  options:setDeflate(8)

  f:write('/pr',self.P:zfloat():re(),options)
  f:write('/pi',self.P:zfloat():im(),options)

  if self.slice then
    local O = self.O[self.slice]:zfloat()
    f:write('/or',O:re(),options)
    f:write('/oi',O:im(),options)

    if self.object_solution then
      local s = self.object_solution[self.slice]
      :zfloat()
      f:write('/solr',s:re())
      f:write('/soli',s:im())
    end
  else
    f:write('/or',self.O:zfloat():re(),options)
    f:write('/oi',self.O:zfloat():im(),options)

    if self.object_solution then
      local s = self.object_solution:zfloat()
      f:write('/solr',s:re())
      f:write('/soli',s:im())
    end
  end
  if self.O_init then
    f:write('/oinitr',self.O_init:re(),options)
    f:write('/oiniti',self.O_init:im(),options)
  end

  if self.bg_solution then
    f:write('/bg',self.bg_solution:float())
  end
  f:write('/scan_info/positions_int',self.pos:clone():float())
  f:write('/scan_info/positions',self.pos:clone():float():add(self.dpos))
  f:write('/scan_info/dpos',self.dpos)
  -- f:write('/parameters',self.par)
  if self.save_raw_data then
    options:setChunked(10,500, 500)
    f:write('/data_unshift',self.a:float(),options)
  end

  f:write('/statistics/dose',torch.FloatTensor({self.electrons_per_angstrom2}))
  f:write('/statistics/MdivN',torch.FloatTensor({self.total_measurements/self.pixels_with_sufficient_exposure}))
  f:write('/statistics/counts_per_pixel',torch.FloatTensor({self.counts_per_valid_pixel}))
  f:write('/statistics/K',torch.FloatTensor({self.K}))

  self:save_error_history(f)
  if self.time then
    f:write('/results/time_elapsed',torch.FloatTensor({self.time}))
  end
  f:close()
end

function engine:generate_data(filename,poisson_noise,save_data)
  u.printf('Generating diffraction data:')
  local a = self.a_buffer1
  a:zero()

  self:initialize_probe()
  self:initialize_object_solution()
  self:initialize_object()
  self:update_views()
  self:calculateO_denom()

  if poisson_noise then
    local P_norm = self.P:normall(2)^2
    u.printf('poisson noise                 : %g',poisson_noise)
    local O_mean_transparency = self.O:abs():mean() - 0.2
    local factor = math.sqrt(poisson_noise / P_norm / O_mean_transparency)
    u.printf('mean object transparency      : %g',O_mean_transparency)
    u.printf('multiply probe with           : %g',factor)
    self.P:mul(factor)
    -- local P_norm = self.P:normall(2)^2
    -- u.printf('P_norm             : %g',P_norm)
  end

  self:update_views()
  self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)

  plt:plot(self.P[1][1]:re():float(),'P')

  for _, params1 in ipairs(self.batch_params) do
    local batch_start1, batch_end1, batch_size1 = table.unpack(params1)
    self:maybe_copy_new_batch_z(batch_start1)
    self.z:view_3D():fftBatched():norm()
    -- sum over probe and object modes
    self.z:sum(1):sum(2)
    local first = false
    for k = 1, batch_size1 do
      local k_full = k + batch_start1 - 1
      if first then
        plt:plot(self.z[k][1][1]:abs():log():float(),'self.z[k][o][1] fft')
      end

      a[k_full]:copy(self.z[k][1][1]:re())

      if self.bg_solution then
        a[k_full]:add(self.bg_solution)
        if first then
          plt:plot(self.bg_solution:float(),'bg_solution')
        end
      end

      if poisson_noise then

        local I_total = a[k_full]:sum()
        -- u.printf('I_total[%d] = %g',k_full,I_total)
        local a_h = a[k_full]:float()
        a_h[a_h:lt(0)] = 0
        local a_h_noisy = u.stats.poisson(a_h)

        -- u.printf('%g',a_h_noisy:sum())
        a[k_full]:copy(a_h_noisy)
      end

      a[k_full]:sqrt()

      if save_data then
        -- plt:plot(a[k_full]:float():log(),'a','dp'..k_full,false)
        first = false
      end
    end
  end

  self.a:copy(a)

  self:calculate_statistics()
  self:print_report()

  plt:plot(self.P[1][1]:re():float(),'P used for generate_data')
  plt:plot(self.O[1][1]:re():float(),'O used for generate_data')
  if save_data then
    self:save_data(filename)
  end
end

function engine:calculate_pixels_with_sufficient_measurements()
  -- O_mask is populated in calculateO_denom currently
  return self.O_mask:sum()
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
        z[k]:view_3D():fftBatched()
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
    -- local title = self.i..'_sum_a'
    -- plt:plot(sum_a:float():log(),title,self.save_path..title,true)
    -- plt:plot(sum_a:float():log(),'sum_a 2')
    d:sqrt()
    -- sum_a = (sum_k |F z_k|^2) / eta - (sum_k d_k)
    local dsum = d:sum(1)
    sum_a:add(-1,dsum)
    -- local title = self.i..'_sum_a_minus_dsum'
    -- plt:plot(sum_a:float():log(),title,self.save_path..title,true)
    -- sum_a = 1/K [ (sum_k d_k) - (sum_k |F z_k|^2) / eta ]
    sum_a:mul(-1/self.K)
    -- plt:plot(sum_a:float(),'sum_a 3')
    self.bg:add(sum_a)
    local title = self.i..'_bg_unclamped'
    plt:plot(self.bg:float():log(),title,self.save_path..title,self.show_plots)
    -- self.bg:clamp(0,1e10)
    -- local title = self.i..'_bg_clamped'
    -- plt:plot(self.bg:float():log(),title,self.save_path..title,true)

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

  z:view_3D():ifftBatched()
  self.module_error = self.module_error / self.norm_a
  return self.module_error, self.mod_updates
end

function engine:P_F()
  local err, updates = 0,0
  if self.i >= self.background_correction_start then
    err, updates = self:P_F_with_background()
  else
    err, updates = self:P_F_without_background()
  end
  self.mod_updates = self.mod_updates + updates
  self.module_error = self.module_error + err
  return err, updates
end

function engine:P_F_without_background()
  local err, updates = self.ops.P_F_without_background(self.P_Fz, self.a, self.a_exp, self.fm, self.fm_exp, self.zk_tmp1_real_PQstore, self.a_tmp_real_PQstore, self.a_tmp3_real_PQstore, self.old_batch_params['P_Fz'], self.i, self.power_threshold, self.fm_support, self.M, self.No, self.Np)
  return err/ self.norm_a, updates
end

function engine:maybe_refine_positions()
  if self.update_positions then
    self:refine_positions()
  end
end

function engine:refine_positions()
end

function engine:overlap_error(z_in,z_out)
  -- print('overlap_error')
  local result = self.P_Fz
  local res, res_denom = 0, 0

  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch_P_Q(k)
    self:maybe_copy_new_batch_z(k)
    local c = z_in:dot(z_out)
    local phase_diff = c/zt.abs(c)
    result:mul(z_out,phase_diff)
    res = res + result:add(-1,z_in):mul(-1):normall(2)
    res_denom = res_denom + z_out:normall(2)
  end
  return res/res_denom
end

-- we assume that self.z has actually the current z if this is called
function engine:relative_error(z1)
  local z = z1 or self.z
  local z_solution = self.z1

  local solution_masked = self.O_buffer:cmul(self.object_solution,self.O_mask)
  -- plt:plotReIm(solution_masked[1][1]:zfloat(),'solution_masked')
  local solution_masked_views = self:create_views(solution_masked)
  self:update_frames(z_solution,self.probe_solution,solution_masked_views,self.maybe_copy_new_batch_z1)

  local z_copy = self.z2:copy(z)
  for k, mview in ipairs(self.O_mask_views) do
    z_copy[k]:cmul(mview)
  end

  if self.object_solution then
    local c = z_solution:dot(z_copy)
    -- print(c)
    local phi = -zt.arg(c).re
    local exp_minus_phi = ztorch.re(math.cos(-phi)) + ztorch.im(math.sin(-phi))
    self.z2:mul(z_copy,exp_minus_phi)
    -- plt:plotcompare({z_solution[45][1][1]:zfloat():abs(),self.z2[45][1][1]:zfloat():abs()},{'sol abs','rec abs'})
    -- plt:plotcompare({z_solution[45][1][1]:zfloat():arg(),self.z2[45][1][1]:zfloat():arg()},{'sol arg','rec arg'})
    self.z2:add(-1,z_solution)
    local result = self.z2:normall(2)/z_solution:normall(2)
    -- print(result)
    return result
  end
end

function engine:image_error()
  local O_res = self.O_buffer
  if self.object_solution then
    local c = O_res:copy(self.O):cmul(self.O_mask):dot(self.object_solution)
    local phi = zt.arg(c).re
    local exp_minus_phi = ztorch.re(math.cos(-phi)) + ztorch.im(math.sin(-phi))
    -- u.printf('phase difference: %g rad',zt.arg(exp_minus_phi).re)
    O_res:mul(self.O,exp_minus_phi)
    -- if self.i==150 then
    --   plt:plotcompare({self.object_solution[1][1]:zfloat():abs(),O_res:clone():cmul(self.O_mask)[1][1]:zfloat():abs()},{'sol abs','rec abs'})
    --   plt:plotcompare({self.object_solution[1][1]:zfloat():arg(),O_res:clone():cmul(self.O_mask)[1][1]:zfloat():arg()},{'sol arg','rec arg'})
    -- end
    O_res:add(-1,self.object_solution):cmul(self.O_mask)
    local O_res_norm = O_res:normall(2)
    local norm1 = O_res_norm/O_res:copy(self.object_solution):cmul(self.O_mask):normall(2)
    return norm1
  end
end
-- free buffer: P_Fz
function engine:probe_error()
  local norm = self.P_tmp1_real_PFstore
  local P_corr = self.P_tmp2_PFstore
  if self.probe_solution then
    local c = self.P[1]:dot(self.probe_solution)
    local phi = -zt.arg(c).re
    local exp_minus_phi = ztorch.re(math.cos(-phi)) + ztorch.im(math.sin(-phi))
    P_corr:mul(self.P,exp_minus_phi)
    P_corr:add(-1,self.probe_solution)
    local norm1 = P_corr:normall(2)/self.probe_solution:normall(2)
    return norm1
  end
end

engine:mustHave("iterate")
engine:mustHave("allocate_error_history")
engine:mustHave("save_error_history")
engine:mustHave("get_errors")

return engine
