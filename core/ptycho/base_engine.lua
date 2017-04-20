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
-- allocate_buffers
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

  if self.probe_do_enforce_modulus then
    self.a_exp_probe = self.a:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
    self.fm_exp_probe = self.fm:view(self.K,1,1,self.M,self.M):expand(self.K,self.No,self.Np,self.M,self.M)
  end

  -- dimensions used for probe and object modes in z[k]
  self.O_dim = 1
  self.P_dim = 2

  local object_size = self.pos:max(1):add(torch.IntTensor({self.a:size(2) + 2*self.margin,self.a:size(3)+ 2*self.margin})):squeeze()

  -- pprint(object_size)
  self.pos:add(self.margin)
  self.Nx = object_size[1] - 1
  self.Ny = object_size[2] - 1

  if self.Nx % 2 ~= 0 then
    self.Nx = self.Nx + 1
  end

  if self.Ny % 2 ~= 0 then
    self.Ny = self.Ny + 1
  end

  if self.object_initial then
    pprint(self.object_initial)
    pprint({{min[1][1],self.Nx},{min[1][2],self.Ny}})
    self.object_initial = self.object_initial[{{1,self.Nx},{1,self.Ny}}]
  end

  self:allocate_buffers(self.K,self.No,self.Np,self.M,self.Nx,self.Ny)

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
    self:initialize_object_solution()
    if #(self.O_denom_views) < 2 then
      self:update_views()
    end
    if not self.P_initialized then
      self:initialize_probe()
      -- plt:plot(self.P[1][1]:zfloat(),'P')
      -- print('probe initialized')
      self:calculateO_denom()
    end
  end

  self:calculate_total_intensity()

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

function engine:calculate_total_intensity()
  self.a_buffer1:cmul(self.a,self.fm):pow(2)
  if self.calculate_dose_from_probe then
    self.I_total = self.K * self.P_buffer:copy(self.P):norm():re():sum()
    self.I_diff_total = self.a_buffer1:sum()
    self.elastic_percentage = self.I_diff_total / self.I_total
  else
    self.I_total = self.a_buffer1:sum()
  end
end

function engine:print_report()
  print(   '------------------------------------------------------------------------------')
  u.printf('K (number of diff. patterns)           : %d',self.K)
  u.printf('N (object dimensions)                  : %d x %d  (%2.1f x %2.1f Angstrom)',self.Nx,self.Ny,self.Nx*self.dx*1e10,self.Ny*self.dx*1e10)
  u.printf('M (probe size)                         : %d',self.M)
  u.printf('resolution (Angstrom)                  : %2.3g',self.dx*1e10)
  print(   '------------------------------------------------------------------------------')
  u.printf('power threshold is                     : %2.3g',self.power_threshold)
  u.printf('total measurements                     : %2.3g',self.total_measurements)
  u.printf('total measurements (nonzero)           : %2.3g',self.total_nonzero_measurements)
  u.printf('# unknowns object                      : %2.3g',self.O:nElement())
  u.printf('# unknowns probe                       : %2.3g',self.P:nElement())
  u.printf('# unknowns total                       : %2.3g',self.O:nElement() + self.P:nElement())
  u.printf('')
  u.printf('total measurements/image_pixels        : %2.3g',
  self.total_measurements/self.pixels_with_sufficient_exposure)
  u.printf('nonzero measurements/image_pixels      : %2.3g',
  self.total_nonzero_measurements/self.pixels_with_sufficient_exposure)
  u.printf('nonzero measurements/# unknowns        : %2.3g',
  self.total_nonzero_measurements/(self.pixels_with_sufficient_exposure + self.P:nElement()))
  u.printf('')
  u.printf('max counts in pattern                  : %2.3g',self.I_max)
  u.printf('rescale_regul_amplitude                : %2.3g',self.rescale_regul_amplitude)
  u.printf('probe intensity                        : %2.3g',self.P_norm)
  u.printf('total counts in this scan              : %2.3g',self.I_total)
  u.printf('total illuminated area (pixels)        : %2.3g',self.pixels_with_sufficient_exposure)
  u.printf('total illuminated area (Angstrom^2)    : %2.3g',self.pixels_with_sufficient_exposure/ (self.dx^2 / 1e-20))
  u.printf('mean counts                            : %2.3g',self.I_mean)
  u.printf('counts per       pixel                 : %3.3g',self.counts_per_pixel)
  u.printf('counts per valid pixel                 : %3.3g',self.counts_per_valid_pixel)
  if self.calculate_dose_from_probe then
    u.printf('percentage of elastically scattered e- : %2.3g %%',self.elastic_percentage * 100)
  end
  u.printf('')
  u.printf('e- / Angstrom^2                        : %2.3g',self.electrons_per_angstrom2)
  print(   '------------------------------------------------------------------------------')
end

-- P_Fz free
function engine:update_frames(z,mul_split,merge_memory_views,batch_copy_func)
  self.ops.Q(z,mul_split,merge_memory_views,self.zk_buffer_update_frames,self.shift_buffer,self.k_to_batch_index, fn.partial(batch_copy_func,self),self.batches,self.K,self.dpos)
end

function engine:calculateO_denom()
  self.ops.calculateO_denom(self.O_denom,self.O_mask,self.O_denom_views,self.P,self.P_buffer,self.Pk_buffer_real,self.object_inertia,self.K,self.O_denom_regul_factor_start,self.O_denom_regul_factor_end,self.i,self.iterations,self.dpos)
  self.O_mask_initialized = true
  -- plt:plot(self.O_denom[1][1]:float() ,'O_denom in calculateO_denom')
end

function engine:P_Q()
  if self.update_probe then
    for i = 1,self.P_Q_iterations do
      self:merge_frames(self.z,self.P,self.O,self.O_views,true)
      -- plt:plot(self.P[1][1]:zfloat(),'P before refine_probe')
      -- local hasnans = torch.any(nans)
      -- local old_P_norm = self.P:normall(2)^2
      -- u.printf('old P norm = %g',old_P_norm)

      -- local nans = torch.ne(self.P:re(),self.P:re())

      -- u.printf('new P norm = %g',new_P_norm)
      -- local new_P_norm2 = self.P:normall(2)^2
      -- u.printf('new P norm rescaled = %g',new_P_norm2)

      -- print(hasnans)
      -- local nans = torch.ne(self.P:im(),self.P:im())
      -- local hasnans = torch.any(nans)
      -- print(nans:sum())
      -- print(hasnans)
      -- plt:plot(self.P[1][1]:zfloat(),'P after refine_probe')
      -- or probe_change > 0.97 * last_probe_change
      local probe_change = self:refine_probe()
      if not probe_change_0 then probe_change_0 = probe_change end
      if probe_change < .1 * probe_change_0  then break end
      -- last_probe_change = probe_change
      u.printf('            probe change : %g',probe_change)
    end
    self:update_frames(self.P_Qz,self.P,self.O_views,self.maybe_copy_new_batch_P_Q)
    plt:plot(self.P[1][1]:zfloat(),'P after refine',path .. '/probe/'.. string.format('%d_P',i),false)
    plt:plot(self.P[1][1]:clone():fft():fftshift():zfloat(),'P fft after refine',path .. '/probe/'.. string.format('%d_Pfft',i),false)
  else
    self:P_Q_plain()
  end
end

function engine:refine_probe()
  local probe_change = self.ops.refine_probe(self.z,self.P,self.O_views,self.P_buffer,self.P_buffer2,self.P_buffer_real,self.P_tmp2_real_PQstore,self.zk_tmp1_PQstore,self.zk_tmp2_PQstore, self.zk_tmp1_real_PQstore,self.k_to_batch_index, fn.partial(self.maybe_copy_new_batch_z,self),self.dpos,self.support,self.probe_inertia,self.probe_lowpass)

  if self.probe_support_fourier then
    self.P:view_3D():fft()
    -- print(self.P:view_3D():size(1))
    -- local k =  self.P:view_3D():size(1)
    -- pprint(self.P:view_3D())
    for i = 1, self.P:view_3D():size(1) do
      self.P:view_3D()[i]:fftshift()
    end
    self.P = self.support_fourier:forward(self.P)
    for i = 1, self.P:view_3D():size(1) do
      self.P:view_3D()[i]:fftshift()
    end
    self.P:view_3D():ifft()
  end

  local new_P_norm = self.P:normall(2)^2
  self.P:div(math.sqrt(new_P_norm))
  self.P:mul(math.sqrt(self.P_norm))
  self:calculateO_denom()
  return probe_change/self.Np
end

function engine:P_Q_plain()
  self:merge_frames(self.z,self.P,self.O,self.O_views,true)
  -- plt:plot(self.O[1][1]:zfloat(),'O after merge')
  -- plt:plot(self.P[1][1]:zfloat(),'P after merge')
  self:update_frames(self.P_Qz,self.P,self.O_views,self.maybe_copy_new_batch_P_Q)
end

function engine:merge_frames_internal(z,mul_merge, merge_memory, merge_memory_views, zk_buffer, P_buffer, do_normalize_merge_memory)
  -- plt:plotReIm(self.O[1][1]:zfloat(),'O before merge')
  self.ops.Q_star(z, mul_merge, merge_memory, merge_memory_views, zk_buffer, P_buffer,self.object_inertia, self.k_to_batch_index,fn.partial(self.maybe_copy_new_batch_z,self), self.batches, self.K,self.dpos,self.object_highpass)
  -- plt:plot(self.O_denom[1][1]:float():log(),'O_denom')
  -- plt:plotReIm(merge_memory[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'O after merge 0')
  -- pprint(merge_memory)
  -- pprint(self.O_denom)
  -- print(self.O_denom:max(),self.O_denom:min())
  if do_normalize_merge_memory then
    merge_memory:cmul(self.O_denom)
  end
  -- plt:plotReIm(merge_memory[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'O after merge 1')
  -- plt:plot(self.O[1][1]:clone():cmul(self.O_mask[1][1]):zfloat(),'O after merge 1')
end

engine:mustHave("merge_frames")

function engine:before_iterate()
  -- print('before_iterate')
  if not self.P_initialized then
    self:initialize_probe()
    self:calculateO_denom()
  end
  self:initialize_object_solution()
  self:initialize_object()
  self:update_views()
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
  self.plot_data[8] = self.O_mask:float()
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

  if self.probe_init == 'copy' then
    if self.probe_solution:dim() == 2 then
      self.P[1][1]:copy(self.probe_solution)
    else
      self.P:copy(self.probe_solution)
    end
  else
    self.P:zero()
    self.P[1][1]:add(300+0i)
  end

  if self.probe_solution:dim() == 2 then
    for i=2,self.Np do
      self.P[1][i]:zero()
      self.P[1][i]:copyRe(Pre)--:copyIm(Pre)
      self.P[1][i]:cmul(self.P[1][1])
      -- self.P[1][i]:add(Pre)
    end
    self.P[1][1]:mul(1-((self.Np-1)*0.05))
  end

  if self.probe_support then
    local probe_size = self.P:size():totable()
    self.support = znn.SupportMask(probe_size,probe_size[#probe_size]*self.probe_support)
    self.P = self.support:forward(self.P)
  else
    self.support = nil
  end
  if self.probe_support_fourier then
    local probe_size = self.P:size():totable()
    self.support_fourier = znn.SupportMask(probe_size,probe_size[#probe_size]*self.probe_support_fourier)
  else
    self.support_fourier = nil
  end
  self.P_initialized = true
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
    -- plt:plot(self.O[1][1]:zfloat())

    local O = ptycho.initialization.truncated_spectral_estimate_power_it(self.z,self.P,self.O_denom,self.object_init_truncation_threshold,self.ops,self.a,self.fm,self.z1,self.a_buffer2,self.zk_buffer_update_frames,self.P_buffer,self.O_buffer,self.batch_params,self.old_batch_params,self.k_to_batch_index,self.batches,self.batch_size,self.K,self.M,self.No,self.Np,self.pos,self.dpos,self.O_mask)
    local angle = O:arg():float()
    -- angle = u.gf(angle,2)
    self.O:polar(torch.CudaTensor(O:size()):fill(1),angle:cuda())
    -- z:view_3D():ifftBatched()
    self.O_init = self.O:zfloat()
    -- plt:plot(self.O[1][1]:zfloat(),'O initialization')
  elseif self.object_init == 'gcl' then
      u.printf('gcl initialization is not implement yet')
      self.O:zero():add(1+0i)
  elseif self.object_init == 'copy' then
    if self.object_initial:dim() == 2 then
      -- pprint(self.O)
      -- pprint(self.object_initial)
      self.O[1][1]:copy(self.object_initial)
    else
      self.O:copy(self.object_initial)
    end
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
  self.O = torch.ZCudaTensor.new(No,1,Nx,Ny)
end


function engine:update_iteration_dependent_parameters(it)
  self.i = it
  rawset(_G, 'iteration', it)
  self.update_probe = it >= self.probe_update_start
  self.update_object = it >= self.object_update_start
  self.update_positions = it >= self.position_refinement_start and it <= self.position_refinement_stop and it % self.position_refinement_every == 0
  self.do_plot = it % self.plot_every == 0 and it > self.plot_start
  self.calculate_new_background = self.i >= self.background_correction_start
  self.do_save_data = it % self.save_interval == 0
  if self.probe_lowpass_fwhm(it) then
    local conv = self.r2:clone():div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    self.probe_lowpass = torch.ZCudaTensor.new(self.r2:size()):zero()
    -- self.probe_lowpass:copyRe(self.r2:clone():sqrt():lt(self.probe_lowpass_fwhm(it)):cuda()):fft()
    self.probe_lowpass:copyRe(conv)
    -- self.probe_lowpass:cmul(conv)
    -- self.probe_lowpass:ifft():fftshift()
    -- self.probe_lowpass:mul(1/self.probe_lowpass:max())
    -- :div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    -- self.probe_lowpass = self.r2:clone():div(-2*(self.probe_lowpass_fwhm(it)/2.35482)^2):exp():cuda():fftshift()
    self.probe_lowpass = self.probe_lowpass:view(1,1,self.M,self.M):expand(1,self.Np,self.M,self.M)
    local title = self.i..' probe_lowpass'
    local savepath = self.save_path .. title
    -- plt:plot(self.probe_lowpass[1][1]:re():float(),title,savepath,self.show_plots)
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

function engine:enforce_probe_modulus()
  self.ops.P_F_without_background_basic(self.P,self.a_exp_probe,self.fm_exp_probe,self.P_buffer_real,  nil, self.M, self.No, self.Np)
end

function engine:generate_data(filename,poisson_noise,save_data)
  u.printf('Generating diffraction data:')
  local a = self.a_buffer1
  a:zero()

  self:initialize_probe()
  self:calculateO_denom()

  self:initialize_object_solution()
  self:initialize_object()
  self:update_views()

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

function engine:maybe_refine_positions(do_project)
  if self.update_positions then
    if self.position_refinement_method == 'marchesini' then
      self:refine_positions_marchesini()
    elseif self.position_refinement_method == 'annealing' then
      self:refine_positions_annealing()
    end
    self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)
    self:calculateO_denom()
    if do_project then
      self:P_Q_plain()
    end
  end
end

function engine:refine_positions_annealing()
  -- function engine:update_frames(z,mul_split,merge_memory_views,batch_copy_func)
  --   print('update_frames')
  --   self.ops.Q(z,mul_split,merge_memory_views,self.zk_buffer_update_frames,self.k_to_batch_index, fn.partial(batch_copy_func,self),self.batches,self.K,self.dpos)
  -- end
  local j = self.i - self.position_refinement_start
  local position_refinement_iterations = self.position_refinement_stop - self.position_refinement_start
  local current_max_dist = (1-(j/position_refinement_iterations)) * self.position_refinement_max_disp
  local dpos_trials = torch.FloatTensor(self.position_refinement_trials,2)
  local k_to_batch_index = torch.linspace(1,self.position_refinement_trials,self.position_refinement_trials)
  local dummy_function = function(k) end
  local z_trials = self.z_buffer_trials
  local a_trials = self.z_buffer_trials_real
  local path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/'
  for k = 1, self.K do
    -- fill views
    local views = {}
    for i = 1, self.position_refinement_trials do
      views[i] = self.O_views[k]
    end
    -- fill trials
    dpos_trials:copy(self.dpos[{{k},{}}]:expandAs(dpos_trials))
    local random_displacement = torch.rand(self.position_refinement_trials-1,2):float()
    random_displacement = (random_displacement - 0.5) * 2 * current_max_dist
    dpos_trials[{{2,self.position_refinement_trials},{}}]:add(random_displacement)

    -- create exit waves
    self.ops.Q(z_trials,self.P,views,self.zk_buffer_update_frames,self.shift_buffer,k_to_batch_index, dummy_function,1,self.position_refinement_trials,dpos_trials)
    -- calculate error
    z_trials:view_3D():fftBatched()
    a_trials:normZ(z_trials)
    local at = a_trials:sum(2)
    local at1 = at:sum(3)
    -- trials x 1 x 1 x M x M
    at1:sqrt()
    -- trials x M x M
    local a = at1:squeeze()
    -- pprint(a)
    -- for l = 1, self.position_refinement_trials do
    --   plt:plotcompare({a[l]:clone():fftshift():float(),self.a[k]:clone():fftshift():float()},{'a_trials '..l,'a '..k},'',path .. string.format('%d_it%d_a',k,i),true)
    -- end
    local a_k_exp = self.a[k]:view(1,self.M,self.M):expandAs(a)
    a:add(-1,a_k_exp)
    a:pow(2)
    local a0 = a:sum(2)
    local a1 = a0:sum(3)
    -- pprint(a1)
    -- select new dpos
    min, imin = torch.min(a1,1)
    -- pprint(imin,min)
    imin = imin[1][1][1]
    min = min[1][1][1]
    -- print('imin',imin,'min',min)
    -- print('dpos_trials',dpos_trials)
    -- print('a1',a1:squeeze())
    if imin ~= 1 then
      u.printf('  Found new position for index %d: (%g,%g) -> (%g,%g) errors: %g -> %g',k,self.dpos[k][1],self.dpos[k][2],dpos_trials[imin][1],dpos_trials[imin][2],a1[1][1][1],min)
    end
    self.dpos[k]:copy(dpos_trials[imin])
  end
end
-- mul_merge: 1 x Np x M x M
-- z_merge  ; No x Np x M x M
-- mul_split: 1 x Np x M x M
-- result   : No x Np x M x M
function engine:merge_and_split_pair(i,j,mul_merge, z_merge , mul_split, result)
  local O_tmp = self.O_tmp_PFstore
  local O_tmp_views = self.O_tmp_PF_views
  local mul_merge_shifted_conj = self.P_tmp3_PFstore
  local mul_split_shifted = self.P_tmp3_PFstore
  local ramp_buffer = result[1][1]

  mul_merge_shifted_conj[1]:shift(mul_merge[1],self.dpos[j],ramp_buffer):conj()
  O_tmp:zero()
  O_tmp_views[j]:add(result:cmul(z_merge,mul_merge_shifted_conj:expandAs(z_merge)):sum(self.P_dim))
  -- plt:plotReIm(self.O_tmp[1]:zfloat(),'merge_and_split_pair self.O_tmp')
  O_tmp:cmul(self.O_denom)
  -- plt:plotReIm(self.O_tmp[1]:zfloat(),'merge_and_split_pair self.O_tmp 2')
  -- plt:plotReIm(ov[1]:zfloat(),'merge_and_split_pair ov')
  mul_split_shifted[1]:shift(mul_split[1],self.dpos[i],ramp_buffer)
  result:cmul(O_tmp_views[i]:expandAs(result),mul_split_shifted:expandAs(result))
  return result
end

function engine:split_single(i,mul_split,result)
    local mul_split_shifted = self.P_tmp3_PFstore:zero()
    local ramp_buffer = self.zk_tmp5_PFstore[1][1]
    -- plt:plotReIm(ov[1]:zfloat(),'ov')
    mul_split_shifted[1]:shift(mul_split[1],self.dpos[i],ramp_buffer)
    -- plt:plotReIm(mul_split_shifted[1][1]:zfloat(),'mul_split_shifted')
    result:cmul(self.O_views[i]:expandAs(result),mul_split_shifted:expandAs(result))
    return result
end

function engine:do_frames_overlap(i,j)
  local pos_i = self.pos[i]
  local pos_j = self.pos[j]
  local r = self.beam_radius
  local d = math.sqrt((pos_i[1]-pos_j[1])^2+(pos_i[2]-pos_j[2])^2)
  local overlap_area = 2 * r^2 * math.acos(d/2/r) - (d/2) * math.sqrt(4*r^2-d^2) -- maximum pi r ^ 2

  local i_and_j_overlap = overlap_area > 0.1 * r^2
  -- local beam_fraction = 0.6
  -- local beam_offset = (1-beam_fraction)*self.M/2
  -- local beam_size = beam_fraction*self.M
  -- local x_lt, x_gt = math.min(pos_i[1],pos_j[1]),math.max(pos_i[1],pos_j[1])
  -- local y_lt, y_gt = math.min(pos_i[2],pos_j[2]),math.max(pos_i[2],pos_j[2])
  -- x_lt = x_lt + beam_offset
  -- x_gt = x_gt + beam_offset
  -- y_lt = y_lt + beam_offset
  -- y_gt = y_gt + beam_offset
  -- local x_gt_within_x_lt_beam = x_gt < x_lt + beam_size
  -- local y_gt_within_y_lt_beam = y_gt < y_lt + beam_size
  -- local i_and_j_overlap = x_gt_within_x_lt_beam and y_gt_within_y_lt_beam
  return i_and_j_overlap
end

-- buffers:
--  1 x sizeof(O) el C
--  6 x sizeof(z[k]) el C
-- free buffer: P_F
function engine:refine_positions_marchesini()
  local H = torch.FloatTensor(2*self.K,2*self.K):zero()
  local overlaps = torch.FloatTensor(self.K,self.K)
  local H1 = H[{{1,self.K},{1,self.K}}]
  local H2 = H[{{self.K+1,2*self.K},{self.K+1,2*self.K}}]
  local Hx1 = H[{{1,self.K},{self.K+1,2*self.K}}]
  local Hx2 = H[{{self.K+1,2*self.K},{1,self.K}}]
  local b = torch.FloatTensor(2*self.K,1)
  local bv = b:view(b:nElement())

  self.P_Fz:zero()

  local zRy = self.zk_tmp1_PFstore
  local zRx = self.zk_tmp2_PFstore
  local Rx = self.P_tmp1_PFstore
  local Ry = self.P_tmp2_PFstore

  local r1 = self.zk_tmp5_PFstore
  local r2 = self.zk_tmp6_PFstore
  local r3 = self.zk_tmp7_PFstore
  local z_under = self.zk_tmp8_PFstore
  local z_i = self.zk_tmp8_PFstore

  local parts = (#self.batch_params) ^ 2
  local l = 0
  for _, params1 in ipairs(self.batch_params) do
    local batch_start1, batch_end1, batch_size1 = table.unpack(params1)
    -- print('params1')
    -- pprint(params1)
    for _, params2 in ipairs(self.batch_params) do
      -- print('params2')
      -- pprint(params2)
      local batch_start2, batch_end2, batch_size2 = table.unpack(params2)


      for i = 1, batch_size1 do
        local i_full = i + batch_start1 - 1
        self:maybe_copy_new_batch_z(batch_start1)
        self:maybe_copy_new_batch_P_Q(batch_start1)
        Rx[1]:dx(self.P[1],zRx[1],zRy[1])
        Ry[1]:dy(self.P[1],zRx[1],zRy[1])

        zRx = self:split_single(i_full,Rx,zRx)
        zRy = self:split_single(i_full,Ry,zRy)
        z_under:add(self.z[i],-1,self.P_Qz[i])

        bv[i_full] = z_under:dot(zRx).re
        bv[self.K + i_full] = z_under:dot(zRy).re
        z_i:copy(self.z[i])

        for j=1, batch_size2 do
          l = l + 1
          xlua.progress(l,self.K*self.K)
          local j_full = j + batch_start2 - 1
          self:maybe_copy_new_batch_z(batch_start2)
          local H1_ij = 0
          local H2_ij = 0
          local Hx_ij = 0
          if i_full == j_full then
            H1_ij = zRx:dot(zRx).re
            H2_ij = zRy:dot(zRy).re
            Hx_ij = zRx:dot(zRy).re
          end

          -- overlaps[{i,j}] = self:do_frames_overlap(i,j) and 1 or 0
          if self:do_frames_overlap(i,j) then
            local O11_ij = self:merge_and_split_pair(i_full,j_full,Rx,self.z[j],Rx,r1)
            local O22_ij = self:merge_and_split_pair(i_full,j_full,Ry,self.z[j],Ry,r2)
            local Ox_ij = self:merge_and_split_pair(i_full,j_full,Rx,self.z[j],Ry,r3)

            H1_ij = H1_ij - z_i:dot(O11_ij).re
            H2_ij = H2_ij - z_i:dot(O22_ij).re
            Hx_ij = Hx_ij - z_i:dot(Ox_ij).re
          end

          H1[{i_full,j_full}] = H1_ij
          H2[{i_full,j_full}] = H2_ij
          Hx1[{i_full,j_full}] = Hx_ij
          Hx2[{i_full,j_full}] = Hx_ij
        end -- end j
      end -- end i
    end -- end params2
  end -- end params1

  local ksi, LU = torch.gesv(b,H)
  -- plt:plot(H:log(),'H')
  local max,imin = torch.max(ksi,1)
  for i=1,self.K do
    local p = torch.FloatTensor{-ksi[i][1],-ksi[i+self.K][1]}
    -- u.printf('%04d : %g,%g',i,-ksi[i][1],-ksi[i+self.K][1])
    self.dpos[i]:add(p:clamp(-self.position_refinement_max_disp,self.position_refinement_max_disp))
  end

  -- local di = self.dpos:int()
  -- print('di')
  -- print(di)
  -- print('self.pos')
  -- print(self.pos)
  -- print('self.dpos')
  -- print(self.dpos)
  -- self.pos:add(di)
  -- self.dpos:add(-1,di:float())
  -- print('self.pos')
  -- print(self.pos)
  -- print('self.dpos')
  -- print(self.dpos)
  -- local overlaps = torch.gt(H,0):sum() / 2

  -- plt:scatter_positions(self.dpos:clone():add(self.pos:float()),self.dpos_solution:clone():add(self.pos:float()))
  if self.dpos_solution then
    local dp = self.dpos_solution:clone():add(-1,self.dpos):abs()
    local max_err = dp:max()
    local pos_err = self.dpos_solution:clone():add(-1,self.dpos):abs():sum()/self.K
    u.printf('ksi[%d] = %g, pos_error = %g, max_pos_error = %g', imin[1][1] , max[1][1] , pos_err, max_err)
  else
    u.printf('ksi[%d] = %g', imin[1][1] , max[1][1])
  end
  -- self:update_views()
  self:calculateO_denom()
  self:P_Q_plain()
end

function engine:filter_object(O)
  -- pprint(self.O_tmp_PQstore)
  local O_fluence = O:normall(2)
  print('filtering object')

  O:view_3D():fftBatched()

  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft unfiltered ',self.save_path .. self.i..' P filtered ',false)
  O:cmul(self.object_highpass)
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft filtered ',self.save_path .. self.i..' P filtered ',false)
  O:view_3D():ifftBatched()

  local O_fluence_new = O:normall(2)
  u.printf('O_fluence/O_fluence_new = %g',O_fluence/O_fluence_new)
  O:mul(O_fluence/O_fluence_new)
end

function engine:filter_probe()
  local P_fluence = self.P_tmp1_real_PQstore:normZ(self.P):sum()
  print('filtering probe')
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P',self.save_path .. self.i..' P',false)
  self.P:view_3D():fftBatched()
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft unfiltered ',self.save_path .. self.i..' P filtered ',false)
  self.P:cmul(self.probe_lowpass)
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft filtered ',self.save_path .. self.i..' P filtered ',false)
  self.P:view_3D():ifftBatched()
  local P_fluence_new = self.P_tmp1_real_PQstore:normZ(self.P):sum()
  u.printf('P_fluence/P_fluence_new = %g',P_fluence/P_fluence_new)
  self.P:mul(P_fluence/P_fluence_new)
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P filtered ',self.save_path .. self.i..' P filtered ',false)
  -- self:calculateO_denom()
  -- self:merge_frames(self.P,self.O,self.O_views)
end

function engine:overlap_error(z_in,z_out)
  -- print('overlap_error')
  local result = self.P_Fz
  local res, res_denom = 0, 0

  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch_P_Q(k)
    self:maybe_copy_new_batch_z(k)
    local c = z_in:dot(z_out)
    local phi = zt.arg(c).re
    local exp_minus_phi = ztorch.re(math.cos(phi)) + ztorch.im(math.sin(phi))
    result:mul(z_out,exp_minus_phi)
    res = res + result:add(-1,z_in):mul(-1):normall(2)
    res_denom = res_denom + z_in:normall(2)
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
    local phi = zt.arg(c).re
    local exp_minus_phi = ztorch.re(math.cos(phi)) + ztorch.im(math.sin(phi))
    self.z2:mul(z_copy,exp_minus_phi)
    -- plt:plotcompare({z_solution[45][1][1]:zfloat():abs(),self.z2[45][1][1]:zfloat():abs()},{'sol abs','rec abs'})
    -- plt:plotcompare({z_solution[45][1][1]:zfloat():arg(),self.z2[45][1][1]:zfloat():arg()},{'sol arg','rec arg'})
    self.z2:add(-1,z_solution)
    -- for i = 1,self.K do
    --   plt:plotReIm(self.z2[i][1][1]:zfloat(),'z diff')
    -- end
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
    -- plt:plotReIm(O_res[1][1]:zfloat(),'image difference')
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
    local phi = zt.arg(c).re
    local exp_minus_phi = ztorch.re(math.cos(phi)) + ztorch.im(math.sin(phi))
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
engine:mustHave("allocate_buffers")

return engine
