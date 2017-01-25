local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local engine, super = classic.class(...,base_engine)

function engine:_init(par)
  super._init(self,par)
end

function engine:DM_update()
  if self.i >= self.background_correction_start then
    -- print('DM_update_with_background')
    return self:DM_update_with_background()
  else
    return self:DM_update_without_background()
  end
end

function engine:DM_update_without_background()
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
    -- self.P_Fz = P_F((2P_Q - I))z
    self.plots = 0
    mod_error, mod_updates = self:P_F()
    --  z_{i+1} = z_i - P_Q z_i + P_F((2P_Q - I))z_i
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
  return {self.mod_errors:narrow(1,1,self.i), self.overlap_errors:narrow(1,1,self.i),self.img_errors:narrow(1,1,self.i)}
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
  for i=1,steps do
    self:update_iteration_dependent_parameters(i)
    self:P_Q()
    if self.has_solution then
      self.img_errors[{i}] = self:image_error()
      probe_error = self:probe_error()
    end
    -- u.printf('rel error P_Qz: %g',self:relative_error(self.P_Qz))
    self:maybe_refine_positions()
    self.overlap_errors[i] = self:overlap_error(self.z,self.P_Qz)
    self.mod_errors[i], mod_updates = self:DM_update()

    if self.has_solution then
      self.rel_errors[{i}] = self:relative_error()
    end

    u.printf('%-10d%-15g%-15g%-15g%-15g%-15g%-15g',i,self.mod_errors[i] or -1,self.overlap_errors[i] or -1 , relative_error or -1,self.img_errors[i] or -1, probe_error or -1, mod_updates/self.K*100.0)

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
