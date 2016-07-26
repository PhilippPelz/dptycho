local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine_shifted"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local engine, super = classic.class(...,base_engine)

function engine:_init(par)
  u.printf('========================== DM  engine ==========================')
  super._init(self,par)
  self:update_views()
  self:calculateO_denom()
  self:update_frames(self.z,self.P,self.O_views,self.maybe_copy_new_batch_z)
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
  return self.module_error, self.mod_updates
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

function engine:iterate(steps)
  self.iterations = steps
  self:initialize_plotting()
  local mod_error, overlap_error, image_error, probe_error, mod_updates = -1,-1,nil, nil, 0
  local probe_change_0, last_probe_change, probe_change = nil, 1e10, 0
  u.printf('%-10s%-15s%-15s%-15s%-15s%-15s','iteration','e_mod','e_overlap','e_image','e_probe','modulus updates %')
  print('----------------------------------------------------------------------------------------------')
  for i=1,steps do
    self:update_iteration_dependent_parameters(i)
    self:P_Q()
    self:maybe_refine_positions()
    overlap_error = self:overlap_error(self.z,self.P_Qz)
    mod_error, mod_updates = self:DM_update()

    image_error = self:image_error()
    probe_error = self:probe_error()

    u.printf('%-10d%-15g%-15g%-15g%-15g%-15g',i,mod_error or -1,overlap_error or -1 ,image_error or -1, probe_error or -1, mod_updates/self.K*100.0)

    self:maybe_plot()
    self:maybe_save_data()

    collectgarbage()
  end
  self:save_data(self.save_path .. 'ptycho_' .. (steps+1))
  -- plt:shutdown_reconstruction_plot()
  -- plt:plot(self.O[1]:zfloat(),'object - it '..steps)
  -- plt:plot(self.P[1]:zfloat(),'new probe')
end

return engine
