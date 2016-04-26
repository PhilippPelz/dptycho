local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine_shifted"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local engine, super = classic.class(...,base_engine)

function engine:_init(par)
  super._init(self,par)
end

function engine:DM_update()
  local mod_error, mod_updates
  -- local overlap_error = self:overlap_error(self.z,self.P_Qz)
  for k = 1, self.K, self.batch_size do
    self:maybe_copy_new_batch(self.z,self.z_h,k)
  -- self.P_Fz = (2P_Q - I)z
    self.P_Fz:mul(self.P_Qz,1+self.beta):add(-self.beta,self.z)
    -- z_{i+1} = z_i - P_Q z_i , P_Qz can be used as buffer now
    self.z:add(-1,self.P_Qz)
    -- self.P_Fz = P_F((2P_Q - I))z
    mod_error, mod_updates = self:P_F()
    --  z_{i+1} = z_i - P_Q z_i + P_F((2P_Q - I))z_i
    self.z:add(self.P_Fz)
  end
  return mod_error, mod_updates
end

function engine:iterate(steps)
  self.iterations = steps
  local mod_error, overlap_error, image_error, probe_error, mod_updates = -1,-1,-1, -1, 0
  local probe_change_0, last_probe_change, probe_change = nil, 1e10, 0
  for i=1,steps do
    self:update_iteration_dependent_parameters(i)
    self:P_Q()
    self:maybe_refine_positions()
    overlap_error = self:overlap_error(self.z,self.P_Qz)
    mod_error, mod_updates = self:DM_update()
    self:maybe_plot()

    image_error = self:image_error()
    probe_error = self:probe_error()

    u.printf('iteration %-3d: e_mod = %-02.02g    e_overlap = %-02.02g    e_image = %-02.02g  e_probe = %-02.02g  %d/%d modulus updates',i,mod_error  or -1,overlap_error or -1 ,image_error or -1, probe_error or -1, mod_updates, self.K)
    -- print('--------------------------------------------------------------------------------')
  end
  plt:plot(self.O[1]:zfloat(),'object - it '..steps)
  plt:plot(self.P[1]:zfloat(),'new probe')
end

return engine
