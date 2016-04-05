local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local engine, super = classic.class(...,base_engine)

function engine:_init(pos,a,nmodes_probe,nmodes_object,solution,probe)
  super._init(self,pos,a,nmodes_probe,nmodes_object,solution,probe)
end

function engine:DM_update()
  self:P_Q(self.z,self.P_Qz)
  local overlap_error = self:overlap_error(self.z,self.P_Qz)
  -- self.P_Fz = (2P_Q - I)z
  self.P_Fz:mul(self.P_Qz,2):add(-1,self.z)
  -- z_{i+1} = z_i - P_Q z_i , P_Qz can be used as buffer now
  self.z:add(-1,self.P_Qz)
  -- self.P_Fz = P_F((2P_Q - I))z
  local mod_error = self:P_F(self.P_Fz,self.P_Fz)
  --  z_{i+1} = z_i - P_Q z_i + P_F((2P_Q - I))z_i
  self.z:add(self.P_Fz)
  return mod_error,overlap_error
end

function engine:iterate(steps)
  local mod_error, overlap_error, image_error = -1,-1,-1
  printMinMax(self.solution,'solution      ')
  for i=1,steps do

    mod_error, overlap_error = self:DM_update()

    if self.update_probe and i >= self.probe_update_start then
      local probe_change_0 = nil
      for j = 1,10 do
        local probe_change = self:refine_probe()
        if not probe_change_0 then probe_change_0 = probe_change end
        self:update_O(self.z)
        if probe_change < .1 * probe_change_0 then break end
        u.printf('probe change : %g',probe_change)
      end

      self:update_z_from_O(self.z)
    end

    if i % 20 == 0 then
      plt:plot(self.O[1]:zfloat(),'object - it '..steps)
      plt:plot(self.P[1]:zfloat(),'new probe')
    end

    image_error = self:image_error()

    u.printf('iteration %03d: e_mod = %g    e_overlap = %g    e_image = %g',i,mod_error,overlap_error,image_error)
    print('--------------------------------------------------------------------------------')
  end
  plt:plot(self.O[1]:zfloat(),'object - it '..steps)
  plt:plot(self.P[1]:zfloat(),'new probe')
end

return engine
