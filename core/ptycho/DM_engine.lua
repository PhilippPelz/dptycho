local classic = require 'classic'
<<<<<<< HEAD
local base_engine = require "dptycho.core.ptycho.base_engine_shifted"
=======
local base_engine = require "dptycho.core.ptycho.base_engine"
>>>>>>> 5400af4241e2b79a156cfca7ef79ec71554c2453
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local engine, super = classic.class(...,base_engine)

<<<<<<< HEAD
function engine:_init(par)
  super._init(self,par)
end

function engine:DM_update()
  local mod_error, mod_updates
  -- self:P_Q(self.z,self.P_Qz)
  -- local overlap_error = self:overlap_error(self.z,self.P_Qz)
  -- self.P_Fz = (2P_Q - I)z
  self.P_Fz:mul(self.P_Qz,1+self.beta):add(-self.beta,self.z)
  -- z_{i+1} = z_i - P_Q z_i , P_Qz can be used as buffer now
  self.z:add(-1,self.P_Qz)
  -- self.P_Fz = P_F((2P_Q - I))z
  mod_error, mod_updates = self:P_F(self.P_Fz,self.P_Fz)
  --  z_{i+1} = z_i - P_Q z_i + P_F((2P_Q - I))z_i
  self.z:add(self.P_Fz)
  return mod_error, mod_updates
end

function engine:iterate(steps)
  self.iterations = steps
  local mod_error, overlap_error, image_error, probe_error, mod_updates = -1,-1,-1, -1, 0
  local probe_change_0, last_probe_change, probe_change = nil, 1e10, 0
  -- printMinMax(self.solution,'solution      ') :cmul(self.O_mask)
  -- plt:plotReIm(self.O_tmp:copy(self.O)[1]:zfloat(),'object - it '..0)
  -- plt:plotReIm(self.P[1]:zfloat(),'probe - it '..0)
  self:P_Q(self.z,self.P_Qz)
  for i=1,steps do
    self:update_iteration_dependent_parameters(i)
    mod_error, mod_updates = self:DM_update()

    if self.update_probe then
      for _ = 1,self.P_Q_iterations do
        self.j = _
        probe_change = self:refine_probe()
        self:update_O(self.z)

        -- or probe_change > 0.97 * last_probe_change
        if not probe_change_0 then probe_change_0 = probe_change end
        if probe_change < .1 * probe_change_0  then break end
        -- last_probe_change = probe_change
        -- u.printf('            probe change : %g',probe_change)
      end

      self:update_z_from_O(self.P_Qz)
    else
      self:P_Q(self.z,self.P_Qz)
    end
    overlap_error = self:overlap_error(self.z,self.P_Qz)
    if self.update_positions then
      -- self:P_Q(self.z,self.P_Qz)
      self:refine_positions()
    end
    if self.do_plot then
      plt:plot(self.O_tmp_PFstore:copy(self.O):cmul(self.O_mask)[1]:zfloat(),'object - it '..i)
=======
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
>>>>>>> 5400af4241e2b79a156cfca7ef79ec71554c2453
      plt:plot(self.P[1]:zfloat(),'new probe')
    end

    image_error = self:image_error()
<<<<<<< HEAD
    probe_error = self:probe_error()

    u.printf('iteration %-3d: e_mod = %-02.02g    e_overlap = %-02.02g    e_image = %-02.02g  e_probe = %-02.02g  %d/%d modulus updates',i,mod_error  or -1,overlap_error or -1 ,image_error or -1, probe_error or -1, mod_updates, self.K)
    -- print('--------------------------------------------------------------------------------')
=======

    u.printf('iteration %03d: e_mod = %g    e_overlap = %g    e_image = %g',i,mod_error,overlap_error,image_error)
    print('--------------------------------------------------------------------------------')
>>>>>>> 5400af4241e2b79a156cfca7ef79ec71554c2453
  end
  plt:plot(self.O[1]:zfloat(),'object - it '..steps)
  plt:plot(self.P[1]:zfloat(),'new probe')
end

return engine