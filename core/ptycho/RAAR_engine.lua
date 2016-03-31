local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine"

local engine, super = classic.class(...,base_engine)

function engine:_init(pos,a,nmodes_probe,nmodes_object,solution,probe)
  super._init(pos,a,nmodes_probe,nmodes_object,solution,probe)
end

function engine:iterate(steps)
  local mod_error, overlap_error, image_error
  printMinMax(self.solution,'solution      ')
  for i=1,steps do
    -- printMinMax(self.z,   'z                             ')
    self:P_Q(self.z,self.P_Qz)
    mod_error = self:P_F(self.z,self.P_Fz)
    -- printMinMax(self.P_Qz,'P_Q z                         ')
    -- printMinMax(self.P_Fz,'P_F z                         ')
    if i % 10 == 0 then
      plt:plot(self.O[1]:zfloat(),'object - it '..i)
    end
    -- (1-2beta)P_F z
    self.P_Fz:mul(1-2*self.beta)
    -- printMinMax(self.P_Fz, (1-2*self.beta)..' *P_F z                 ')
    -- beta(P_Q-I) z
    self.z:add(-1,self.P_Qz)
    self.z:mul(self.beta)
    -- printMinMax(self.z, self.beta..' *(P_Q-I) z                 ')
    -- beta(P_Q-I) z + (1-2beta)P_F z
    self.z:add(self.P_Fz)
    self.P_Fz:mul(1/(1-2*self.beta))
    -- printMinMax(self.z,   self.beta..' *(I-P_Q) z + '.. (1-2*self.beta) ..' P_F z')
    -- 2 beta P_Q P_F
    self:P_Q(self.P_Fz,self.P_Qz)
    overlap_error = self:overlap_error(self.P_Fz,self.P_Qz)

    self.P_Qz:mul(2*self.beta)
    -- printMinMax(self.P_Qz,   (2*self.beta)..' *P_Q P_F z              ')
    -- 2 beta P_Q P_F + beta(P_Q-I) z + (1-2beta)P_F z
    self.z:add(self.P_Qz)
    -- printMinMax(self.z,   'z_{i+1}                       ')
    if self.update_probe then
      self:simpleRefineProbe()
    end
    image_error = self:image_error()

    u.printf('iteration %03d: e_mod = %g    e_overlap = %g    e_image = %g',i,mod_error,overlap_error,image_error)
    print('--------------------------------------------------------------------------------')
  end
end
