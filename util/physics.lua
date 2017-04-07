local classic = require 'classic'
require 'pprint'
local m1 = classic.class(...)

function m1:_init(V)
  local h = 6.626068e-34
  local e = 1.60217646e-19
  local m = 9.10938188e-31
  local c = 2.99792458e8
  self.el = 1.60217646e-19 --C
  self.h = 6.62606896e-34 -- J*S
  self.h_ev = 4.13566733e-15 --%eV*s
  self.h_bar = 1.054571628e-34 --J*s
  self.h_bar_ev = 6.58211899e-16 --eV*s

  self.Na = 6.02214179e23 -- mol-1
  self.re = 2.817940289458e-15 -- m
  self.rw=2.976e-10 --m

  self.me = 9.10938215e-31 -- kg
  self.me_ev = 0.510998910e6 -- ev/c^2
  self.kb = 1.3806503e-23 --m^2kgs^-2K^-1

  self.eps0 = 8.854187817620e-12 -- F/m
  self.lambda = h/math.sqrt(e*V*m*(e/m*V/c^2 + 2 ))
  -- print(self.lambda)
  self.relmass = self.me + self.el*V/(c^2)
  self.sigma = 2*math.pi*self.relmass*self.el*self.lambda/(self.h^2)
  return self
end

function m1:real_space_resolution(z,dpix,Npix)
  local res = (self.lambda * z)
  res= res / (dpix * Npix)
  return res
end

-- function m1:real_space_resolution(alpha)
--   return self.lambda / math.sin(alpha)
-- end

function m1:qmax(alpha)
  return math.sin(alpha)/self.lambda
end


return m1
