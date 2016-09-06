local classic = require 'classic'

local m1 = classic.class(...)

m1.el = 1.60217646e-19 --C
m1.h = 6.62606896e-34 -- J*S
m1.h_ev = 4.13566733e-15 --%eV*s
m1.h_bar = 1.054571628e-34 --J*s
m1.h_bar_ev = 6.58211899e-16 --eV*s

m1.Na = 6.02214179e23 -- mol-1
m1.re = 2.817940289458e-15 -- m
m1.rw=2.976e-10 --m

m1.me = 9.10938215e-31 -- kg
m1.me_ev = 0.510998910e6 -- ev/c^2
m1.kb = 1.3806503e-23 --m^2kgs^-2K^-1

m1.eps0 = 8.854187817620e-12 -- F/m

function m1:_init(V)
  local h = 6.626068e-34
  local e = 1.60217646e-19
  local m = 9.10938188e-31
  local c = 2.99792458e8
  m1.lambda = h/math.sqrt(e*V*m*(e/m*V/c^2 + 2 ))
  m1.relmass = m1.me + m1.el*V/(c^2)
  m1.sigma = 2*math.pi*m1.relmass*m1.el*m1.lambda/(m1.h^2)
  return m1
end


return m1
