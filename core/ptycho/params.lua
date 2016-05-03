local classic = require 'classic'
local u = require "dptycho.util"
local m = classic.module(...)

local _DEFAULT_PARAMS = {
  pos = torch.FloatTensor(),
  dpos = torch.FloatTensor(),
  a = nil,
  nmodes_probe = 1,
  nmodes_object = 1,
  solution = nil,
  probe = nil,
  fmask = nil,
  bg_solution = nil,
  plot_every = 5,
  beta = 1,
  fourier_relax_factor = 5e-2,
  position_refinement_start = 1e5,
  probe_update_start = 5,
  object_inertia = 0.1,
  probe_inertia = 1e-9,
  P_Q_iterations = 5,
  copy_solution = false,
  margin = 10,
  background_correction_start = 1e5,
  save_interval = 5,
  save_path = '/tmp/'
}

function m.DEFAULT_PARAMS()
  return u.copytable(_DEFAULT_PARAMS)
end

return m
