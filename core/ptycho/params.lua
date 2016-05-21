local classic = require 'classic'
local u = require "dptycho.util"
local m = classic.module(...)

local _DEFAULT_PARAMS = {
  pos = torch.FloatTensor(),
  dpos = torch.FloatTensor(),
  dpos_solution = nil,
  a = nil,
  Np = 1,
  No = 1,
  solution = nil,
  probe = nil,
  fmask = nil,
  bg_solution = nil,
  plot_every = 5,
  plot_start = 1,
  show_plots = true,
  beta = 1,
  fourier_relax_factor = 5e-2,
  position_refinement_start = 1e5,
  position_refinement_max_disp = 2,
  probe_update_start = 5,
  probe_support = nil,
  fm_support = nil,
  probe_regularization_amplitude = function(it) return nil end,
  probe_inertia = 1e-9,
  probe_lowpass_fwhm = function(it) return nil end,
  object_highpass_fwhm = function(it) return nil end,
  object_inertia = 0.1,
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
