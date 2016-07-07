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
  fm_support_radius = function(it) return nil end,
  fm_mask_radius = function(it) return nil end,
  probe_regularization_amplitude = function(it) return nil end,
  probe_inertia = 1e-9,
  probe_lowpass_fwhm = function(it) return nil end,
  object_highpass_fwhm = function(it) return nil end,
  object_inertia = 0.1,
  P_Q_iterations = 5,
  copy_solution = false,
  margin = 0,
  background_correction_start = 1e5,
  save_interval = 5,
  save_path = '/tmp/',
  save_raw_data = false,
  twf = {
    a_h = 25,
    mu_max = 0.01,
    tau0 = 330,
  }
}

function m.DEFAULT_PARAMS()
  return u.copytable(_DEFAULT_PARAMS)
end

function m.DEFAULT_PARAMS_TWF()
  local t = u.copytable(_DEFAULT_PARAMS)
  t.object_inertia = nil
  return t
end

return m
