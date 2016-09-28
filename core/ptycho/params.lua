local classic = require 'classic'
local u = require "dptycho.util"
local ptycho = require("dptycho.core.ptycho")
local m = classic.module(...)

local _DEFAULT_PARAMS = {
  pos = torch.FloatTensor(),
  dpos = torch.FloatTensor(),
  dpos_solution = nil,
  a = nil,
  Np = 1,
  No = 1,
  object_solution = nil,
  probe_solution = nil,
  probe = nil,
  fmask = nil,
  bg_solution = nil,
  plot_every = 5,
  plot_start = 1,
  show_plots = true,
  beta = 1,
  fourier_relax_factor = 5e-2,
  position_refinement_start = 1e5,
  position_refinement_every = 3,
  position_refinement_max_disp = 2,
  fm_support_radius = function(it) return nil end,
  fm_mask_radius = function(it) return nil end,

  probe_update_start = 5,
  probe_support = nil,
  probe_regularization_amplitude = function(it) return nil end,
  probe_inertia = 1e-9,
  probe_lowpass_fwhm = function(it) return nil end,

  object_highpass_fwhm = function(it) return nil end,
  object_inertia = 1e-8,
  object_init = 'const', -- trunc, gcl, copy_solution
  object_init_truncation_threshold = 0.8,

  P_Q_iterations = 5,
  copy_probe = false,
  copy_object = false,
  margin = 0,
  background_correction_start = 1e5,

  calculate_dose_from_probe = false,
  stopping_threshold = 1e-5,

  save_interval = 5,
  save_path = '/tmp/',
  save_raw_data = false,
  run_label = 'ptycho',

  O_denom_regul_factor_start = 1e-6,
  O_denom_regul_factor_end = 1e-9,
  twf = {
    a_h = 25,
    a_lb = 0.3,
    a_ub = 25,
    mu_max = 0.01,
    tau0 = 330,
    nu = 1e-2
  },
  experiment = {
    z = 0.5,
    E = 300e3,
    det_pix = 30e-6,
    N_det_pix = 256
  }
}

function m.DEFAULT_PARAMS()
  local default =  u.copytable(_DEFAULT_PARAMS)
  default.ops = ptycho.ops
  return default
end

function m.DEFAULT_PARAMS_TWF()
  local default =  u.copytable(_DEFAULT_PARAMS)
  default.ops = ptycho.ops
  default.object_inertia = 0
  return default
end

return m
