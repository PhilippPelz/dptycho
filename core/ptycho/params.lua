local classic = require 'classic'
local u = require "dptycho.util"
local ptycho = require("dptycho.core.ptycho")
local m = classic.module(...)
local znn = require 'dptycho.znn'
local optim = require 'optim'

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
  position_refinement_method = 'marchesini', -- 'marchesini', 'annealing'
  position_refinement_start = 1e5,
  position_refinement_stop = 1e6,
  position_refinement_every = 3,
  position_refinement_max_disp = 2,
  position_refinement_trials = 4,
  fm_support_radius = function(it) return nil end,
  fm_mask_radius = function(it) return nil end,

  probe_update_start = 5,
  probe_support = nil,
  probe_support_fourier = nil,
  probe_regularization_amplitude = function(it) return nil end,
  probe_inertia = 1e-9,
  probe_lowpass_fwhm = function(it) return nil end,
  probe_keep_intensity = true,
  probe_init = 'copy', -- 'copy,'disc'
  probe_do_enforce_modulus = false,
  probe_modulus = nil,

  object_highpass_fwhm = function(it) return nil end,
  object_inertia = 1e-8,
  object_init = 'const', -- trunc, gcl, copy_solution
  object_init_truncation_threshold = 0.8,
  object_update_start = 0,

  P_Q_iterations = 5,
  margin = 0,
  background_correction_start = 1e5,

  calculate_dose_from_probe = false,
  stopping_threshold = 1e-5,

  save_interval = 5,
  save_path = '/tmp/',
  save_raw_data = false,
  run_label = 'ptycho',

  regularizer = znn.SpatialSmoothnessCriterion,
  optimizer = optim.cg,

  O_denom_regul_factor_start = 1e-6,
  O_denom_regul_factor_end = 1e-9,
  twf = {
    a_h = 25,
    a_lb = 0.3,
    a_ub = 25,
    mu_max = 0.01,
    tau0 = 330,
    nu = 1e-2,
    do_truncate = false,
    diagnostics = false,
    gradient_damping_radius = nil,
    gradient_damping_factor = 0.1
  },
  experiment = {
    z = 0.5,
    E = 300e3,
    det_pix = 30e-6,
    N_det_pix = 256
  },
  regularization_params = {
    start_denoising = 5,
    denoise_interval = 2,
    sigma_denoise = 25
  },
  optim_state_probe = {},
  optim_state = {},
  optim_config = {
    learningRate = 4e-5,
    learningRateDecay = 0,
    weightDecay = 0,
    momentum = 0
  },
  optim_config_probe = {
    learningRate = 4e-5,
    learningRateDecay = 0,
    weightDecay = 0,
    momentum = 0
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
  default.probe_inertia = 0
  return default
end

return m
