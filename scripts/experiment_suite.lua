require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'hypero'
local classic = require 'classic'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local znn = require "dptycho.znn"
local plt = plot()
local zt = require "ztorch.complex"
local stats = require "dptycho.util.stats"
local simul = require 'dptycho.simulation'
local ptycho = require 'dptycho.core.ptycho'

function get_data(pot_path,dose,npos,probe_type)
  local s = simul.simulator()
  local pot = s:load_potential(pot_path)
  local pos = s:get_positions_raster(npos,500-N)
  pos = pos:int() + 1

  if probe_type == 1 then
    local N = 256
    local d = 2.0
    local alpha_rad = 5e-3
    local C3_um = 500
    local defocus_nm = 1.8e3
    local C5_mm = 800
    local tx = 0
    local ty = 0
    local Nedge = 10
    local plotit = true
  if probe_type == 1 then
    local probe = s:focused_probe(E, N, d, alpha_rad, defocus_nm, C3_um , C5_mm, tx ,ty , Nedge , plotit)
  elseif probe_type == 2 then
    local probe = s:random_probe(N)
  elseif probe_type == 3 then
    local probe = s:random_probe3(N,0.10,0.2,0.10)
  end

  local binning = 8
  local E = 100e3

  local I = s:dp_multislice(pos,probe, N, binning, E, dose)

  local result = {}
  result.a = I:float():sqrt()
  result.probe = probe
  result.object = s.T_proj
  result.positions = pos

  return result
end

function main()
  local conn = hypero.connect()
  local bat = conn:battery('bayes_opt', '1.0')
  local hs = hypero.Sampler()

  local doses = {}
  local npos = {}
  local nu = {4e-1,3e-1,2e-1,5e-2}

  local par = ptycho.params.DEFAULT_PARAMS_TWF()

  par.Np = 1
  par.No = 1
  par.bg_solution = nil
  par.plot_every = 20
  par.plot_start = 1
  par.show_plots = true
  par.beta = 0.9
  par.fourier_relax_factor = 8e-2
  par.position_refinement_start = 250
  par.position_refinement_every = 3
  par.position_refinement_max_disp = 2
  par.fm_support_radius = function(it) return nil end
  par.fm_mask_radius = function(it) return nil end

  par.probe_update_start = 200
  par.probe_support = 0.5
  par.probe_regularization_amplitude = function(it) return nil end
  par.probe_inertia = 0
  par.probe_lowpass_fwhm = function(it) return nil end

  par.object_highpass_fwhm = function(it) return nil end
  par.object_inertia = 0
  par.object_init = 'const'
  par.object_init_truncation_threshold = 94

  par.P_Q_iterations = 10
  par.copy_probe = true
  par.copy_object = false--true
  par.margin = 0
  par.background_correction_start = 1e5

  par.save_interval = 250
  par.save_path = '/tmp/'
  par.save_raw_data = true
  par.run_label = 'ptycho2'

  par.O_denom_regul_factor_start = 0
  par.O_denom_regul_factor_end = 0

  par.P = nil
  par.O = nil

  par.twf.a_h = 25
  par.twf.a_lb = 1e-3
  par.twf.a_ub = 1e1
  par.twf.mu_max = 0.01
  par.twf.tau0 = 10
  par.twf.nu = 2e-1

  par.experiment.z = 0.6
  par.experiment.E = 100e3
  par.experiment.det_pix = 40e-6
  par.experiment.N_det_pix = 256

  for d = 1,2 do
    local data = get_data('/home/philipp/vol26.h5',1e8,300,1)

    par.pos = pos
    par.dpos = dpos
    par.dpos_solution = dpos_solution
    par.object_solution = object_solution
    par.probe_solution = probe
    par.a = a
    par.fmask = fmask

    local run_config = {{200,ptycho.TWF_engine}
    -- ,{200,ptycho.TWF_engine}
    }

    local runner = ptycho.Runner(run_config,par)
    runner:run()
  end
end

main()
