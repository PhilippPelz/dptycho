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
local ptychocore = require 'dptycho.core'

function get_data(pot_path,dose,overlap,N,E,probe)

  local probe_int = probe:clone():norm()
  probe_int:div(probe_int:max())
  local probe_mask = torch.ge(probe_int:re(),1e-2):int()
  -- plt:plot(probe_mask:float())

  local s = simul.simulator()
  local pot = s:load_potential(pot_path)
  local pos = s:raster_positions_overlap(500-N,probe_mask,overlap)
  pos = pos:int() + 1

  local binning = 8

  local I = s:dp_multislice(pos,probe, N, binning, E, dose)

  local result = {}
  result.a = I:cuda():sqrt()
  result.probe = probe
  result.object = s.T_proj
  result.pos = pos

  return result
end

function main()
  local conn = hypero.connect()
  local bat = conn:battery('bayes_opt', '1.0')
  local hs = hypero.Sampler()

  local dose = {2e6}
  local electrons_per_angstrom = {5.62341325,    10.        ,    17.7827941 ,    31.6227766 ,
          56.23413252,   100.        ,   177.827941  ,   316.22776602,
         562.34132519,  1000.}
  local overlap = {0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9}
  local nu = {4e-1,3e-1,2e-1,5e-2}

  local par = ptycho.params.DEFAULT_PARAMS_TWF()

  local N = 256
  local E = 100e3
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
  par.experiment.E = E
  par.experiment.det_pix = 40e-6
  par.experiment.N_det_pix = N

  for probe_type = 3,3 do
    local s = simul.simulator()
    local probe = nil
    local d = 2.0

    if probe_type == 1 then

      local alpha_rad = 6e-3
      local C3_um = 500
      local defocus_nm = 1.2e3
      local C5_mm = 800
      local tx = 0
      local ty = 0
      local Nedge = 5
      local plotit = true

      probe = s:random_fzp(N,800,3)
      -- plt:plot(probe:zfloat(),'defocused RFZP')
      probe:fftshift()
      local prop = ptychocore.propagators.fresnel(N,d*1e-10,300e-9,u.physics(E).lambda)
      prop:fftshift()
      -- plt:plot(prop:zfloat(),'prop')
      plt:plot(probe:zfloat(),'defocused RFZP')
      -- for i=1,10 do
      probe:cmul(prop)
      probe:ifft()
      probe:fftshift()
      plt:plot(probe:zfloat(),'defocused RFZP')
        -- probe:fft()
      -- end
      -- plt:plotcx(probe)
      -- plt:plot(probe:zfloat(),'defocused RFZP')
    elseif probe_type == 2 then
      probe = s:fzp(N,800,3)
      -- plt:plot(probe:zfloat():fftshift(),'FZP')
      probe:fftshift()
      local prop = ptychocore.propagators.fresnel(N,d*1e-10,300e-9,u.physics(E).lambda)
      prop:fftshift()
      probe:cmul(prop)
      probe:ifft()
      probe:fftshift()
      -- plt:plot(probe:zfloat(),'defocused RFZP')
    elseif probe_type == 3 then
      probe = s:random_probe2(N,0.12,0.2,0.05)
      -- plt:plot(probe:zfloat(),'band limited random')
    end

    -- print(overlap[1])
    local data = get_data('/home/philipp/vol26.h5',3.6e6,overlap[1],N,E,probe)
    par.pos = data.pos
    pprint(data.pos)
    par.dpos = data.pos:clone():add(-1,data.pos:clone():int())
    par.object_solution = data.object
    par.probe_solution = probe
    par.a = data.a
    par.fmask = data.a:clone():fill(1)

    local eng = ptycho.TWF_engine(par)
    local dose = eng.electrons_per_angstrom2

    u.printf('e-/A^2 : %f',dose)

    -- local run_config = {{200,ptycho.TWF_engine}
    -- -- ,{200,ptycho.TWF_engine}
    -- }
    --
    -- local runner = ptycho.Runner(run_config,par)
    -- runner:run()
  end
end

main()
