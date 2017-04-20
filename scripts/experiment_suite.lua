require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
-- require 'hypero'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local optim = require "optim"
local znn = require "dptycho.znn"

local zt = require "ztorch.complex"
local stats = require "dptycho.util.stats"
local simul = require 'dptycho.simulation'
local ptycho = require 'dptycho.core.ptycho'
local ptychocore = require 'dptycho.core'


function get_data(pot_path,dose,overlap,N,E,probe,probe_diameter,shift)

  local probe_int = probe:clone():norm()
  probe_int:div(probe_int:max())
  local probe_mask = torch.ge(probe_int:re(),1e-2):int()
  -- plt:plot(probe_mask:float())
  local s = simul.simulator()
  local pot = s:load_potential(pot_path)
    -- local pos = s:round_scan_ROI_positions(probe_diameter*(1-overlap),pot:size(2)-N-1,pot:size(2)-N-1,7)
  local pos = 0
  pos = s:raster_positions_overlap(pot:size(2)-N,probe_mask,overlap,shift,4)
  pos = pos:int() + 1

  local binning = 8

  local I = s:dp_multislice(pos,probe, N, binning, E, dose)
  -- local I = s:dp_projected(pos,probe, N, binning, E, dose)

  local result = {}
  result.a = I:cuda():sqrt()
  result.probe = probe:view(1,1,probe:size(1),probe:size(2))
  result.object = s.T_proj
  result.pos = pos

  return result
end

function main()
  -- local conn = hypero.connect()
  -- local bat = conn:battery('bayes_opt 4v6x 300', '1.0')

  -- local f = hdf5.open('/home/philipp/drop/Philipp/mypapers/lowdose/data/figure4_gradient_steps/init.h5','r')
  -- local Or = f:read('/oinitr'):all()
  -- local Oi = f:read('/oiniti'):all()
  -- local obj = torch.ZCudaTensor(Or:size()):copyRe(Or:cuda()):copyIm(Oi:cuda())
  -- obj =
  -- pprint(obj)
  -- plt:plot(obj[1][1],'oinit')
  Or = nil
  Oi = nil
  path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/scan10/'
  local dose = {4.8e6}
  -- local dose = {4.8e6}6e5,1.5e6,2.8e6,4.8e6,8.4e6,1.5e7,2.6e7,4.6e7,8.5e7,1.45e8
  local overlap = {0.7}--,0.7,0.75,0.8}0.45,0.5,0.55,0.6,
  local sampl = {'4V6X_200','1RYP_200','4HHB_200'}
  local lr = {2.5e-4}--{1e-4,5e-4}
  local momentum = {0}--{0,0.99,0.95}
  local ID =3

  for l=1,20 do
  local par = ptycho.params.DEFAULT_PARAMS_TWF()

  local N = 256
  local E = 300e3
  par.Np = 1
  par.No = 1
  par.bg_solution = nil
  par.plot_every = 30000
  par.plot_start = 1
  par.show_plots = false
  par.beta = 0.9
  par.fourier_relax_factor = 8e-2
  par.position_refinement_start = 250
  par.position_refinement_every = 3
  par.position_refinement_max_disp = 2
  par.fm_support_radius = function(it) return nil end
  par.fm_mask_radius = function(it) return nil end

  par.probe_update_start = 6000
  par.probe_support = 0.5
  par.probe_regularization_amplitude = function(it) return nil end
  par.probe_inertia = 0
  par.probe_lowpass_fwhm = function(it) return nil end

  par.object_highpass_fwhm = function(it) return nil end
  par.object_inertia = 0
  par.object_init = 'rand'
  par.object_init_truncation_threshold = 85
  par.object_initial = nil

  par.P_Q_iterations = 10
  par.probe_init = 'copy'
  par.copy_object = false--true
  par.margin = 0
  par.background_correction_start = 1e5

  par.save_interval = 60000
  par.save_raw_data = true
  par.save_path = '/home/philipp/dropbox/Philipp/mypapers/lowdose/data/figure3_averaging/1ryp/5/'

  par.O_denom_regul_factor_start = 0
  par.O_denom_regul_factor_end = 0

  par.P = nil
  par.O = nil

  par.twf.a_h = 50
  par.twf.a_lb = 1e-4
  par.twf.a_ub = 2e1
  par.twf.mu_max = 0.01
  par.twf.tau0 = 10
  par.twf.do_truncate = false
  par.twf.diagnostics = false
  par.twf.nu = 0

  for w,lr0 in ipairs(lr) do
    for q,mom0 in ipairs(momentum) do

  -- config for sgd
  -- par.optim_config = {}
  -- par.optim_state = {}
  -- par.optim_config.learningRate = lr0
  -- par.optim_config.learningRateDecay = mom0
  -- par.optim_config.weightDecay = 0
  -- par.optim_config.momentum = 0

  -- config for cg
  par.optim_config = {}
  par.optim_config.maxIter = 3
  par.optim_config.sig = 0.5
  par.optim_config.red = 1e8
  par.optim_config.break_on_success = true
  par.optim_config.verbose = false
  par.optim_state_object = {}

  par.regularizer = znn.SpatialSmoothnessCriterion --znn.SpatialSmoothnessCriterion--znn.BM3D_MSE_Criterion--znn.SpatialSmoothnessCriterion
  par.optimizer = optim.cg -- nag sgd cg

  par.regularization_params = {}
  par.regularization_params.amplitude = 2
  par.regularization_params.start_denoising = 2
  par.regularization_params.denoise_interval = 1
  par.regularization_params.sigma_denoise = 0.03

  par.calculate_dose_from_probe = true
  par.stopping_threshold = 1e-5

  par.experiment.z = 1.82
  par.experiment.E = E
  par.experiment.det_pix = 70e-6
  par.experiment.N_det_pix = N
  local probes = {'probe_rfzp','probe_fzp','probe_blr','probe_coneblr','probe_def7'}
  local s = simul.simulator()
  local d = 2.0

  local f = hdf5.open('/home/philipp/drop/Public/'..probes[3]..'.h5','r')
  local pr = f:read('/pr'):all()
  local pi = f:read('/pi'):all()
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  -- plt:plot(probe:zfloat(),'probe_rfzp')
  f:close()

  local overlap0 = overlap[1]
  for i,dose0 in ipairs(dose) do
    for j,sample in ipairs(sampl) do
      -- local sample = '1RYP_200'
      -- print(dose0)

      local shift = torch.FloatTensor{-4 ,-2}
      local data = get_data('/home/philipp/drop/Public/'..sample..'.h5',dose0,overlap0,N,E,probe,63,shift)

      par.pos = data.pos
      par.dpos = data.pos:clone():add(-1,data.pos:clone():int()):float()
      par.object_solution = data.object:clone()
      par.probe_solution = data.probe:clone()
      par.fmask = data.a:clone():fill(1)
      par.a = data.a
      par.run_label = string.format('%05d_s_%s_ov_%d_d_%d_run_%d',ID,sample,overlap0*100,dose0,1)

      local eng = ptycho.TWF_engine(par)
      eng:iterate(100)
      -- local hp = {run_label = str, nu = nu0, dose = eng.electrons_per_angstrom2, total_counts = eng.I_total, counts_per_valid_pixel = eng.counts_per_valid_pixel, MoverN = eng.total_nonzero_measurements/eng.pixels_with_sufficient_exposure, overlap = overlap0, probe_type = probe_type, method = 'cg', learningRate = par.optim_config.learningRate, learningRateDecay = par.optim_config.learningRateDecay, momentum = par.optim_config.momentum}
      -- local md = {hostname = 'work', dataset = sample}
      -- local res = { final_img_error = eng.img_errors[eng.i], final_rel_error = eng.rel_errors[eng.i], img_err = eng.img_errors:totable(), rel_err = eng.rel_errors:totable()}

      -- hex:setParam(hp)
      -- hex:setMeta(md)
      -- hex:setResult(res)

      eng = nil
      ID = ID + 1
      collectgarbage()
    end -- end nu
  end -- end dose
end
end
end
end

main()
