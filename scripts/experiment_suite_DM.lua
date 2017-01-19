require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'hypero'
local classic = require 'classic'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local znn = require "dptycho.znn"

local zt = require "ztorch.complex"
local stats = require "dptycho.util.stats"
local simul = require 'dptycho.simulation'
local ptycho = require 'dptycho.core.ptycho'
local ptychocore = require 'dptycho.core'

function get_data(pot_path,dose,overlap,N,E,probe)

  local probe_int = probe:clone():norm()
  probe_int:div(probe_int:max())
  local probe_mask = torch.ge(probe_int:re(),5e-2):int()
  -- plt:plot(probe_mask:float())

  local s = simul.simulator()
  local pot = s:load_potential(pot_path)
  local pos = s:raster_positions_overlap(500-N,probe_mask,overlap)
  pos = pos:int() + 1

  local binning = 8

  local I = s:dp_multislice(pos,probe, N, binning, E, dose)
  -- local I = s:dp_projected(pos,probe, N, binning, E, dose)

  local result = {}
  result.a = I:cuda()
  result.probe = probe:view(1,1,probe:size(1),probe:size(2))
  result.object = s.T_proj
  result.pos = pos

  return result
end

function main()
  local conn = hypero.connect()
  local bat = conn:battery('bayes_opt 4v6x 300', '1.0')

  local f = hdf5.open('/home/philipp/drop/Philipp/mypapers/lowdose/data/figure4_gradient_steps/init.h5','r')
  local Or = f:read('/oinitr'):all()
  local Oi = f:read('/oiniti'):all()
  local obj = torch.ZCudaTensor(Or:size()):copyRe(Or:cuda()):copyIm(Oi:cuda())
  -- obj =
  -- pprint(obj)
  -- plt:plot(obj[1][1],'oinit')
  Or = nil
  Oi = nil

  local dose = {6e12}
  -- local dose = {4.8e6}6e5,1.5e6,2.8e6,4.8e6,8.4e6,1.5e7,2.6e7,4.6e7,8.5e7,1.45e8
  -- local dose = {1.45e8,8.5e7,4.6e7,2.6e7,1.5e7,8.4e6,4.8e6,2.8e6,1.5e6}
  local electrons_per_angstrom = {5.62341325,    10.        ,    17.7827941 ,    31.6227766 ,
          56.23413252,   100.        ,   177.827941  ,   316.22776602,
         562.34132519}
 local electrons_per_angstrom = {5.62341325,    10.        ,    17.7827941 ,    31.6227766 ,
         56.23413252,   100.        ,   177.827941  ,   316.22776602,
        562.34132519}
  local electrons_per_angstrom = {5.62341325,    10.        ,    17.7827941 ,    31.6227766 ,
          56.23413252,   100.        ,   177.827941  ,   316.22776602,
         562.34132519}
  local overlap = {0.70}--,0.7,0.75,0.8}0.45,0.5,0.55,0.6,
  -- local overlap = {0.45}
  local nu = {10e-2}--{10e-2,5e-2,1e-2,5e-3}--4e-2,2e-1,1e-1,
  local lr = {2.5e-5}--{1e-4,5e-4}
  local momentum = {0}--{0,0.99,0.95}
  local ID =23
  for l=1,1 do
  local par = ptycho.params.DEFAULT_PARAMS_TWF()

  local N = 256
  local E = 300e3
  par.Np = 1
  par.No = 1
  par.bg_solution = nil
  par.plot_every = 5
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
  par.object_init = 'trunc'
  par.object_init_truncation_threshold = 80
  par.object_initial = obj

  par.P_Q_iterations = 10
  par.copy_probe = true
  par.copy_object = false--true
  par.margin = 0
  par.background_correction_start = 1e5

  par.save_interval = 60000
  par.save_raw_data = true
  par.save_path = '/home/philipp/drop/Philipp/mypapers/lowdose/data/probe/'

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

  for w,lr0 in ipairs(lr) do
    for q,mom0 in ipairs(momentum) do

  -- config for sgd
  par.optim_config = {}
  par.optim_state = {}
  par.optim_config.learningRate = lr0
  par.optim_config.learningRateDecay = mom0
  par.optim_config.weightDecay = 0
  par.optim_config.momentum = 0

  -- config for L-BFGS
  -- par.optim_config = {}
  -- par.optim_state = {}
  -- par.optim_config.maxIter = 10
  -- par.optim_config.maxEval = par.optim_config.maxIter*1.25
  -- par.optim_config.tolFun = 1e-5
  -- par.optim_config.tolX = 1e-9
  -- par.optim_config.nCorrection = 5
  -- par.optim_config.lineSearch = optim.lswolfe
  --
  -- par.optim_config.lineSearchOptions = {}
  -- par.optim_config.lineSearchOptions.c1 = 1e-4
  -- par.optim_config.lineSearchOptions.c2 = 0.9
  -- par.optim_config.lineSearchOptions.tolX = 1e-9
  -- par.optim_config.lineSearchOptions.maxIter = 20
  -- par.optim_config.lineSearchOptions.verbose = false
  --
  -- par.optim_config.learningRate = 1
  -- par.optim_config.verbose = false

  -- config for nag
  -- par.optim_config = {}
  -- par.optim_state = {}
  -- par.optim_config.learningRate = 1e-8
  -- par.optim_config.learningRateDecay = 3e-2
  -- par.optim_config.weightDecay = 0
  -- par.optim_config.momentum = 0.9
  -- par.optim_config.dampening = nil

  -- config for cg
  par.optim_config = {}
  par.optim_config.maxIter = 10
  par.optim_config.sig = 0.5
  par.optim_config.red = 1e8
  par.optim_config.break_on_success = true
  par.optim_config.verbose = false
  par.optim_state = {}

  par.regularizer = znn.BM3D_MSE_Criterion --znn.SpatialSmoothnessCriterion--znn.BM3D_MSE_Criterion--znn.SpatialSmoothnessCriterion
  par.optimizer = optim.cg -- nag sgd cg

  par.regularization_params = {}
  par.regularization_params.amplitude = 6e-1
  par.regularization_params.start_denoising = 10
  par.regularization_params.denoise_interval = 15
  par.regularization_params.sigma_denoise = 0.1

  par.calculate_dose_from_probe = true
  par.stopping_threshold = 1e-5

  par.experiment.z = 1.1
  par.experiment.E = E
  par.experiment.det_pix = 40e-6
  par.experiment.N_det_pix = N

  for probe_type = 3,3 do
    local s = simul.simulator()
    local probe = nil
    local d = 2.0

    if probe_type == 1 then
      local f = hdf5.open('/home/philipp/drop/Public/probe_rfzp.h5','r')
      local pr = f:read('/pr'):all()
      local pi = f:read('/pi'):all()
      probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
      -- plt:plot(probe:zfloat(),'probe_rfzp')
      f:close()
    elseif probe_type == 2 then
      local f = hdf5.open('/home/philipp/drop/Public/probe_fzp.h5','r')
      local pi = f:read('/pi'):all()
      local pr = f:read('/pr'):all()
      probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
      f:close()
    elseif probe_type == 3 then
      local f = hdf5.open('/home/philipp/drop/Public/probe_blr2.h5','r')
      local pr = f:read('/pr'):all()
      local pi = f:read('/pi'):all()
      probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
      -- plt:plot(probe:zfloat(),'probe')
      f:close()
    elseif probe_type == 4 then
      local f = hdf5.open('/home/philipp/drop/Public/probe_coneblr.h5','r')
      local pr = f:read('/pr'):all()
      local pi = f:read('/pi'):all()
      probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
      -- plt:plot(probe:zfloat(),'defocused probe')
      f:close()
    elseif probe_type == 5 then
      local f = hdf5.open('/home/philipp/drop/Public/probe_def5.h5','r')
      local pr = f:read('/pr'):all()
      local pi = f:read('/pi'):all()
      probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
      -- plt:plot(probe:zfloat(),'defocused probe')
      f:close()
    end


    for i,dose0 in ipairs(dose) do
      for j,overlap0 in ipairs(overlap) do
        local sample = '4V6X_200'
        -- print(dose0)
        local data = get_data('/home/philipp/drop/Public/'..sample..'.h5',dose0,overlap0,N,E,probe)
        par.pos = data.pos
        par.dpos = data.pos:clone():add(-1,data.pos:clone():int()):float()
        par.object_solution = data.object:clone()
        par.probe_solution = data.probe:clone()
        par.fmask = data.a:clone():fill(1)

        for k,nu0 in ipairs(nu) do
          local nu = string.gsub(string.format('%g',nu0),',','p')
          local str = string.format('%05d_s_%s_ov_%d_d_%d_nu_%s',ID,sample,overlap0*100,dose0,nu)
          par.run_label = str
          par.twf.nu = nu0
          par.a = data.a:clone()
          -- print()
          -- print('a sum')
          -- print(par.a:sum())
          -- print()

            local nu = string.gsub(string.format('%g',nu0),',','p')
            local str = string.format('%05d_s_%s_ov_%d_d_%d_nu_%s_run_%d',ID,sample,overlap0*100,dose0,nu,1)
            par.run_label = str
            par.twf.nu = 3e-2
            par.a = data.a:clone()
            -- print(par.a:sum())
            -- print()

            local hex = bat:experiment()

            local eng = ptycho.DM_engine_subpix(par)
            eng:iterate(50)
            local hp = {run_label = str, nu = nu0, dose = eng.electrons_per_angstrom2, total_counts = eng.I_total, counts_per_valid_pixel = eng.counts_per_valid_pixel, MoverN = eng.total_nonzero_measurements/eng.pixels_with_sufficient_exposure, overlap = overlap0, probe_type = probe_type, method = 'cg', learningRate = par.optim_config.learningRate, learningRateDecay = par.optim_config.learningRateDecay, momentum = par.optim_config.momentum}
            local md = {hostname = 'work', dataset = sample}
            local res = { final_img_error = eng.img_error[eng.i], final_rel_error = eng.rel_error[eng.i], img_err = eng.img_error:totable(), rel_err = eng.rel_error:totable()}

            hex:setParam(hp)
            hex:setMeta(md)
            hex:setResult(res)

            eng = nil
            ID = ID + 1
            collectgarbage()
          -- end
        end
      end -- end nu
    end -- end dose
  end
end
end
end
end

main()
