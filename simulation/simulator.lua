local classic = require 'classic'
local s = require 'dptycho.simulation'
local u = require 'dptycho.util'
local prop = require 'dptycho.core.propagators'
local py = require('fb.python')
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'hdf5'

local c = classic.class(...)

function c:_init()
  local pyfile = u.read_file(u.script_path() .. '/mtfdqe.py')
  local pyfile2 = u.read_file(u.script_path() .. '/probe.py')
  local pyfile3 = u.read_file(u.script_path() .. '/random_probe.py')
  py.exec(pyfile)
  py.exec(pyfile2)
  py.exec(pyfile3)
end

function c:load_potential(filename)
  local f = hdf5.open(filename,'r')
  self.dx = f:read('/dx'):all()[1]
  self.dz = f:read('/dz'):all()[1]
  local vr = f:read('/vreal'):all():cuda()
  local vi = f:read('/vimag'):all():cuda()
  self.v =  torch.ZCudaTensor(vr:size()):copyIm(vi):copyRe(vr)
  return self.v
end

function c:scale_probe_intensity(probe, total_dose, n_exposures)
  u.printf('n_exposures                   : %g',n_exposures)
  local poisson_noise = total_dose / n_exposures
  u.printf('poisson noise                 : %g',poisson_noise)
  local P_norm = probe:normall(2)^2
  local O_mean_transparency = self.T_proj:abs():mean() - 0.2
  local factor = math.sqrt(poisson_noise / P_norm / O_mean_transparency)
  u.printf('mean object transparency      : %g',O_mean_transparency)
  u.printf('multiply probe with           : %g',factor)
  probe:mul(factor)
  u.printf('P_norm                        : %g',probe:normall(2)^2)
end

function c:exitwaves_multislice(pos,in_psi, view_size, E, total_dose)
  self.physics = u.physics(E)
  -- u.printf('sigma=%g',self.physics.sigma)

  self.T = torch.ZCudaTensor(self.v:size()):polar(self.v:im():mul(-self.physics.sigma*self.dz):exp(),self.v:re():mul(self.physics.sigma*self.dz))

  local v_proj = self.v:sum(1)[1]
  -- plt:plot(v_proj,'proj potential')
  self.T_proj = torch.ZCudaTensor(v_proj:size()):polar(v_proj:im():mul(-self.physics.sigma*self.dz):exp(),v_proj:re():mul(self.physics.sigma*self.dz))
  -- plt:plot(self.T_proj,'proj T')

  self:scale_probe_intensity(in_psi,total_dose,pos:size(1))
  -- plt:plot(in_psi,'in_psi')
  local views = s.create_views(self.T, pos, view_size,1)
  local P0 = prop.fresnel(view_size,self.dx,self.dz,self.physics.lambda)
  P0:fftshift()
  -- plt:plot(P0:zfloat(),'fresnel prop')
  local out_psi = in_psi:view(1,view_size,view_size):repeatTensor(pos:size(1),1,1)
  local P = P0:view(1,view_size,view_size):expandAs(out_psi)

  for slice = 1, self.v:size(1) do
    for i, view in ipairs(views) do
      -- if i == 74 then
      --   plt:plot(view[slice]:zfloat(),'view[' .. i)
      -- end
      out_psi[i]:cmul(view[slice])
    end
    -- print(slice)
    out_psi:fftBatched()
    -- if true then
    --   plt:plot(out_psi[74]:zfloat(),'out_psi')
    -- end
    out_psi:cmul(P)
    out_psi:ifftBatched()
    -- if true then
    --   plt:plot(out_psi[22]:zfloat(),'out_psi')
    -- end
  end
  -- for i, view in ipairs(views) do
  --   plt:plot(out_psi[i],'out_psi '..i)
  -- end

  return out_psi
end

function c:exitwaves_projected(pos,in_psi, view_size, E, total_dose)
  self.physics = u.physics(E)
  local v_proj = self.v:sum(1)[1]
  -- plt:plot(v_proj,'proj potential')
  self.T_proj = torch.ZCudaTensor(v_proj:size()):polar(v_proj:im():mul(-self.physics.sigma*self.dz):exp(),v_proj:re():mul(self.physics.sigma*self.dz))
  -- plt:plot(self.T_proj,'proj T')

  self:scale_probe_intensity(in_psi,total_dose,pos:size(1))
  local views = s.create_views(self.T_proj, pos, view_size,0)
  local out_psi = in_psi:view(1,view_size,view_size):repeatTensor(pos:size(1),1,1)
  for i, view in ipairs(views) do
    out_psi[i]:cmul(view)
    -- plt:plot(out_psi[i],'out_psi[i]')
  end
  return out_psi
end

function c:dp_multislice(pos,in_psi, view_size, binning, E, total_dose)
  u.printf('total_dose                   : %g',total_dose)
  local out_psi = self:exitwaves_multislice(pos,in_psi, view_size, E, total_dose)
  local mtf, dqe = self:get_detector_MTF_DQE('K2',binning,view_size)
  out_psi:fftBatched()
  local I_noisy = self:simulate_detector(out_psi,mtf:cuda(),dqe:cuda(),total_dose)
  return I_noisy
end

function c:dp_projected(pos,in_psi, view_size, binning, E, total_dose)

  self.poisson_noise = total_dose / pos:size(1)
  u.printf('poisson noise                 : %g',self.poisson_noise)

  local out_psi = self:exitwaves_projected(pos,in_psi, view_size, E, total_dose)
  -- for i=1,5 do
  --   plt:plot(out_psi[i]:zfloat(),string.format('out_psi[%d]',i))
  -- end
  local mtf, dqe = self:get_detector_MTF_DQE('K2',binning,view_size)
  out_psi:fftBatched()
  local I_noisy = self:simulate_detector(out_psi,mtf:cuda(),dqe:cuda(),total_dose)
  return I_noisy
end

function c:simulate_detector(in_psi, mtf, dqe, total_dose)
  mtf:mul(mtf:nElement()/mtf:sum())
  local sqrt_nnps = mtf:clone():pow(2):cdiv(dqe)
  sqrt_nnps:mul(sqrt_nnps:nElement()/sqrt_nnps:sum())
  u.printf('sum(nnps)=%g elem(nnps)=%d',sqrt_nnps:sum(),sqrt_nnps:nElement())
  sqrt_nnps:sqrt()
  local sqrt_nnps_exp = sqrt_nnps:view(1,in_psi:size(2),in_psi:size(3)):expandAs(in_psi)
  local mtf_exp = mtf:view(1,in_psi:size(2),in_psi:size(3)):expandAs(in_psi)
  local dqe_exp = dqe:view(1,in_psi:size(2),in_psi:size(3)):expandAs(in_psi)
  local I = in_psi:norm()
  local Isum = I:sum(2):sum(3)
  pprint(Isum)
  -- print(I:normall(1))
  for i=1,I:size(1) do
      I[i]:fftshift()
  end
  -- for i=1,20 do
  --   plt:plot(I[i]:re():float(),string.format('I[%d]',i))
  --   u.printf('sum(I[%d]) = %g',i,I[i]:re():sum())
  -- end
  -- I = I:div(I:normall(1))
  -- local dose_per_exposure = total_dose / I:size(1)
  local ftI = I:fftBatched()
  -- for i=1,5 do
  --   plt:plot(I[i]:re():fftshift():float():log(),string.format('ftI[%d]',i))
  -- end
  local Sn = ftI:cmul(mtf_exp)--:cdiv(sqrt_nnps_exp)
  local I2 = Sn:ifftBatched():re()
  I2[torch.lt(I2,0)] = 0
  -- local I2norm = I2:div(I2:norm(1))
  local I_noise = u.stats.poisson(I2:float())
  for i=1,I:size(1) do
      I_noise[i]:copy(I_noise[i]:cuda():fftshift():float())
  end
  -- I:copyRe(I_noise:cuda()):fillIm(0)
  -- I:fftBatched()
  -- I:cmul(sqrt_nnps_exp)
  -- I:ifftBatched()

  -- local Isum2 = I:sum(2):sum(3)
  -- local Ifac = Isum:cdiv(Isum2):expandAs(mtf_exp)
  -- I:cmul(Ifac)

  -- for i=1,10 do
  --   plt:plot(I[i]:re():float(),string.format('I_noise[%d]',i))
  --   u.printf('sum(I_noise[%d]) = %g',i,
  --   I[i]:re():sum())
  -- end
  -- return I:re() -- I_noise is float tensor

  return I_noise
end

function c:get_positions_raster(Npos,size)
  return py.eval('raster_positions(n,s)',{n=Npos,s=size}):float()
end

function c:raster_positions_overlap(size,probe_mask,overlap)
  return py.eval('raster_positions_overlap(s,p,o)',{s=size,p=probe_mask,o=overlap}):float()
end

function c:focused_probe(E, N, d, alpha_rad, defocus_nm, C3_um , C5_mm, tx ,ty , Nedge , plot)
  local C3 = C3_um or 1000
  local C5 = C5_mm or 1
  local tx0 = tx or 0
  local ty0 = ty or 0
  local Ne = Nedge or 5
  local doplot = plot or false

  local pr, pi = table.unpack(py.eval('focused_probe(E, N, d, alpha_rad, defocus_nm, C3_um , C5_mm , tx ,ty , Nedge , plot)',{E=E,N=N,d=d,alpha_rad=alpha_rad,defocus_nm=defocus_nm,C3_um=C3,C5_mm=C5,tx=tx0,ty=ty0,Nedge=Ne,plot=doplot}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())

  return probe
end
function c:random_probe(N)
  local pr, pi = table.unpack(py.eval('blr_probe(N)',{N=N}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  return probe
end
function c:random_probe2(N,rs_rad,fs_rad1,fs_rad2)
  local pr, pi = table.unpack(py.eval('blr_probe2(N,rs_rad,fs_rad1,fs_rad2)',{N=N,rs_rad=rs_rad,fs_rad1=fs_rad1,fs_rad2=fs_rad2}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  return probe
end
function c:random_probe3(N,rs_rad,fs_rad1,fs_rad2)
  local pr, pi = table.unpack(py.eval('blr_probe3(N,rs_rad,fs_rad1,fs_rad2)',{N=N,rs_rad=rs_rad,fs_rad1=fs_rad1,fs_rad2=fs_rad2}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  return probe
end

function c:fzp(N,factor)
  local pr, pi = table.unpack(py.eval('fzp(N,fac)',{N=N,fac=factor}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  return probe
end

function c:random_fzp(N,factor,sections)
  local pr, pi = table.unpack(py.eval('random_fzp(N,fac,sec)',{N=N,fac=factor,sec=sections}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  return probe
end

function c:random_fzp2(N,factor,sections)
  local pr, pi = table.unpack(py.eval('random_fzp2(N,fac,sec)',{N=N,fac=factor,sec=sections}))
  -- plt:plot(pr)
  -- plt:plot(pi)
  local probe = torch.ZCudaTensor(pr:size()):copyRe(pr:cuda()):copyIm(pi:cuda())
  return probe
end

function c:get_detector_MTF_DQE(detector_type,binning,size)
  local mtf,dqe = table.unpack(py.eval('MTF_DQE_2D(t,b,s,path)',{t=detector_type,b=binning,s=size,path=u.script_path()}))
  return mtf,dqe
end

return c
