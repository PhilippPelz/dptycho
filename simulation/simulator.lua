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

function c:exitwaves_multislice(pos,in_psi, view_size, E)
  self.physics = u.physics(E)
  local views = s.create_views(self.v, pos, view_size,1)
  local P0 = prop.fresnel(view_size,self.dx,self.dz,self.physics.lambda)
  P0:fftshift()
  plt:plot(P0:zfloat())
  local out_psi = in_psi:view(1,view_size,view_size):repeatTensor(pos:size(1),1,1)
  local P = P0:view(1,view_size,view_size):expandAs(out_psi)
  for slice = 1, self.v:size(1) do
    for i, view in ipairs(views) do
      out_psi[i]:cmul(view[slice])
    end
    out_psi:fftBatched()
    out_psi:cmul(P)
    out_psi:ifftBatched()
  end
  for i, view in ipairs(views) do
    plt:plot(out_psi[i],'out_psi '..i)
  end

  return out_psi
end

function c:exitwaves_projected(pos,in_psi, view_size, E)
  self.physics = u.physics(E)
  local v_proj = self.v:sum(1)[1]
  plt:plot(v_proj,'proj potential')
  local views = s.create_views(v_proj, pos, view_size,0)
  local out_psi = in_psi:view(1,view_size,view_size):repeatTensor(pos:size(1),1,1)
  for i, view in ipairs(views) do
    out_psi[i]:cmul(view)
    -- plt:plot(out_psi[i],'out_psi[i]')
  end
  return out_psi
end

function c:dp_multislice(pos,in_psi, view_size, binning, E, total_dose)
  local out_psi = self:exitwaves_multislice(pos,in_psi, view_size, E)
  local mtf, dqe = self:get_detector_MTF_DQE('K2',binning,view_size)
  out_psi:fftBatched()
  local I_noisy = self:simulate_detector(out_psi,mtf:cuda(),dqe:cuda(),total_dose)
  return I_noisy
end

function c:dp_projected(pos,in_psi, view_size, binning, E, total_dose)
  local out_psi = self:exitwaves_projected(pos,in_psi, view_size, E)
  local mtf, dqe = self:get_detector_MTF_DQE('K2',binning,view_size)
  out_psi:fftBatched()
  local I_noisy = self:simulate_detector(out_psi,mtf:cuda(),dqe:cuda(),total_dose)
  return I_noisy
end

function c:simulate_detector(in_psi, mtf, dqe, total_dose)
  local mtf_exp = mtf:view(1,in_psi:size(2),in_psi:size(3)):expandAs(in_psi)
  local dqe_exp = dqe:view(1,in_psi:size(2),in_psi:size(3)):expandAs(in_psi)
  local I = in_psi:norm()
  -- print(I:normall(1))
  for i=1,in_psi:size(1) do
    plt:plot(I[i]:abs():fftshift():float():log(),string.format('I[%d]',i))
  end
  I = I:div(I:normall(1))
  local dose_per_exposure = total_dose / I:size(1)
  local ftI = I:fftBatched()
  local nnps = mtf_exp:clone():pow(2):cdiv(dqe_exp)
  local Sn = ftI:cmul(mtf_exp):cdiv(nnps:sqrt())
  local I2 = Sn:ifftBatched():real()
  local I2norm = I2:div(I2:norm(1))
  local I2 = I2norm:mul(dose_per_exposure):float()
  local I_noise = u.stats.poisson(I2)
  return I_noise -- I_noise is float tensor
end

function c:get_positions_raster(Npos,size)
  return py.eval('raster_positions(n,s)',{n=Npos,s=size}):float()
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

function c:get_detector_MTF_DQE(detector_type,binning,size)
  local mtf,dqe = table.unpack(py.eval('MTF_DQE_2D(t,b,s,path)',{t=detector_type,b=binning,s=size,path=u.script_path()}))
  return mtf,dqe
end

return c
