local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local zt = require 'ztorch'
require 'pprint'
local TruncatedPoissonLikelihood, parent = torch.class('znn.TruncatedPoissonLikelihood', 'nn.Criterion')

-- 1.Bian, L. et al. Fourier ptychographic reconstruction using Poisson maximum likelihood and truncated Wirtinger gradient. arXiv:1603.04746 [physics] (2016).
-- 1.Chen, Y. & Candes, E. J. Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems. arXiv:1505.05114 [cs, math, stat] (2015).

function TruncatedPoissonLikelihood:__init(a_h, a_lb, a_ub, gradInput, mask, buffer1, buffer2, z_buffer_real, K, No, Np, M, Nx, Ny, diagnostics, do_truncate)
   parent.__init(self)

   self.gradInput = gradInput
   self.lhs = buffer1
   self.rhs = buffer2
   self.valid_gradients = buffer1
   self.z_real = z_buffer_real
   self.mask = mask
   self.No = No
   self.Np = Np
   self.Nx = Nx
   self.Ny = Ny
   self.M = M
   self.K = K
   self.a_h = a_h
   self.a_lb = a_lb
   self.a_ub = a_ub
   self.diagnostics = diagnostics
   self.do_truncate = do_truncate
   self.output = torch.CudaTensor(1):fill(0)
   self.shifts = torch.FloatTensor(self.K,2)
end

function TruncatedPoissonLikelihood:updateOutput(z, I_target)
  -- plt:plot(z[1][1][1]:abs():float():log(),'in_psi abs')
  z:view_3D():fftBatched()
  self.I_model = self.z_real:normZ(z)

  -- sum over probe and object modes
  self.I_model = self.I_model:sum(2):sum(3)
  -- self.I_model_shifted = self.I_model:clone()
  -- plt:plot(self.I_model[1][1][1]:float(),'Im[1]')
  -- plt:plot(I_target[1]:float(),'I_target[1]')
  -- -- optical axis shift here
  -- local Im = torch.ZCudaTensor.new({self.K,self.M,self.M}):fillIm(0)
  -- local It = torch.ZCudaTensor.new({self.K,self.M,self.M}):fillIm(0)
  -- Im:copyRe(self.I_model)
  -- It:copyRe(I_target)
  --
  -- for i = 1, self.K do
  --   Im[i]:fftshift()
  --   It[i]:fftshift()
  -- end
  --
  -- -- plt:plot(Im[1]:abs():log():float(),'Im[i]')
  -- Im:fftBatched():conj()
  -- -- plt:plot(Im[1]:abs():log():float(),'Im[i]')
  -- It:fftBatched()
  -- local xcorr = Im:cmul(It)
  -- -- plt:plot(xcorr[1]:abs():log():float(),'xcorr[i]')
  -- xcorr:ifftBatched()
  -- -- plt:plot(xcorr[1]:abs():log():float(),'xcorr[i]')
  --
  --
  -- It:copyRe(I_target)
  -- for i = 1,self.K do
  --   local max1, imax1 = torch.max(xcorr[i]:re():fftshift(),1)
  --   local max2,imax2 = torch.max(max1,2)
  --   local imax = torch.FloatTensor{imax1[1][imax2[1][1]],imax2[1][1]}
  --   local max =  max2[1][1]
  --   self.shifts[i] = imax - self.M/2
  --   print(self.shifts[i])
  --   -- pprint(self.I_model_shifted)
  --   -- pprint(self.I_model)
  --   Im[i]:copyRe(self.I_model[i])
  --   Im[i]:fftshift()
  --   -- pprint(self.I_model_shifted[i][1])
  --   -- pprint(Im[i])
  --   self.I_model_shifted[i][1]:shift(Im[i]:re():view(table.unpack(self.I_model_shifted[i][1]:size():totable())),-self.shifts[i])
  --   -- self.I_model_shifted[i][1][1]:fftshift()
  --   -- print('imax')
  --   -- pprint(imax)
  --
  --   -- plt:plot(xcorr[i]:re():float(),'xcorr '..i)
    -- plt:plotcompare({self.I_model_shifted[i][1][1]:clone():add(-1,It[i]:re():fftshift()):float(),self.I_model[i][1][1]:clone():fftshift():add(-1,It[i]:re():fftshift()):float()},{'I_model_shifted - I_target '..i,'I_model - I_target '..i})
  for i = 1,3 do
    plt:plotcompare({self.I_model[i][1][1]:clone():fftshift():float(),I_target[i]:clone():fftshift():float()},{'I_model'..i,'I_model'..i})
  end
  -- end
  -- plt:plot(I_model[1][1][1]:float():log(),'I_model')
  local path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/'
  for i =1,10 do
    -- plt:plotcompare({self.I_model[i][1][1]:clone():fftshift():float(),I_target[i]:clone():fftshift():float()},{'I_model '..i,'I_target '..i},'',path .. string.format('%d_it%d_a',i,1),true)
  end
  -- local I_m = self.I_model:clone():log():cmul(I_target)
  -- local fac = self.I_model:clone():add(-1,I_m):cmul(self.mask)
  -- local L = fac:sum()
  -- print(L)

  self.I_model.THNN.TruncatedPoissonLikelihood_updateOutput(self.I_model:cdata(),I_target:cdata(),self.mask:cdata(),self.output:cdata())
  -- print(self.output[1])
  return self.output[1]
end

function TruncatedPoissonLikelihood:calculateXsi(z,I_target,it)
  if self.do_truncate then
    -- ||c - |Az|^2||_1 / (K*M*M)
    local K_t = self.rhs:add(I_target,-1,self.I_model):norm(1)/(I_target:nElement())
    local z_2norm = z:normall(2)/self.M/self.Np/self.No
    -- sqrt(n) * |a| / ( ||a|| ||z||) with ||a|| = n because DFT
    self.rhs:sqrt(self.I_model):div(z_2norm)
    -- u.printf('I_model 11 max,min: %g, %g',self.I_model:max(),self.I_model:min())

    -- calculate E_1, condition for denominator
    local E_1 = torch.ge(self.rhs,self.a_lb)
    self.lhs:le(self.rhs,self.a_ub)
    E_1:cmul(self.lhs)
    if self.diagnostics then
      u.printf('Percent filfilling condition     1: %2.2f',E_1:sum()/E_1:nElement()*100.0)
      -- local rhs = self.rhs:float()
      -- print(rhs:max(),rhs:min())
      plt:hist(self.rhs:float():view(self.rhs:nElement()),'rhs, cond 1')
      -- u.printf('K_t = %g',K_t)
      -- u.printf('a_h = %g',self.a_h)
      -- u.printf('K_t * a_h = %g',K_t*self.a_h)
    end
    -- plt:plot(E_1[1][1][1]:float(),'E1')

    self.rhs:mul(K_t*self.a_h)

    -- |c - |Az|^2|
    self.lhs:add(I_target,-1,self.I_model):abs()

    -- local ab = self.I_model[1][1][1]:clone():fftshift():float():log()
    -- local ak = I_target[1]:clone():fftshift():float():log()
    -- local DI = ab:clone():add(-1,ak)
    -- plt:plotcompare({ab,ak},{'I_model','I_target'})
    -- plt:plot(DI:float():log(),'DI')

    -- E_2, condition for numerator
    self.valid_gradients:le(self.lhs,self.rhs)
    if self.diagnostics then
      u.printf('Percent filfilling condition     2: %2.2f',self.valid_gradients:sum()/self.valid_gradients:nElement()*100.0)
      plt:hist(self.rhs:float():view(self.rhs:nElement()),'rhs')
      plt:hist(self.lhs:float():view(self.lhs:nElement()),'lhs')
    end

    -- E_1 && E_2
    self.valid_gradients:cmul(E_1)
    if self.diagnostics then
      -- u.printf('Percent filfilling condition 1 & 2: %2.2f',self.valid_gradients:sum()/self.valid_gradients:nElement()*100.0)
    end
  end
  -- in-place calculation of I_model <-- fm * (1 - I_target/I_model)
  self.I_model.THNN.TruncatedPoissonLikelihood_GradientFactor(self.I_model:cdata(),I_target:cdata(),self.mask:cdata())
  -- gradinput is set to z, thats why only in-place multiply
  -- z * 2 * fm * (1 - I_target/I_model)
  self.gradInput:cmul(self.I_model:expandAs(self.gradInput))
  -- print('it '..it)
  if it < 5 then
    self.gradInput:maskedFill(torch.gt(self.gradInput:abs(),30),zt.re(0)+zt.im(0))
  end
  local path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/'
  for i =80,120 do
    plt:plot(self.gradInput[i][1][1]:clone():fftshift():zfloat(),'gradInput '..i,path .. string.format('%d_it%d_gradInput',i,1),true)
  end
  self.gradInput:view_3D():ifftBatched()
  for i =100,110 do
    plt:plot(self.gradInput[i][1][1]:zfloat(),'gradInput '..i,path .. string.format('%d_it%d_gradInput',i,1),true)
  end
  return self.gradInput
end

function TruncatedPoissonLikelihood:updateGradInput(z, I_target,i)
  self.gradInput = self:calculateXsi(z,I_target,i)

  if self.do_truncate then
    local vg_expanded = self.valid_gradients:view(self.K,1,1,self.M,self.M):expandAs(self.gradInput)
    self.gradInput:cmul(vg_expanded)
  local path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/'
    for i =1,3 do
      plt:plot(self.gradInput[i][1][1]:clone():fftshift():zfloat(),'gradInput '..i,path .. string.format('%d_it%d_gradInput',i,1),true)
    end
  end
  return self.gradInput, self.valid_gradients:sum()
end

function TruncatedPoissonLikelihood:__call__(input, target)
--    self.output = self:forward(input, target)
--    self.gradInput = self:backward(input, target)
--    return self.output, self.gradInput
end
return TruncatedPoissonLikelihood
