local classic = require 'classic'
local znn = require 'dptycho.znn'
local nn = require 'nn'
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local plt = plot()
local zt = require "ztorch.fcomplex"
local pprint = require "pprint"
local base_engine = require "dptycho.core.ptycho.base_engine"
local z = require "ztorch.fcomplex"
local xlua = require "xlua"
local gnuplot = require "gnuplot"

local engine, super = classic.class(...,base_engine)

function engine:_init(par)
  super._init(self,par)
end

function engine:merge_and_split_pair(i,j,mul_merge, z_merge , mul_split, result)
  local O_tmp = self.O_tmp_PFstore
  local O_tmp_views = self.O_tmp_PF_views

  result:shift(mul_merge,self.dpos[j]):conj()

  O_tmp:zero()
  O_tmp_views[j]:add(result:cmul(z_merge):sum(1))
  -- plt:plotReIm(self.O_tmp[1]:zfloat(),'merge_and_split_pair self.O_tmp')
  O_tmp:cmul(self.O_denom)
  -- plt:plotReIm(self.O_tmp[1]:zfloat(),'merge_and_split_pair self.O_tmp 2')
  local ov = O_tmp_views[i]:repeatTensor(self.Np,1,1)
  -- plt:plotReIm(ov[1]:zfloat(),'merge_and_split_pair ov')
  result:shift(mul_split,self.dpos[i])
  result:cmul(ov)
  -- plt:plotReIm(mul_split[1]:zfloat(),'merge_and_split_pair mul_split')
  return result
end

function engine:split_single(i,mul_split,tmp)
    local mul_split_shifted = self.P_tmp8_PFstore
    local ov = self.O_views[i]:repeatTensor(self.Np,1,1)
    -- plt:plotReIm(ov[1]:zfloat(),'ov')
    mul_split_shifted:shift(mul_split,self.dpos[i])
    mul_split_shifted:cmul(ov)
    -- plt:plotReIm(mul_split[1]:zfloat(),'mul_split')
    return mul_split:copy(mul_split_shifted)
end

function engine:do_frames_overlap(i,j)
  local pos_i = self.pos[i]
  local pos_j = self.pos[j]
  local beam_fraction = 0.6
  local beam_offset = (1-beam_fraction)*self.M/2
  local beam_size = beam_fraction*self.M
  local x_lt, x_gt = math.min(pos_i[1],pos_j[1]),math.max(pos_i[1],pos_j[1])
  local y_lt, y_gt = math.min(pos_i[2],pos_j[2]),math.max(pos_i[2],pos_j[2])
  x_lt = x_lt + beam_offset
  x_gt = x_gt + beam_offset
  y_lt = y_lt + beam_offset
  y_gt = y_gt + beam_offset
  local x_gt_within_x_lt_beam = x_gt < x_lt + beam_size
  local y_gt_within_y_lt_beam = y_gt < y_lt + beam_size
  local i_and_j_overlap = x_gt_within_x_lt_beam and y_gt_within_y_lt_beam
  return i_and_j_overlap
end

-- z_underscore = [ I - P_Q ] z
-- buffers:
--  1 x sizeof(O) el C
--  7 x sizeof(P) el C
-- free buffer: P_F
function engine:refine_positions()
  -- plt:plotReIm(z_under[1][1]:zfloat(),'z_under[i]')
  local H = torch.FloatTensor(2*self.K,2*self.K):zero()
  local overlaps = torch.FloatTensor(self.K,self.K)
  local H1 = H[{{1,self.K},{1,self.K}}]
  local H2 = H[{{self.K+1,2*self.K},{self.K+1,2*self.K}}]
  local Hx1 = H[{{1,self.K},{self.K+1,2*self.K}}]
  local Hx2 = H[{{self.K+1,2*self.K},{1,self.K}}]
  local b = torch.FloatTensor(2*self.K,1)
  local bv = b:view(b:nElement())

  -- self.tmp = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local zRy = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local zRx = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local P3 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local Rx = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local Ry = torch.ZCudaTensor.new(self.Np,self.M,self.M)

  local r1 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local r2 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local r3 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local z_under = torch.ZCudaTensor.new(self.Np,self.M,self.M)

  -- self.P_Fz:zero()

  -- local zRy = self.P_tmp1_PFstore
  -- local zRx = self.P_tmp2_PFstore
  -- local Rx = self.P_tmp3_PFstore
  -- local Ry = self.P_tmp4_PFstore
  --
  -- local r1 = self.P_tmp5_PFstore
  -- local r2 = self.P_tmp6_PFstore
  -- local r3 = self.P_tmp7_PFstore
  -- local z_under = self.P_tmp9_PFstore
  -- local r4 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  for i = 1, self.K do
    xlua.progress(i,self.K)
    Rx:dx(self.P,zRx,zRy)
    Ry:dy(self.P,zRx,zRy)
    -- zS11:dx2(self.P,zRx,zRy)
    -- zS22:dy2(self.P,zRx,zRy)
    -- zSx:dxdy(self.P,zRx,zRy,r1,r2)
    zRx:copy(Rx)
    zRy:copy(Ry)

    zRx = self:split_single(i,zRx,r1)
    zRy = self:split_single(i,zRy,r1)

    z_under:add(self.z[i],-1,self.P_Qz[i])

    bv[i] = z_under:dot(zRx).re
    bv[self.K + i] = z_under:dot(zRy).re
    if i == 1 then
      -- plt:plotReIm(z_under[i][1]:zfloat(),'z_under[i]')
      -- plt:plotReIm(Rx[1]:zfloat(),'Rx')
      -- plt:plotReIm(Ry[1]:zfloat(),'Ry')
      -- plt:plotReIm(zRx[1]:zfloat(),'zRx')
      -- plt:plotReIm(zRy[1]:zfloat(),'zRy')
      -- u.printf('bv[%d] = %f,    bv[%d] = %f',i,bv[i],self.K + i,bv[self.K + i])
    end
    for j=1, self.K do
      local H1_ij = 0
      local H2_ij = 0
      local Hx_ij = 0
      if i == j then
        H1_ij = zRx:dot(zRx).re
        H2_ij = zRy:dot(zRy).re
        Hx_ij = zRx:dot(zRy).re
      end

      -- u.printf('%3d overlap %3d: %s',i,j,tostring(self:do_frames_overlap(i,j)))
      -- overlaps[{i,j}] = self:do_frames_overlap(i,j) and 1 or 0
      if self:do_frames_overlap(i,j) then

        local O11_ij = self:merge_and_split_pair(i,j,Rx,self.z[j],Rx,r1)
        local O22_ij = self:merge_and_split_pair(i,j,Ry,self.z[j],Ry,r2)
        local Ox_ij = self:merge_and_split_pair(i,j,Rx,self.z[j],Ry,r3)

        -- if i == 1 then
        --   plt:plotReIm(O11_ij[1]:zfloat(),'O11_ij')
        --   plt:plotReIm(O22_ij[1]:zfloat(),'O22_ij')
        --   plt:plotReIm(Ox_ij[1]:zfloat(),'Ox_ij')
        -- end

        H1_ij = H1_ij - self.z[i]:dot(O11_ij).re
        H2_ij = H2_ij - self.z[i]:dot(O22_ij).re
        Hx_ij = Hx_ij - self.z[i]:dot(Ox_ij).re
      end
      H1[{i,j}] = H1_ij
      H2[{i,j}] = H2_ij
      Hx1[{i,j}] = Hx_ij
      Hx2[{i,j}] = Hx_ij
    end
  end
  -- plt:plot(overlaps,'overlaps')
  -- plt:plot(H:clone():log(),'H')
  -- plt:plot(b,'b')
  -- gnuplot.plot('b',b:clone(),'+')
  -- gnuplot.plot('H[1]',H[1]:clone(),'+')

  local ksi, LU = torch.gesv(b,H)
  -- pprint(ksi)
  -- self.dpos:zero()
  for i=1,self.K do
    local p = torch.FloatTensor{-ksi[i][1],-ksi[i+self.K][1]}
    u.printf('%04d : %g,%g',i,-ksi[i][1],-ksi[i+self.K][1])
    self.dpos[i]:add(p:clamp(-5,5))
  end
  -- print(self.ksi)
  -- self.dpos:add(self.ksi:mul(-1))
  -- print(self.dpos_solution)
  local dp = self.dpos_solution:clone():add(-1,self.dpos):abs()
  local max_err = dp:max()
  local pos_err = self.dpos_solution:clone():add(-1,self.dpos):abs():sum()/self.K
  self:calculateO_denom()
  self:update_O(self.z)
  self:update_z_from_O(self.P_Qz)
  -- print(self.dpos)
  local max,imax = torch.max(ksi,1)
  u.printf('ksi[%d] = %g, pos_error = %g, max_pos_error = %g', imax[1][1] , max[1][1] , pos_err, max_err)
  local answer=io.read()
end

function engine:prepare_P_ksi(P,k)
  P:add(self.ksi[k][1],self.dxP):add(self.ksi[k][2],self.dyP)
  P:add(self.ksi[k][1]*self.ksi[k][2],self.dxdyP)
  P:add(0.5*self.ksi[k][1]^2,self.dx2P)
  P:add(0.5*self.ksi[k][2]^2,self.dy2P)
  -- plt:plotReIm(P[1]:zfloat(),'P')
  return P
end

-- buffers:
--  0 x sizeof(P) el R
--  1 x sizeof(P) el C
function engine:merge_frames(z, mul_merge, merge_memory, merge_memory_views)
  -- print('merge_frames shifted')
  local product_shifted = self.P_tmp2_PQstore
  merge_memory:mul(self.object_inertia)
  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(merge_memory_views) do
    pos:fill(1):cmul(self.dpos[k])
    -- plt:plot(product_shifted[1]:zfloat(),'product')
    product_shifted:shift(mul_merge,pos)
    product_shifted = product_shifted:conj()
    product_shifted = product_shifted:cmul(z[k]):sum(1)
    -- if self.i == 5 then
    --   plt:plotReIm(product_shifted[1]:zfloat(),'product_shifted')
    -- end
    -- plt:plot(product_shifted[1]:zfloat(),'product_shifted')
    view:add(product_shifted[1])
  end
  -- if self.i == 5 then
  --   plt:plotReIm(self.O[1]:zfloat(),'O')
  -- end
  self.O:cmul(self.O_denom)
  -- if self.i == 5 then
  --   plt:plotReIm(self.O[1]:zfloat(),'O')
  -- end
end

-- buffers:
--  0 x sizeof(P) el R
--  1 x sizeof(P) el C
function engine:update_frames(z,mul_split,merge_memory_views)
  -- print('update_frames shifted')
  local mul_split_shifted = self.P_tmp1_PFstore
  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(merge_memory_views) do
    local ov = view:repeatTensor(self.Np,1,1)
    pos:fill(1):cmul(self.dpos[k])
    -- print(pos)
    mul_split_shifted:shift(mul_split,pos)
    -- plt:plot(mul_split_shifted[1]:zfloat(),'mul_split_shifted')
    z[k]:copy(ov):cmul(mul_split_shifted)
  end
end

-- buffers:
--  3 x sizeof(P) el R
--  3 x sizeof(P) el C
function engine:refine_probe()
  -- print('refine_probe shifted '..self.i)
  local new_probe = self.P_tmp1_PQstore
  local oview_conj = self.P_tmp2_PQstore
  local oview_conj_shifted = self.P_tmp3_PQstore

  local dP = self.P_tmp3_PQstore
  local dP_abs = self.P_tmp3_real_PQstore

  local denom = self.P_tmp1_real_PQstore
  local denom_shifted = self.P_tmp2_real_PQstore
  local tmp = self.P_tmp3_real_PQstore

  new_probe:mul(self.P,self.probe_inertia)
  denom:fill(self.probe_inertia)
  local pos = torch.FloatTensor{1,1}

  for k, view in ipairs(self.O_views) do
    denom_shifted:zero()
    pos:fill(-1):cmul(self.dpos[k])
    local ovk = view:repeatTensor(self.Np,1,1)
    oview_conj:conj(ovk)
    tmp = tmp:normZ(oview_conj):sum(1)
    oview_conj:cmul(self.z[k])
    oview_conj_shifted:shift(oview_conj,pos)
    denom_shifted:shift(tmp,pos)

    denom:add(denom_shifted)
    new_probe:add(oview_conj_shifted)
  end
  -- plt:plot(denom[1]:float(),'denom')
  -- plt:plot(new_probe[1]:zfloat(),'new_probe finish 1')
  new_probe:cdiv(denom)
  -- plt:plot(new_probe[1]:zfloat(),'new_probe finish 2')
  -- new_probe = self.support:forward(new_probe)
  dP:add(new_probe,-1,self.P)
  dP_abs:absZ(dP)
  local probe_change = dP_abs:sum()
  self.P:copy(new_probe)
  -- plt:plot(self.P[1]:zfloat(),'self.P')
  self:calculateO_denom()
  return probe_change
end

-- recalculate (Q*Q)^-1
-- buffers:
--  2 x sizeof(P) el R
function engine:calculateO_denom()
  -- print('calculateO_denom shifted') *self.K
  self.O_denom:fill(self.object_inertia)
  local norm_P_shifted = self.P_tmp1_real_PQstore
  local norm_P =  self.P_tmp2_real_PQstore:normZ(self.P):sum(1)
  local tmp = self.P_tmp2_real_PQstore
  -- plt:plot(norm_P[1]:float(),'norm_P - calculateO_denom')
  local pos = torch.FloatTensor{1,1}
  for k,view in ipairs(self.O_denom_views) do
    pos:fill(1):cmul(self.dpos[k])
    norm_P_shifted:shift(norm_P,pos)
    view:add(norm_P_shifted)
    -- if self.i == 5 and k > 83 then
    --   plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
    -- end
  end

  local abs_max = tmp:absZ(self.P):max()
  -- local sigma = abs_max * abs_max * 1e-4
  local sigma = 1e-6
  -- print('sigma = '..sigma)
  -- plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
  self.InvSigma(self.O_denom,sigma)
  -- plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom 2')
  self.O_mask = self.O_denom:lt(1e-3)
end

return engine
