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

-- mul_merge: 1 x Np x M x M
-- z_merge  ; No x Np x M x M
-- mul_split: 1 x Np x M x M
-- result   : No x Np x M x M
function engine:merge_and_split_pair(i,j,mul_merge, z_merge , mul_split, result)
  local O_tmp = self.O_tmp_PFstore
  local O_tmp_views = self.O_tmp_PF_views
  local mul_merge_shifted_conj = self.P_tmp3_PFstore
  local mul_split_shifted = self.P_tmp3_PFstore

  mul_merge_shifted_conj[1]:shift(mul_merge[1],self.dpos[j]):conj()
  O_tmp:zero()
  O_tmp_views[j]:add(result:cmul(z_merge,mul_merge_shifted_conj:expandAs(z_merge)):sum(self.P_dim))
  -- plt:plotReIm(self.O_tmp[1]:zfloat(),'merge_and_split_pair self.O_tmp')
  O_tmp:cmul(self.O_denom)
  -- plt:plotReIm(self.O_tmp[1]:zfloat(),'merge_and_split_pair self.O_tmp 2')
  -- plt:plotReIm(ov[1]:zfloat(),'merge_and_split_pair ov')
  mul_split_shifted[1]:shift(mul_split[1],self.dpos[i])
  result:cmul(O_tmp_views[i]:expandAs(result),mul_split_shifted:expandAs(result))
  return result
end

function engine:split_single(i,mul_split,result)
    local mul_split_shifted = self.P_tmp3_PFstore:zero()
    -- plt:plotReIm(ov[1]:zfloat(),'ov')
    mul_split_shifted[1]:shift(mul_split[1],self.dpos[i])
    -- plt:plotReIm(mul_split_shifted[1][1]:zfloat(),'mul_split_shifted')
    result:cmul(self.O_views[i]:expandAs(result),mul_split_shifted:expandAs(result))
    return result
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

-- buffers:
--  1 x sizeof(O) el C
--  6 x sizeof(z[k]) el C
-- free buffer: P_F
function engine:refine_positions()
  local H = torch.FloatTensor(2*self.K,2*self.K):zero()
  local overlaps = torch.FloatTensor(self.K,self.K)
  local H1 = H[{{1,self.K},{1,self.K}}]
  local H2 = H[{{self.K+1,2*self.K},{self.K+1,2*self.K}}]
  local Hx1 = H[{{1,self.K},{self.K+1,2*self.K}}]
  local Hx2 = H[{{self.K+1,2*self.K},{1,self.K}}]
  local b = torch.FloatTensor(2*self.K,1)
  local bv = b:view(b:nElement())

  self.P_Fz:zero()

  local zRy = self.zk_tmp1_PFstore
  local zRx = self.zk_tmp2_PFstore
  local Rx = self.P_tmp1_PFstore
  local Ry = self.P_tmp2_PFstore

  local r1 = self.zk_tmp5_PFstore
  local r2 = self.zk_tmp6_PFstore
  local r3 = self.zk_tmp7_PFstore
  local z_under = self.zk_tmp8_PFstore
  local z_i = self.zk_tmp8_PFstore

  local r4 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  local parts = (#self.batch_params) ^ 2
  local l = 0
  for _, params1 in ipairs(self.batch_params) do
    local batch_start1, batch_end1, batch_size1 = table.unpack(params1)
    -- print('params1')
    -- pprint(params1)
    for _, params2 in ipairs(self.batch_params) do
      -- print('params2')
      -- pprint(params2)
      local batch_start2, batch_end2, batch_size2 = table.unpack(params2)


      for i = 1, batch_size1 do
        local i_full = i + batch_start1 - 1
        self:maybe_copy_new_batch_z(batch_start1)
        self:maybe_copy_new_batch_P_Q(batch_start1)
        Rx[1]:dx(self.P[1],zRx[1],zRy[1])
        Ry[1]:dy(self.P[1],zRx[1],zRy[1])

        zRx = self:split_single(i_full,Rx,zRx)
        zRy = self:split_single(i_full,Ry,zRy)
        z_under:add(self.z[i],-1,self.P_Qz[i])

        bv[i_full] = z_under:dot(zRx).re
        bv[self.K + i_full] = z_under:dot(zRy).re
        z_i:copy(self.z[i])

        for j=1, batch_size2 do
          l = l + 1
          xlua.progress(l,self.K*self.K)
          local j_full = j + batch_start2 - 1
          self:maybe_copy_new_batch_z(batch_start2)
          local H1_ij = 0
          local H2_ij = 0
          local Hx_ij = 0
          if i_full == j_full then
            H1_ij = zRx:dot(zRx).re
            H2_ij = zRy:dot(zRy).re
            Hx_ij = zRx:dot(zRy).re
          end

          -- overlaps[{i,j}] = self:do_frames_overlap(i,j) and 1 or 0
          if self:do_frames_overlap(i,j) then
            local O11_ij = self:merge_and_split_pair(i_full,j_full,Rx,self.z[j],Rx,r1)
            local O22_ij = self:merge_and_split_pair(i_full,j_full,Ry,self.z[j],Ry,r2)
            local Ox_ij = self:merge_and_split_pair(i_full,j_full,Rx,self.z[j],Ry,r3)

            H1_ij = H1_ij - z_i:dot(O11_ij).re
            H2_ij = H2_ij - z_i:dot(O22_ij).re
            Hx_ij = Hx_ij - z_i:dot(Ox_ij).re
          end

          H1[{i_full,j_full}] = H1_ij
          H2[{i_full,j_full}] = H2_ij
          Hx1[{i_full,j_full}] = Hx_ij
          Hx2[{i_full,j_full}] = Hx_ij
        end -- end j
      end -- end i
    end -- end params2
  end -- end params1

  local ksi, LU = torch.gesv(b,H)
  -- plt:plot(H:log(),'H')
  local max,imax = torch.max(ksi,1)
  for i=1,self.K do
    local p = torch.FloatTensor{-ksi[i][1],-ksi[i+self.K][1]}
    -- u.printf('%04d : %g,%g',i,-ksi[i][1],-ksi[i+self.K][1])
    self.dpos[i]:add(p:clamp(-self.position_refinement_max_disp,self.position_refinement_max_disp))
  end

  -- local di = self.dpos:int()
  -- print('di')
  -- print(di)
  -- print('self.pos')
  -- print(self.pos)
  -- print('self.dpos')
  -- print(self.dpos)
  -- self.pos:add(di)
  -- self.dpos:add(-1,di:float())
  -- print('self.pos')
  -- print(self.pos)
  -- print('self.dpos')
  -- print(self.dpos)

  -- plt:scatter_positions(self.dpos:clone():add(self.pos:float()),self.dpos_solution:clone():add(self.pos:float()))
  if self.dpos_solution then
    local dp = self.dpos_solution:clone():add(-1,self.dpos):abs()
    local max_err = dp:max()
    local pos_err = self.dpos_solution:clone():add(-1,self.dpos):abs():sum()/self.K
    u.printf('ksi[%d] = %g, pos_error = %g, max_pos_error = %g', imax[1][1] , max[1][1] , pos_err, max_err)
  else
    u.printf('ksi[%d] = %g', imax[1][1] , max[1][1])
  end
  -- self:update_views()
  self:calculateO_denom()
  self:P_Q_plain()


  -- local answer=io.read()
end

-- z_underscore = [ I - P_Q ] z
-- buffers:
--  1 x sizeof(O) el C
--  7 x sizeof(P) el C
-- free buffer: P_F
function engine:refine_positions2()
  -- plt:plotReIm(z_under[1][1]:zfloat(),'z_under[i]')
  local H = torch.FloatTensor(2*self.K,2*self.K):zero()
  local overlaps = torch.FloatTensor(self.K,self.K)
  local H1 = H[{{1,self.K},{1,self.K}}]
  local H2 = H[{{self.K+1,2*self.K},{self.K+1,2*self.K}}]
  local Hx1 = H[{{1,self.K},{self.K+1,2*self.K}}]
  local Hx2 = H[{{self.K+1,2*self.K},{1,self.K}}]
  local b = torch.FloatTensor(2*self.K,1)
  local bv = b:view(b:nElement())

  self.P_Fz:zero()

  local zRy = self.zk_tmp1_PFstore
  local zRx = self.zk_tmp2_PFstore
  local Rx = self.P_tmp1_PFstore
  local Ry = self.P_tmp2_PFstore

  local r1 = self.zk_tmp5_PFstore
  local r2 = self.zk_tmp6_PFstore
  local r3 = self.zk_tmp7_PFstore
  local z_under = self.zk_tmp8_PFstore
  local r4 = torch.ZCudaTensor.new(self.Np,self.M,self.M)
  for i = 1, self.K do
    xlua.progress(i,self.K)
    Rx[1]:dx(self.P[1],zRx[1],zRy[1])
    Ry[1]:dy(self.P[1],zRx[1],zRy[1])

    zRx = self:split_single(i,Rx,zRx)
    zRy = self:split_single(i,Ry,zRy)
    z_under:add(self.z[i],-1,self.P_Qz[i])

    bv[i] = z_under:dot(zRx).re
    bv[self.K + i] = z_under:dot(zRy).re
    for j=1, self.K do
      local H1_ij = 0
      local H2_ij = 0
      local Hx_ij = 0
      if i == j then
        H1_ij = zRx:dot(zRx).re
        H2_ij = zRy:dot(zRy).re
        Hx_ij = zRx:dot(zRy).re
      end

      -- overlaps[{i,j}] = self:do_frames_overlap(i,j) and 1 or 0
      if self:do_frames_overlap(i,j) then

        local O11_ij = self:merge_and_split_pair(i,j,Rx,self.z[j],Rx,r1)
        local O22_ij = self:merge_and_split_pair(i,j,Ry,self.z[j],Ry,r2)
        local Ox_ij = self:merge_and_split_pair(i,j,Rx,self.z[j],Ry,r3)

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

  local ksi, LU = torch.gesv(b,H)
  local max,imax = torch.max(ksi,1)
  for i=1,self.K do
    local p = torch.FloatTensor{-ksi[i][1],-ksi[i+self.K][1]}
    -- u.printf('%04d : %g,%g',i,-ksi[i][1],-ksi[i+self.K][1])
    self.dpos[i]:add(p:clamp(-1,1))
  end

  plt:scatter_positions(self.dpos:clone():add(self.pos:float()),self.dpos_solution:clone():add(self.pos:float()))

  local dp = self.dpos_solution:clone():add(-1,self.dpos):abs()
  local max_err = dp:max()
  local pos_err = self.dpos_solution:clone():add(-1,self.dpos):abs():sum()/self.K

  self:calculateO_denom()
  self:P_Q_plain()

  u.printf('ksi[%d] = %g, pos_error = %g, max_pos_error = %g', imax[1][1] , max[1][1] , pos_err, max_err)
  -- local answer=io.read()
end

function engine:filter_object()
  pprint(self.O_tmp_PQstore)
  local O_fluence = self.O_tmp_PQstore:copy(self.O):normall(2)
  print('filtering object')
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P',self.save_path .. self.i..' P',false)

  for o = 1, self.No do
    self.O[o]:fftBatched()
  end

  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft unfiltered ',self.save_path .. self.i..' P filtered ',false)
  self.O:cmul(self.object_highpass)
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft filtered ',self.save_path .. self.i..' P filtered ',false)
  for o = 1, self.No do
    self.O[o]:ifftBatched()
  end

  local O_fluence_new = self.O_tmp_PQstore:copy(self.O):normall(2)
  u.printf('O_fluence/O_fluence_new = %g',O_fluence/O_fluence_new)
  self.O:mul(O_fluence/O_fluence_new)
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P filtered ',self.save_path .. self.i..' P filtered ',false)
  -- -- self:calculateO_denom()
  -- -- self:merge_frames(self.P,self.O,self.O_views)
end
function engine:merge_frames(mul_merge, merge_memory, merge_memory_views)
  self:merge_frames_internal(self.z, mul_merge, merge_memory, merge_memory_views, self.zk_tmp1_PQstore, self.P_tmp1_PQstore, true)
end

-- buffers:
--  0 x sizeof(P) el R
--  1 x sizeof(z[k]) el C
function engine:merge_frames_internal(frames, mul_merge, merge_memory, merge_memory_views, product_shifted_buffer, mul_merge_shifted_buffer, do_normalize_merge_memory)
  -- print('merge_frames')
  local z = frames
  local product_shifted = product_shifted_buffer
  local mul_merge_shifted = mul_merge_shifted_buffer
  merge_memory:mul(self.object_inertia)
  local pos = torch.FloatTensor{1,1}
  u.printram('before merge_frames')

  for k, view in ipairs(merge_memory_views) do
    if self.batches > 2 then xlua.progress(k,self.K) end
    pos:fill(1):cmul(self.dpos[k])
    self:maybe_copy_new_batch_z(k)
    local ind = self.k_to_batch_index[k]
    -- print(k,ind)
    mul_merge_shifted[1]:shift(mul_merge[1],pos)
    mul_merge_shifted[1]:conj()
    product_shifted = product_shifted:cmul(z[ind],mul_merge_shifted:expandAs(z[ind])):sum(self.P_dim)
    view:add(product_shifted)
  end
  -- plt:plot(self.O_denom[1][1]:float():log(),'O_denom')
  -- plt:plot(merge_memory[1][1]:zfloat(),'merge_memory')
  if do_normalize_merge_memory then
    merge_memory:cmul(self.O_denom)
  end

  -- O_norm = self.O_tmp_PQstore:norm(merge_memory):sum()
  -- u.printf('object norm: %g',O_norm)
  -- plt:plot(merge_memory[1][1]:zfloat(),'merge_memory 2')
  u.printram('after merge_frames')
end

-- buffers:
--  0 x sizeof(P) el R
--  1 x sizeof(P) el C
function engine:update_frames(z,mul_split,merge_memory_views,batch_copy_func)
  -- print('update_frames')
  local mul_split_shifted = self.zk_buffer_update_frames
  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(merge_memory_views) do
    if self.batches > 2 then xlua.progress(k,self.K) end
    batch_copy_func(self,k)
    local ind = self.k_to_batch_index[k]
    -- print(k,ind)
    pos:fill(1):cmul(self.dpos[k])
    -- mul_split_shifted[1]:shift(mul_split[1],pos)
    for i = 1, self.No do
      mul_split_shifted[i]:shift(mul_split[i],pos)
    end
    local view_exp = view:expandAs(z[ind])
    z[ind]:cmul(mul_split_shifted,view_exp)
    -- plt:plot(self.P_Qz[ind][1][1]:zfloat(),'self.P_Qz[ind]')
  end
  -- plt:plot(z[1][1][1]:zfloat(),'z[1]')
  -- plt:plot(z[2][1][1]:zfloat(),'z[2]')
  -- plt:plot(z[3][1][1]:zfloat(),'z[3]')
  -- plt:plot(z[4][1][1]:zfloat(),'z[4]')
  -- self:maybe_copy_new_batch_P_Q(1)
  u.printram('after update_frames')
end

-- del_xf = U.delxf(x,axis=ax0)
-- del_yf = U.delxf(x,axis=ax1)
-- del_xb = U.delxb(x,axis=ax0)
-- del_yb = U.delxb(x,axis=ax1)
--
-- self.delxy = [del_xf, del_yf, del_xb, del_yb]
-- self.g = 2 * self.amplitude*(del_xb + del_yb - del_xf - del_yf)
function engine:Del_regularize(target,amplitude,tmp,result)
  result:zero()
  local d = tmp

  d:dx_bw(target)
  result:add(d)

  d:dy_bw(target)
  result:add(d)

  d:dx_fw(target)
  result:add(-1,d)

  d:dy_fw(target)
  result:add(-1,d)

  result:mul(2*amplitude*self.rescale_regul_amplitude)
  return result
end

function engine:TV_regularize(target,amplitude,tmp,tmp_real,tmp_real2,result)
  local d_abs = tmp_real
  local d = tmp
  local mask = tmp_real2
  local eps = 1e-10

  result:zero()

  d:dx_bw(target)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  result:add(d)

  d:dy_bw(target)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  result:add(d)

  d:dx_fw(target)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  result:add(-1,d)

  d:dy_fw(target)
  d_abs:absZ(d)
  mask:lt(d_abs,eps)
  d_abs:maskedFill(mask,eps)
  d:cdiv(d_abs)
  result:add(-1,d)

  result:mul(amplitude)

  return result
end

function engine:regularize_probe()
  print('regularizing probe')
  local regul = self:Del_regularize(self.P,self.probe_regularization_amplitude(self.i),self.P_tmp3_PQstore,self.P_tmp1_PQstore)
 local title = self.i..' regul'
 local title1 = self.i..'_new_P regul'
 -- plt:plot(regul[1][1]:zfloat(),title,self.save_path ..title,false)
 -- plt:plot(self.P[1][1]:zfloat(),self.i..'new_P ',self.save_path .. self.i..' new_P ',false)
 self.P:add(regul)
 -- plt:plot(self.P[1][1]:zfloat(),title1,self.save_path ..title1,false)
 -- self:calculateO_denom()
 -- self:merge_frames(self.P,self.O,self.O_views)
end

function engine:filter_probe()
  local P_fluence = self.P_tmp1_real_PQstore:normZ(self.P):sum()
  print('filtering probe')
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P',self.save_path .. self.i..' P',false)
  self.P[1]:fftBatched()
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft unfiltered ',self.save_path .. self.i..' P filtered ',false)
  self.P:cmul(self.probe_lowpass)
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft filtered ',self.save_path .. self.i..' P filtered ',false)
  self.P[1]:ifftBatched()
  local P_fluence_new = self.P_tmp1_real_PQstore:normZ(self.P):sum()
  u.printf('P_fluence/P_fluence_new = %g',P_fluence/P_fluence_new)
  self.P:mul(P_fluence/P_fluence_new)
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P filtered ',self.save_path .. self.i..' P filtered ',false)
  -- self:calculateO_denom()
  -- self:merge_frames(self.P,self.O,self.O_views)
end

-- buffers:
--  3 x sizeof(z[k]) el C
--  2 x sizeof(P) el C
--  1 x sizeof(P) el R
function engine:refine_probe()
  -- print('refine_probe')
  local new_P = self.P_tmp1_PQstore
  local oview_conj = self.zk_tmp1_PQstore
  local oview_conj_shifted = self.zk_tmp2_PQstore

  local dP = self.P_tmp3_PQstore
  local dP_abs = self.P_tmp3_real_PQstore

  local new_P_denom = self.P_tmp1_real_PQstore
  local denom_shifted = self.P_tmp2_real_PQstore
  local denom_tmp = self.zk_tmp1_real_PQstore

  new_P:mul(self.P,self.probe_inertia)
  new_P_denom:fill(self.probe_inertia)

  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(self.O_views) do
    self:maybe_copy_new_batch_z(k)
    local ind = self.k_to_batch_index[k]
    denom_shifted:zero()
    pos:fill(-1):cmul(self.dpos[k])
    oview_conj:conj(view:expandAs(self.z[ind]))

    denom_tmp = denom_tmp:normZ(oview_conj):sum(self.O_dim)
    denom_shifted[1]:shift(denom_tmp[1],pos)

    oview_conj:cmul(self.z[ind])
    for o = 1, self.No do
      oview_conj_shifted[o]:shift(oview_conj[o],pos)
    end

    new_P_denom:add(denom_shifted)
    new_P:add(oview_conj_shifted:sum(self.O_dim))
  end
  new_P:cdiv(new_P_denom)
  local probe_change = dP_abs:normZ(dP:add(new_P,-1,self.P)):sum()
  local P_norm = dP_abs:normZ(self.P):sum()
  u.printf('total current: %g, max_pow/total_current = %g',P_norm,self.max_power/P_norm)
  -- new_P:mul(self.max_power/P_norm)
  -- local regul = self:TV_regularize(new_P,self.probe_regularization_amplitude(self.i),dP,dP_abs,new_P_denom,self.P)
  -- local regul = self:Del_regularize(new_P,self.probe_regularization_amplitude(self.i),dP,self.P)
  --
  -- local title = self.i..'regul'
  -- local title1 = self.i..'_new_P regul'
  -- plt:plot(regul[1][1]:zfloat(),title,self.save_path ..title)
  -- plt:plot(new_P[1][1]:zfloat(),'new_P '..self.i,self.save_path ..'new_P '..self.i)
  --
  -- new_P:add(regul)
  --
  -- plt:plot(new_P[1][1]:zfloat(),title1,self.save_path ..title1)

  self.P:copy(new_P)
  if self.probe_support then self.P = self.support:forward(self.P) end

  -- self.P[1]:fftBatched()
  -- self.P = self.bandwidth_limit:forward(self.P)
  -- self.P[1]:ifftBatched()

  -- plt:plot(self.P[1]:zfloat(),'self.P')
  if self.probe_regularization_amplitude(self.i) then self:regularize_probe() end
  if self.probe_lowpass_fwhm(self.i) then self:filter_probe() end

  self:calculateO_denom()
  u.printram('after refine_probe')
  return math.sqrt(probe_change/P_norm/self.Np)
end

-- recalculate (Q*Q)^-1
-- buffers:
--  2 x sizeof(P) el R
function engine:calculateO_denom()
  self.O_denom:fill(self.object_inertia)
  local norm_P_shifted = self.a_tmp_real_PQstore
  local norm_P =  self.P_tmp2_real_PQstore:normZ(self.P):sum(self.P_dim)
  local tmp = self.P_tmp2_real_PQstore
  -- plt:plot(norm_P[1]:float(),'norm_P - calculateO_denom')
  for k,view in ipairs(self.O_denom_views) do
    -- pprint(norm_P_shifted[1])
    -- pprint(norm_P[1])
    norm_P_shifted[1]:shift(norm_P[1],self.dpos[k])
    local np_exp = norm_P_shifted:expandAs(view)
    view:add(np_exp)
  end

  local abs_max = tmp:absZ(self.P):max()
  local fact_start, fact_end = 1e-6, 1e-9

  local fact = fact_start-(self.i/self.iterations)*(fact_start-fact_end)
  local sigma = abs_max * abs_max * fact
  -- print('sigma = '..sigma)
  -- plt:plot(self.O_denom[1]:float():log(),'calculateO_denom  self.O_denom')
  self.InvSigma(self.O_denom,sigma)
  -- plt:plot(self.O_denom[1][1]:float():log(),'calculateO_denom  self.O_denom 2')
  self.O_mask = self.O_denom:lt(1e-3)
  u.printram('after calculateO_denom')
end

return engine
